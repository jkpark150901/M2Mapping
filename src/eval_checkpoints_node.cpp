#include <filesystem>
#include <iostream>
#include <set>
#include <string>
#include <vector>

#include "neural_mapping/neural_mapping.h"

#define BACKWARD_HAS_DW 1
#include "backward.hpp"
namespace backward {
backward::SignalHandling sh;
}

static void print_usage(const char *prog) {
  std::cerr
      << "Usage: " << prog
      << " <output_dir> [options]\n"
         "\n"
         "  <output_dir>   Training output directory (contains "
         "local_map_checkpoint.pt,\n"
         "                 config/scene/config.yaml, checkpoints/iter_XXXXXX/)\n"
         "\n"
         "Options:\n"
         "  --all          Evaluate all iter checkpoints found under "
         "checkpoints/\n"
         "  --iters N,...  Comma-separated list of iterations to evaluate\n"
         "                 (e.g. --iters 1000,5000,10000)\n"
         "  --render       Run render_path + eval_render\n"
         "  --mesh         Run save_mesh + eval_mesh\n"
         "\n"
         "Example:\n"
         "  eval_checkpoints_node /results/iae --all --render --mesh\n"
         "  eval_checkpoints_node /results/iae --iters 5000,10000 --mesh\n";
}

static std::vector<int> parse_iters(const std::string &s) {
  std::vector<int> out;
  std::string tok;
  for (char c : s + ',') {
    if (c == ',') {
      if (!tok.empty()) { out.push_back(std::stoi(tok)); tok.clear(); }
    } else {
      tok += c;
    }
  }
  return out;
}

// Scan checkpoints/ directory and collect available iter numbers.
static std::vector<int> scan_checkpoint_iters(const std::filesystem::path &output_dir) {
  std::set<int> iters;
  auto ckpt_root = output_dir / "checkpoints";
  if (!std::filesystem::is_directory(ckpt_root)) return {};
  for (const auto &entry : std::filesystem::directory_iterator(ckpt_root)) {
    if (!entry.is_directory()) continue;
    const std::string name = entry.path().filename().string();
    // Expect "iter_XXXXXX"
    if (name.size() > 5 && name.substr(0, 5) == "iter_") {
      try { iters.insert(std::stoi(name.substr(5))); } catch (...) {}
    }
  }
  return {iters.begin(), iters.end()};
}

int main(int argc, char **argv) {
  if (argc < 2) { print_usage(argv[0]); return 1; }

  torch::manual_seed(0);
  torch::cuda::manual_seed_all(0);

  std::filesystem::path output_dir = argv[1];
  bool do_all    = false;
  bool do_render = false;
  bool do_mesh   = false;
  std::vector<int> requested_iters;

  for (int i = 2; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--all") {
      do_all = true;
    } else if (arg == "--render") {
      do_render = true;
    } else if (arg == "--mesh") {
      do_mesh = true;
    } else if (arg == "--iters" && i + 1 < argc) {
      requested_iters = parse_iters(argv[++i]);
    } else {
      std::cerr << "Unknown option: " << arg << '\n';
      print_usage(argv[0]);
      return 1;
    }
  }

  if (!do_render && !do_mesh) {
    std::cerr << "Error: specify at least one of --render or --mesh\n";
    print_usage(argv[0]);
    return 1;
  }

  if (!do_all && requested_iters.empty()) {
    std::cerr << "Error: specify --all or --iters N,...\n";
    print_usage(argv[0]);
    return 1;
  }

  // Discover checkpoints
  std::vector<int> eval_iters;
  if (do_all) {
    eval_iters = scan_checkpoint_iters(output_dir);
    if (eval_iters.empty()) {
      std::cerr << "No checkpoints found under " << output_dir / "checkpoints" << '\n';
      return 1;
    }
    std::cout << "Found " << eval_iters.size() << " checkpoint(s): ";
    for (int it : eval_iters) std::cout << it << ' ';
    std::cout << '\n';
  } else {
    eval_iters = requested_iters;
  }

  // Validate checkpoint files exist
  auto ckpt_root = output_dir / "checkpoints";
  for (int it : eval_iters) {
    char buf[16];
    std::snprintf(buf, sizeof(buf), "iter_%06d", it);
    auto path = ckpt_root / buf / "local_map_checkpoint.pt";
    if (!std::filesystem::exists(path)) {
      std::cerr << "Missing checkpoint: " << path << '\n';
      return 1;
    }
  }

  // Initialize NeuralSLAM in view mode (mode=0).
  // Constructor expects config at: output_dir/config/scene/config.yaml
  auto config_path = output_dir / "config" / "scene" / "config.yaml";
  if (!std::filesystem::exists(config_path)) {
    std::cerr << "Config not found: " << config_path << '\n';
    return 1;
  }

  try {
    auto slam = std::make_shared<NeuralSLAM>(0, config_path);

    for (int it : eval_iters) {
      char buf[16];
      std::snprintf(buf, sizeof(buf), "iter_%06d", it);
      auto ckpt_path = ckpt_root / buf;
      auto out_path  = output_dir / "eval" / buf;

      std::cout << "\n[eval] iter=" << it
                << "  ckpt=" << ckpt_path
                << "  out=" << out_path << '\n';

      slam->eval_checkpoint(ckpt_path, out_path, do_render, do_mesh);

      std::cout << "[eval] iter=" << it << " done → " << out_path << '\n';
    }

    std::cout << "\n[eval] All done.\n";
  } catch (const std::exception &e) {
    std::cerr << "[eval] Error: " << e.what() << '\n';
    return 1;
  }

  return 0;
}
