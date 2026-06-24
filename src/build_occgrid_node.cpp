#include "neural_mapping/neural_mapping.h"

#define BACKWARD_HAS_DW 1
#include "backward.hpp"
namespace backward {
backward::SignalHandling sh;
}

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "Usage: build_occgrid_node <config_path> <data_path>\n"
              << "  Builds (or loads from cache) the occgrid for a dataset\n"
              << "  and writes occgrid_cache/ into <data_path>.\n";
    return 1;
  }

  torch::manual_seed(0);
  torch::cuda::manual_seed_all(0);

  const std::string config_path = argv[1];
  const std::string data_path   = argv[2];

  try {
    // mode=0: data is loaded but training thread is NOT started.
    auto slam = std::make_shared<NeuralSLAM>(0, config_path, data_path);
    const bool ok = slam->run_build_occgrid();
    if (!ok) {
      std::cerr << "[build_occgrid] build_occ_map() returned false\n";
      return 1;
    }
    std::cout << "[build_occgrid] Done. Cache written to "
              << data_path << "/occgrid_cache/\n";
  } catch (const std::exception &e) {
    std::cerr << "[build_occgrid] Error: " << e.what() << '\n';
    return 1;
  }

  return 0;
}
