#include "neural_mapping/neural_mapping.h"

#include <cstdlib>

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
    // mode=2: build-occgrid only. data is loaded, but NO pretrained load and NO
    // training thread (mode=0 would call load_pretrained -> pt.yaml -> crash).
    auto slam = std::make_shared<NeuralSLAM>(2, config_path, data_path);
    const bool ok = slam->run_build_occgrid();
    if (!ok) {
      std::cerr << "[build_occgrid] build_occ_map() returned false\n";
      return 1;
    }
    std::cout << "[build_occgrid] Done. occgrid_cache written under data dir.\n";
  } catch (const std::exception &e) {
    std::cerr << "[build_occgrid] Error: " << e.what() << '\n';
    return 1;
  }

  // 결과는 이미 디스크에 저장됨. libtorch+CUDA 전역 소멸자 순서로 인한 종료 시
  // free()/double-free/segfault 를 피하려고 소멸자를 건너뛰고 즉시 종료한다.
  std::cout.flush();
  std::cerr.flush();
  std::_Exit(0);
}
