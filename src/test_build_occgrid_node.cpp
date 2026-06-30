// Standalone "build occ grid only" 테스트 노드.
//
// 목적: 학습 없이 occ grid 빌드만 단독 실행/검증한다.
//
// run_build_occgrid() 는 neural_mapping.cpp(neural_mapping_lib)에 정의돼 있다.
// 여기서는 그 멤버를 호출만 한다.

#include "neural_mapping/neural_mapping.h"

#include <cstdlib>
#include <string>

#define BACKWARD_HAS_DW 1
#include "backward.hpp"
namespace backward {
backward::SignalHandling sh;
}

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr
        << "Usage: test_build_occgrid_node <config_path> <data_path> "
           "[--vis-lut [downscale]]\n"
        << "  학습 없이 occ grid 만 빌드한다 (mode=2).\n"
        << "  --vis-lut [d] : occ voxel×카메라 visibility LUT 도 생성\n"
        << "                  (occgrid_cache/visibility_lut.bin, downscale 기본 8)\n"
        << "  결과: <data>/occgrid_cache/{qpts.pt,meta.bin}, "
           "<output>/as_prior.ply\n";
    return 1;
  }

  torch::manual_seed(0);
  torch::cuda::manual_seed_all(0);

  const std::string config_path = argv[1];
  const std::string data_path = argv[2];

  // 옵션: --vis-lut [downscale]
  bool vis_lut = false;
  int vis_downscale = 8;
  for (int a = 3; a < argc; ++a) {
    std::string arg = argv[a];
    if (arg == "--vis-lut") {
      vis_lut = true;
      if (a + 1 < argc && argv[a + 1][0] != '-')
        vis_downscale = std::atoi(argv[++a]);
    }
  }

  try {
    // mode=2: occ-build only. 데이터는 로드하되 pretrained 로드/학습 스레드 없음.
    auto slam = std::make_shared<NeuralSLAM>(2, config_path, data_path);
    const bool ok = slam->run_build_occgrid();
    if (!ok) {
      std::cerr << "[test_build_occgrid] build_occ_map() returned false\n";
      return 1;
    }
    std::cout << "[test_build_occgrid] Done. occ grid cache -> " << data_path
              << "/occgrid_cache/\n";

    // octree 준비 완료(build/cache-hit 무관) -> 옵션이면 visibility LUT 생성
    if (vis_lut) {
      std::cout << "[test_build_occgrid] building visibility LUT "
                   "(downscale="
                << vis_downscale << ") ...\n";
      slam->build_visibility_lut(vis_downscale);
    }
  } catch (const std::exception &e) {
    std::cerr << "[test_build_occgrid] Error: " << e.what() << '\n';
    return 1;
  }

  // 결과는 이미 디스크에 flush/close 됨. libtorch+CUDA 전역 소멸자 순서로 인한
  // 종료 시 double-free/segfault 를 피하려고 소멸자를 건너뛰고 즉시 종료한다.
  std::cout.flush();
  std::cerr.flush();
  std::_Exit(0);
}
