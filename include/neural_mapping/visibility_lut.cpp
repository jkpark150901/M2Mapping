// 최종 occ grid voxel × 카메라 프레임 visibility LUT (occlusion-aware).
//
// 목적: occ grid 의 모든 voxel 에 대해 "어느 카메라 프레임에서 (가림 고려) 보이는가"
//       를 미리 계산해 occgrid_cache/visibility_lut.bin 으로 저장한다.
//       IBRNet 식 feature aggregation 에서 "어떤 view 를 모을지" 정하는 입력.
//
// 방법: per-camera z-buffer (occlusion 근사)
//   카메라마다 모든 voxel 중심을 투영 -> 다운샘플 격자에서 픽셀당 최근접 voxel 만
//   visible. visible(v,i) <=> z_v <= min_cell_depth + tol.
//
// 비용: O(C × N) (카메라수 × voxel수) 이지만 쌍당 O(1)(투영+min). frustum 밖은 컷.
//
// 저장 포맷 (visibility_lut.bin, little-endian):
//   int64 N; int64 C; int64 nbytes(=ceil(C/8)); int32 downscale;
//   float32 xyz[N*3]   (voxel world 중심, query 매칭용)
//   uint8   lut[N*nbytes]  (voxel n 의 카메라 i 비트: lut[n*nbytes + i/8] >> (i%8) & 1)
//
// 기존 neural_mapping.cpp 본문은 건드리지 않기 위해 멤버 정의를 이 별도 파일에 둔다.

#include "neural_mapping/neural_mapping.h"
#include "params/params.h"

#include "kaolin_wisp_cpp/spc_ops/spc_ops.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <vector>

void NeuralSLAM::build_visibility_lut(int downscale, float tol_factor) {
  torch::NoGradGuard no_grad;

  if (!local_map_ptr || !local_map_ptr->p_acc_strcut_) {
    std::cerr << "[vis_lut] octree 가 준비되지 않음 - skip\n";
    return;
  }

  // --- 1) 최종 occ grid voxel 중심 (world, CPU float) ---
  auto dense = local_map_ptr->p_acc_strcut_->get_quantized_points();
  auto normalized = spc_ops::quantized_points_to_fpoints(dense, k_octree_level);
  auto Xw = local_map_ptr->m1p1_pts_to_xyz(normalized)
                .to(torch::kCPU)
                .to(torch::kFloat32)
                .contiguous();
  const int64_t N = Xw.size(0);

  // --- 2) color(train) pose + intrinsic ---
  auto &dp = *data_loader_ptr->dataparser_ptr_;
  const int C = dp.size(dataparser::DataType::TrainColor);
  if (N == 0 || C == 0) {
    std::cerr << "[vis_lut] voxel(" << N << ") 또는 camera(" << C
              << ") 가 0 - skip\n";
    return;
  }
  auto poses = dp.get_pose(torch::arange(C, torch::kLong),
                           dataparser::DataType::TrainColor)
                   .to(torch::kCPU)
                   .to(torch::kFloat32)
                   .contiguous(); // [C,4,4] camera-to-world

  const auto &cam = dp.sensor_.camera;
  const float fx = cam.fx, fy = cam.fy, cx = cam.cx, cy = cam.cy;
  const int W = cam.width, H = cam.height;
  const int s = downscale > 0 ? downscale : 1;
  const int Wc = (W + s - 1) / s, Hc = (H + s - 1) / s;
  const float tol = tol_factor * k_leaf_size;

  const int64_t nbytes = (C + 7) / 8;
  std::vector<uint8_t> lut((size_t)N * nbytes, 0);
  std::vector<float> zbuf((size_t)Wc * Hc);

  auto t0 = std::chrono::high_resolution_clock::now();
  std::cout << "[vis_lut] voxels=" << N << " cameras=" << C << " leaf="
            << k_leaf_size << " downscale=" << s << " tol=" << tol << "\n";

  // --- 3) per-camera z-buffer ---
  for (int i = 0; i < C; ++i) {
    auto c2w = poses[i];                                   // [4,4]
    auto R = c2w.narrow(0, 0, 3).narrow(1, 0, 3);          // [3,3] c2w 회전
    auto tvec = c2w.narrow(0, 0, 3).narrow(1, 3, 1).reshape({3});
    auto Xc = torch::matmul(Xw - tvec, R).contiguous();    // R^T (Xw - t) = [N,3]
    auto zT = Xc.select(1, 2).contiguous();
    auto uT = (Xc.select(1, 0) * fx / zT + cx).contiguous();
    auto vT = (Xc.select(1, 1) * fy / zT + cy).contiguous();
    auto validT = ((zT > 0) & (uT >= 0) & (uT < (float)W) & (vT >= 0) &
                   (vT < (float)H))
                      .to(torch::kUInt8)
                      .contiguous();

    const float *z = zT.data_ptr<float>();
    const float *u = uT.data_ptr<float>();
    const float *v = vT.data_ptr<float>();
    const uint8_t *vm = validT.data_ptr<uint8_t>();

    std::fill(zbuf.begin(), zbuf.end(), 1e30f);
    // pass 1: 셀별 최근접 depth
    for (int64_t n = 0; n < N; ++n) {
      if (!vm[n])
        continue;
      int cell = (int)(v[n] / s) * Wc + (int)(u[n] / s);
      if (z[n] < zbuf[cell])
        zbuf[cell] = z[n];
    }
    // pass 2: 최근접(±tol) voxel 만 visible -> 비트 set
    const uint8_t bit = (uint8_t)(1u << (i & 7));
    const int64_t col = i >> 3;
    for (int64_t n = 0; n < N; ++n) {
      if (!vm[n])
        continue;
      int cell = (int)(v[n] / s) * Wc + (int)(u[n] / s);
      if (z[n] <= zbuf[cell] + tol)
        lut[(size_t)n * nbytes + col] |= bit;
    }
    if ((i + 1) % 200 == 0)
      std::cout << "  [vis_lut] " << (i + 1) << "/" << C << " cams\n";
  }

  // --- 4) 저장 ---
  auto base = k_dataset_path;
  if (std::filesystem::is_regular_file(base))
    base = base.parent_path();
  auto dir = base / "occgrid_cache";
  std::filesystem::create_directories(dir);
  auto path = dir / "visibility_lut.bin";

  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  if (!f) {
    std::cerr << "[vis_lut] 저장 실패: " << path << "\n";
    return;
  }
  const int64_t Ni = N, Ci = C, nb = nbytes;
  const int32_t sd = s;
  f.write(reinterpret_cast<const char *>(&Ni), sizeof(Ni));
  f.write(reinterpret_cast<const char *>(&Ci), sizeof(Ci));
  f.write(reinterpret_cast<const char *>(&nb), sizeof(nb));
  f.write(reinterpret_cast<const char *>(&sd), sizeof(sd));
  f.write(reinterpret_cast<const char *>(Xw.data_ptr<float>()),
          (std::streamsize)(N * 3 * sizeof(float)));
  f.write(reinterpret_cast<const char *>(lut.data()),
          (std::streamsize)((size_t)N * nbytes));
  f.close();

  // 간단 통계
  int64_t total_seen = 0, never = 0;
  for (int64_t n = 0; n < N; ++n) {
    int cnt = 0;
    for (int64_t b = 0; b < nbytes; ++b)
      cnt += __builtin_popcount(lut[(size_t)n * nbytes + b]);
    total_seen += cnt;
    if (cnt == 0)
      never++;
  }
  std::cout << "[vis_lut] saved -> " << path << "  ("
            << (double)(N * nbytes) / 1e6 << " MB)\n";
  double ms = std::chrono::duration<double, std::milli>(
                  std::chrono::high_resolution_clock::now() - t0)
                  .count();
  std::cout << "[vis_lut] voxel당 visible 카메라 평균 "
            << (double)total_seen / (double)N << ", 한번도 안보임 " << never
            << "/" << N << "  (" << ms / 1000.0 << "s)\n";
}
