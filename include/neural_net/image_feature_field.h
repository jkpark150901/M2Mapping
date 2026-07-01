#pragma once
//
// LUT 기반 multi-view image feature 필드 (SparseNeuS 식 conditional volume, OGM).
//
// 구조 (SparseNeuS get_conditional_volume + sdf 와 동일):
//   [init 1회] back-project: 각 occupied voxel 중심을 LUT 가 visible 이라고 한
//     카메라들에만 투영 → 2D feature bilinear → multi-view mean/var/coverage 통합
//     → voxel 노드별 feature 를 conditional volume `vox_feat_` 로 저장.
//   [query] get_sdf(x): x 주변 8개 voxel 노드를 찾아 그 feature 를 trilinear 보간.
//
// 즉 back-projection 은 1회 bake, query 는 lookup+trilinear 만(연속·저비용).
// occlusion 은 visibility_lut.bin 사용. frozen.

#include "utils/sensor_utils/cameras.hpp"
#include <unordered_map>
#include <string>
#include <torch/torch.h>
#include <vector>

struct ImageFeatureConvBnReLUImpl : torch::nn::Module {
  ImageFeatureConvBnReLUImpl(int in_channels, int out_channels, int kernel_size,
                             int stride = 1, int padding = 1);
  torch::Tensor forward(const torch::Tensor &x);

  torch::nn::Conv2d conv{nullptr};
  torch::nn::BatchNorm2d bn{nullptr};
};
TORCH_MODULE(ImageFeatureConvBnReLU);

struct SparseNeuSFeatureNetImpl : torch::nn::Module {
  SparseNeuSFeatureNetImpl();
  std::vector<torch::Tensor> forward(const torch::Tensor &x);

  torch::Tensor upsample_add(const torch::Tensor &x, const torch::Tensor &y);

  torch::nn::Sequential conv0{nullptr}, conv1{nullptr}, conv2{nullptr};
  torch::nn::Conv2d toplayer{nullptr}, lat1{nullptr}, lat0{nullptr};
  torch::nn::Conv2d smooth1{nullptr}, smooth0{nullptr};
};
TORCH_MODULE(SparseNeuSFeatureNet);

struct ImageFeatureField : torch::nn::Module {
  typedef std::shared_ptr<ImageFeatureField> Ptr;
  ImageFeatureField();

  /// @param train_color       [N,H,W,3] CPU float[0,1] (LUT 카메라 순서와 동일)
  /// @param train_color_poses [N,3,4] camera-to-world
  /// @param lut_path          occgrid_cache/visibility_lut.bin
  /// @param leaf              voxel size, views_cap K, feat_scale(메모리)
  void init(const torch::Tensor &train_color,
            const torch::Tensor &train_color_poses,
            const sensor::Cameras &camera, const std::string &lut_path,
            float leaf, int views_cap, float feat_scale);

  /// xyz [M,3] world -> { z_img [M, 2C+1] }  (voxel 노드 trilinear)
  std::vector<torch::Tensor> sample(const torch::Tensor &xyz);
  std::vector<torch::Tensor> sample_baked(const torch::Tensor &xyz);
  std::vector<torch::Tensor> sample_trainable(const torch::Tensor &xyz);
  // 현재(학습된) FeatureNet 가중치로 conditional volume(vox_feat_) 을 1회 bake.
  // trainable 모드의 렌더/평가(no-grad)에서 get_sdf 마다 CNN 재실행하는 것을 회피.
  void bake_volume();

  static int out_dim(int feat_dim) { return 2 * feat_dim + 1; }
  int out_dim() const { return out_dim(feat_dim_); }
  bool initialized() const { return keys_sorted_.defined(); }

  // ---- conditional volume (bake 후 유지되는 것만) ----
  torch::Tensor vox_feat_;     // [Nv, 2C+1] voxel 노드별 baked feature
  torch::Tensor keys_sorted_;  // [Nv] int64 정렬된 voxel-key
  torch::Tensor perm_;         // [Nv] int64 sorted pos -> voxel row
  torch::Tensor corner_off_;   // [8,3] float {0,1}^3
  torch::Tensor vox_q_;        // [Nv,3] int64 voxel node coords
  torch::Tensor vox_cams_;     // [Nv,K] int64 visible camera ids

  torch::Tensor train_color_;  // [V,H,W,3] CPU float32, for trainable mode
  torch::Tensor w2c_R_;        // [V,3,3]
  torch::Tensor w2c_t_;        // [V,3]

  int feat_dim_ = 3;
  float leaf_ = 0;
  int views_cap_ = 0;
  int Hs_ = 0, Ws_ = 0;
  float fxs_ = 0, fys_ = 0, cxs_ = 0, cys_ = 0;

  SparseNeuSFeatureNet feature_net_{nullptr};
  ImageFeatureConvBnReLU compress_layer_{nullptr};
  // encoder(FeatureNet+compress) 는 frozen. trilinear 조회 결과를 이 학습가능
  // MLP 로 임베딩해서 학습한다 (prior 붕괴 없이 조건을 적응).
  torch::nn::Sequential embed_mlp_{nullptr};
};
