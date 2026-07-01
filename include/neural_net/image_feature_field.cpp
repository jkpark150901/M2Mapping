#include "neural_net/image_feature_field.h"
#include "params/params.h"

#include <c10/cuda/CUDACachingAllocator.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <vector>

namespace {
constexpr int64_t OFF = 1 << 20; // voxel 좌표 음수 보정 (|q| < 2^20)

// voxel 정수좌표 [.,3] -> int64 key.
// q+OFF in [0, 2^21) 이므로 bit-pack 대신 곱셈+덧셈으로 동일 유일키 (버전 안전).
torch::Tensor encode_key(const torch::Tensor &q) {
  const int64_t S1 = (int64_t)1 << 21;
  const int64_t S2 = (int64_t)1 << 42;
  auto x = q.select(1, 0) + OFF;
  auto y = q.select(1, 1) + OFF;
  auto z = q.select(1, 2) + OFF;
  return x * S2 + y * S1 + z;
}
} // namespace

ImageFeatureConvBnReLUImpl::ImageFeatureConvBnReLUImpl(
    int in_channels, int out_channels, int kernel_size, int stride,
    int padding) {
  conv = register_module(
      "conv", torch::nn::Conv2d(torch::nn::Conv2dOptions(
                                    in_channels, out_channels, kernel_size)
                                    .stride(stride)
                                    .padding(padding)
                                    .bias(false)));
  bn = register_module(
      "bn", torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(out_channels)));
}

torch::Tensor ImageFeatureConvBnReLUImpl::forward(const torch::Tensor &x) {
  return torch::relu(bn->forward(conv->forward(x)));
}

SparseNeuSFeatureNetImpl::SparseNeuSFeatureNetImpl() {
  conv0 = register_module(
      "conv0", torch::nn::Sequential(
                   ImageFeatureConvBnReLU(3, 8, 3, 1, 1),
                   ImageFeatureConvBnReLU(8, 8, 3, 1, 1)));
  conv1 = register_module(
      "conv1", torch::nn::Sequential(
                   ImageFeatureConvBnReLU(8, 16, 5, 2, 2),
                   ImageFeatureConvBnReLU(16, 16, 3, 1, 1),
                   ImageFeatureConvBnReLU(16, 16, 3, 1, 1)));
  conv2 = register_module(
      "conv2", torch::nn::Sequential(
                   ImageFeatureConvBnReLU(16, 32, 5, 2, 2),
                   ImageFeatureConvBnReLU(32, 32, 3, 1, 1),
                   ImageFeatureConvBnReLU(32, 32, 3, 1, 1)));

  toplayer = register_module(
      "toplayer", torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 32, 1)));
  lat1 = register_module(
      "lat1", torch::nn::Conv2d(torch::nn::Conv2dOptions(16, 32, 1)));
  lat0 = register_module(
      "lat0", torch::nn::Conv2d(torch::nn::Conv2dOptions(8, 32, 1)));
  smooth1 = register_module(
      "smooth1",
      torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 16, 3).padding(1)));
  smooth0 = register_module(
      "smooth0",
      torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 8, 3).padding(1)));
}

torch::Tensor SparseNeuSFeatureNetImpl::upsample_add(const torch::Tensor &x,
                                                     const torch::Tensor &y) {
  return torch::nn::functional::interpolate(
             x, torch::nn::functional::InterpolateFuncOptions()
                    .size(std::vector<int64_t>{y.size(2), y.size(3)})
                    .mode(torch::kBilinear)
                    .align_corners(true)) +
         y;
}

std::vector<torch::Tensor>
SparseNeuSFeatureNetImpl::forward(const torch::Tensor &x) {
  auto conv0_out = conv0->forward(x);
  auto conv1_out = conv1->forward(conv0_out);
  auto conv2_out = conv2->forward(conv1_out);

  auto feat2 = toplayer->forward(conv2_out);
  auto feat1 = upsample_add(feat2, lat1->forward(conv1_out));
  auto feat0 = upsample_add(feat1, lat0->forward(conv0_out));

  feat1 = smooth1->forward(feat1);
  feat0 = smooth0->forward(feat0);
  return {feat2, feat1, feat0};
}

ImageFeatureField::ImageFeatureField() {
  feature_net_ = register_module("sparse_neus_feature_net",
                                 SparseNeuSFeatureNet());
  compress_layer_ =
      register_module("compress_layer", ImageFeatureConvBnReLU(56, 16, 3, 1, 1));
}

void ImageFeatureField::init(const torch::Tensor &train_color,
                             const torch::Tensor &train_color_poses,
                             const sensor::Cameras &camera,
                             const std::string &lut_path, float leaf,
                             int views_cap, float feat_scale) {
  torch::NoGradGuard ng;
  // 필드 텐서는 get_sdf 의 xyz 와 같은 device(k_device)에 둬야 함.
  // (train_color_poses_ 는 CPU 라 device 를 그대로 쓰면 CPU/GPU 혼용 크래시)
  auto dev = k_device;
  leaf_ = leaf;
  feat_dim_ = (k_image_feature_backbone == 1) ? 16 : k_image_feature_dim;
  const int K = views_cap;
  views_cap_ = views_cap;
  feature_net_->to(dev);
  compress_layer_->to(dev);

  // ---- 1) feature map: [V,Hs,Ws,C] ----
  int V = train_color.size(0);
  int H = train_color.size(1), W = train_color.size(2);
  torch::Tensor feat_hwc;
  float fxs, fys, cxs, cys;

  if (k_image_feature_backbone == 2) {
    // ---- precomputed 외부 백본(DINOv2 등) feature 로드 ----
    // <lut dir>/image_features.bin : int64 V,C,Hs,Ws + float32[V*Hs*Ws*C]
    auto feat_path =
        std::filesystem::path(lut_path).parent_path() / "image_features.bin";
    std::ifstream ff(feat_path, std::ios::binary);
    if (!ff) {
      std::cerr << "[img_feat] precomputed feature 로드 실패: " << feat_path
                << " (precompute_image_features.py 먼저 실행)\n";
      return;
    }
    int64_t Vf = 0, Cf = 0, Hf = 0, Wf = 0;
    ff.read((char *)&Vf, 8);
    ff.read((char *)&Cf, 8);
    ff.read((char *)&Hf, 8);
    ff.read((char *)&Wf, 8);
    std::vector<float> buf((size_t)Vf * Hf * Wf * Cf);
    ff.read((char *)buf.data(), (std::streamsize)(buf.size() * sizeof(float)));
    feat_hwc = torch::from_blob(buf.data(), {Vf, Hf, Wf, Cf}, torch::kFloat32)
                   .clone()
                   .to(dev); // [V,Hs,Ws,C]
    Hs_ = (int)Hf;
    Ws_ = (int)Wf;
    feat_dim_ = (int)Cf;
    if (Vf != V)
      std::cerr << "[img_feat] 경고: precomputed views(" << Vf
                << ") != train views(" << V << ")\n";
    if ((int)Cf != k_image_feature_dim)
      std::cerr << "[img_feat] 경고: precomputed C(" << Cf
                << ") != image_feature_dim(" << k_image_feature_dim
                << ") -> decoder 차원 불일치\n";
    // feature map 해상도(Hf,Wf)에 맞춰 intrinsic 스케일
    fxs = camera.fx * (float)Ws_ / (float)W;
    fys = camera.fy * (float)Hs_ / (float)H;
    cxs = camera.cx * (float)Ws_ / (float)W;
    cys = camera.cy * (float)Hs_ / (float)H;
    std::cout << "[img_feat] precomputed features: V=" << Vf << " C=" << Cf
              << " " << Wf << "x" << Hf << "\n";
  } else {
    // ---- frozen encoder(FeatureNet) 를 뷰 청크로 실행해 feature map 생성 ----
    int Hs = std::max(1, (int)std::round(H * feat_scale));
    int Ws = std::max(1, (int)std::round(W * feat_scale));
    Hs_ = Hs;
    Ws_ = Ws;
    std::vector<torch::Tensor> fm;
    const int VCH = (k_image_feature_backbone == 1) ? 8 : 32;
    feature_net_->eval();
    compress_layer_->eval();
    for (int vs = 0; vs < V; vs += VCH) {
      int ve = std::min(vs + VCH, V);
      auto chw = train_color.slice(0, vs, ve)
                     .permute({0, 3, 1, 2})
                     .to(dev)
                     .to(torch::kFloat32)
                     .contiguous();
      torch::Tensor feat_chw;
      if (k_image_feature_backbone == 1) {
        auto pyr = feature_net_->forward(chw);
        auto fused = torch::cat(
            {torch::nn::functional::interpolate(
                 pyr[0], torch::nn::functional::InterpolateFuncOptions()
                             .size(std::vector<int64_t>{H, W})
                             .mode(torch::kBilinear)
                             .align_corners(true)),
             torch::nn::functional::interpolate(
                 pyr[1], torch::nn::functional::InterpolateFuncOptions()
                             .size(std::vector<int64_t>{H, W})
                             .mode(torch::kBilinear)
                             .align_corners(true)),
             pyr[2]},
            1); // [c,56,H,W]
        feat_chw = compress_layer_->forward(fused); // [c,16,H,W]
      } else {
        feat_chw = chw;
      }
      auto sm = torch::nn::functional::interpolate(
          feat_chw, torch::nn::functional::InterpolateFuncOptions()
                        .size(std::vector<int64_t>{Hs, Ws})
                        .mode(torch::kBilinear)
                        .align_corners(false));
      fm.push_back(sm.permute({0, 2, 3, 1}).contiguous()); // [c,Hs,Ws,C]
    }
    feat_hwc = torch::cat(fm, 0); // [V,Hs,Ws,C]
    float s = feat_scale;
    fxs = camera.fx * s;
    fys = camera.fy * s;
    cxs = camera.cx * s;
    cys = camera.cy * s;
  }
  fxs_ = fxs;
  fys_ = fys;
  cxs_ = cxs;
  cys_ = cys;

  // ---- 2) world->camera (poses [V,3,4] camera-to-world) ----
  auto P = train_color_poses.to(dev).to(torch::kFloat32);
  auto R = P.slice(2, 0, 3);                  // [V,3,3] c2w
  auto t = P.slice(2, 3, 4);                  // [V,3,1]
  auto w2c_R = R.transpose(1, 2).contiguous();                 // R^T [V,3,3]
  auto w2c_t = (-torch::matmul(w2c_R, t)).squeeze(-1).contiguous(); // [V,3]
  w2c_R_ = w2c_R;
  w2c_t_ = w2c_t;

  // ---- 3) visibility_lut.bin 로드 -> voxel 좌표 + voxel당 visible cam K ----
  std::ifstream f(lut_path, std::ios::binary);
  if (!f) {
    std::cerr << "[img_feat] visibility_lut 로드 실패: " << lut_path << "\n";
    return;
  }
  int64_t Nv = 0, C = 0, nbytes = 0; int32_t ds = 0;
  f.read((char *)&Nv, 8); f.read((char *)&C, 8);
  f.read((char *)&nbytes, 8); f.read((char *)&ds, 4);
  std::vector<float> xyz(Nv * 3);
  f.read((char *)xyz.data(), (std::streamsize)(Nv * 3 * sizeof(float)));
  std::vector<uint8_t> lut((size_t)Nv * nbytes);
  f.read((char *)lut.data(), (std::streamsize)((size_t)Nv * nbytes));
  if (C != V)
    std::cerr << "[img_feat] 경고: LUT cameras(" << C << ") != train views("
              << V << "). 카메라 인덱스 정렬 확인 필요.\n";

  std::vector<int64_t> vq(Nv * 3);
  std::vector<int32_t> vcams((size_t)Nv * K, -1);
  for (int64_t n = 0; n < Nv; ++n) {
    for (int d = 0; d < 3; ++d)
      vq[n * 3 + d] = (int64_t)std::floor(xyz[n * 3 + d] / leaf);
    int cnt = 0;
    for (int64_t b = 0; b < nbytes && cnt < K; ++b) {
      uint8_t byte = lut[(size_t)n * nbytes + b];
      if (!byte) continue;
      for (int bit = 0; bit < 8 && cnt < K; ++bit)
        if (byte & (1u << bit)) {
          int cam = (int)(b * 8 + bit);
          if (cam < V) vcams[(size_t)n * K + cnt++] = cam;
        }
    }
  }
  auto vox_q = torch::from_blob(vq.data(), {Nv, 3}, torch::kInt64).clone().to(dev);
  auto vox_cams =
      torch::from_blob(vcams.data(), {Nv, (int64_t)K}, torch::kInt32)
          .clone().to(dev).to(torch::kLong);
  vox_q_ = vox_q;
  vox_cams_ = vox_cams;

  auto keys = encode_key(vox_q);
  auto sorted = keys.sort();
  keys_sorted_ = std::get<0>(sorted).contiguous();
  perm_ = std::get<1>(sorted).contiguous();

  corner_off_ = torch::zeros({8, 3}, torch::TensorOptions().device(dev));
  for (int c = 0; c < 8; ++c) {
    corner_off_[c][0] = (c >> 2) & 1;
    corner_off_[c][1] = (c >> 1) & 1;
    corner_off_[c][2] = c & 1;
  }

  // ---- 4) back-project bake: voxel 노드 feature [Nv, 2C+1] (chunk 처리) ----
  auto centers_all = (vox_q.to(torch::kFloat32) + 0.5f) * leaf; // [Nv,3]
  auto feat_hwc_dev = feat_hwc; // [V,Hs,Ws,C]
  std::vector<torch::Tensor> chunks;
  const int64_t CH = 50000;
  for (int64_t s0 = 0; s0 < Nv; s0 += CH) {
    int64_t e0 = std::min(s0 + CH, Nv);
    int64_t c = e0 - s0;
    auto ctr = centers_all.slice(0, s0, e0);              // [c,3]
    auto cams = vox_cams.slice(0, s0, e0);                // [c,K]
    auto cam_ok = (cams >= 0);                            // [c,K]
    auto camc = cams.clamp_min(0);
    auto Rg = w2c_R.index_select(0, camc.reshape(-1)).reshape({c, K, 3, 3});
    auto tg = w2c_t.index_select(0, camc.reshape(-1)).reshape({c, K, 3});
    auto Xc = torch::einsum("nkij,nj->nki", {Rg, ctr}) + tg; // [c,K,3]
    auto z = Xc.select(2, 2);
    auto zc = torch::where(z.abs() < 1e-9f, torch::full_like(z, 1e-9f), z);
    auto upx = Xc.select(2, 0) * fxs / zc + cxs;          // [c,K]
    auto vpx = Xc.select(2, 1) * fys / zc + cys;
    auto ok = (z > 0) & (upx >= 0) & (upx < (float)Ws_) & (vpx >= 0) &
              (vpx < (float)Hs_) & cam_ok;                 // [c,K]

    auto ci = camc.reshape(-1);
    auto uf = upx.reshape(-1), vf = vpx.reshape(-1);
    auto okf = ok.reshape(-1).to(torch::kFloat32);
    auto x0 = torch::floor(uf), y0 = torch::floor(vf);
    auto wx = (uf - x0).unsqueeze(1), wy = (vf - y0).unsqueeze(1);
    auto x0i = x0.to(torch::kLong).clamp(0, Ws_ - 1);
    auto x1i = (x0.to(torch::kLong) + 1).clamp(0, Ws_ - 1);
    auto y0i = y0.to(torch::kLong).clamp(0, Hs_ - 1);
    auto y1i = (y0.to(torch::kLong) + 1).clamp(0, Hs_ - 1);
    auto gg = [&](const torch::Tensor &yy, const torch::Tensor &xx) {
      return feat_hwc_dev.index({ci, yy, xx}); // [c*K, C]
    };
    auto feat = gg(y0i, x0i) * (1 - wx) * (1 - wy) +
                gg(y0i, x1i) * wx * (1 - wy) +
                gg(y1i, x0i) * (1 - wx) * wy + gg(y1i, x1i) * wx * wy;
    feat = (feat * okf.unsqueeze(1)).reshape({c, K, feat_dim_});

    auto cnt = ok.to(torch::kFloat32).sum(1);            // [c]
    auto denom = cnt.clamp_min(1).unsqueeze(-1);
    auto mean = feat.sum(1) / denom;                     // [c,C]
    auto var = ((feat * feat).sum(1) / denom - mean * mean).clamp_min(0);
    auto cov = (cnt / (float)K).unsqueeze(-1);           // [c,1]
    chunks.push_back(torch::cat({mean, var, cov}, -1));  // [c,2C+1]
  }
  vox_feat_ = torch::cat(chunks, 0).contiguous();        // [Nv,2C+1]

  // bake 용 feature map(대용량, ~수 GB)은 이제 불필요 -> 명시적 해제 후 캐시 반환.
  // (PyTorch 캐시에 남으면 tcnn cuMemCreate 가 쓸 연속 메모리를 잠식해 backward OOM)
  feat_hwc = torch::Tensor();
  feat_hwc_dev = torch::Tensor();
  c10::cuda::CUDACachingAllocator::emptyCache();

  // encoder(FeatureNet+compress) 는 frozen (pretrained 사용 전제) -> grad X
  for (auto &p : feature_net_->parameters())
    p.set_requires_grad(false);
  for (auto &p : compress_layer_->parameters())
    p.set_requires_grad(false);

  // trilinear 조회 결과를 임베딩할 학습가능 MLP (frozen 조건을 적응)
  if (k_image_feature_trainable) {
    int OD = out_dim();
    embed_mlp_ = register_module(
        "embed_mlp",
        torch::nn::Sequential(torch::nn::Linear(OD, 64), torch::nn::ReLU(),
                              torch::nn::Linear(64, OD)));
    embed_mlp_->to(dev);
  }

  std::cout << "[img_feat] baked conditional volume: voxels=" << Nv
            << " views=" << V << " featmap=" << Ws_ << "x" << Hs_ << " K=" << K
            << " out_dim=" << out_dim()
            << " embed_mlp=" << (embed_mlp_ ? "on" : "off") << "\n";
}

std::vector<torch::Tensor> ImageFeatureField::sample(const torch::Tensor &xyz) {
  // frozen conditional volume 을 trilinear 조회(값싸고 view 차원 없음),
  // 그 결과를 학습가능 MLP 로 임베딩 (grad 는 MLP 로만 흐름).
  auto z = sample_baked(xyz)[0]; // [M, 2C+1] (frozen volume, NoGrad)
  if (embed_mlp_)
    z = embed_mlp_->forward(z); // [M, 2C+1] 학습가능 변환
  return {z};
}

void ImageFeatureField::bake_volume() {
  if (!train_color_.defined() || !keys_sorted_.defined())
    return;
  torch::NoGradGuard ng;
  auto dev = k_device;
  const int V = train_color_.size(0);
  const int H = train_color_.size(1), W = train_color_.size(2);
  const int Hs = Hs_, Ws = Ws_;
  const int K = views_cap_;

  // ---- feature map [V,Hs,Ws,C] : 현재 학습된 FeatureNet 가중치 사용 ----
  // 학습 중(mid export)에도 불릴 수 있으니 BN 모드 저장/복원.
  bool ft_train = feature_net_->is_training();
  bool ct_train = compress_layer_->is_training();
  feature_net_->eval();
  compress_layer_->eval();
  std::vector<torch::Tensor> fm;
  const int VCH = (k_image_feature_backbone == 1) ? 8 : 32;
  for (int vs = 0; vs < V; vs += VCH) {
    int ve = std::min(vs + VCH, V);
    auto chw = train_color_.slice(0, vs, ve)
                   .permute({0, 3, 1, 2})
                   .to(dev)
                   .to(torch::kFloat32)
                   .contiguous();
    torch::Tensor feat_chw;
    if (k_image_feature_backbone == 1) {
      auto pyr = feature_net_->forward(chw);
      auto fused = torch::cat(
          {torch::nn::functional::interpolate(
               pyr[0], torch::nn::functional::InterpolateFuncOptions()
                           .size(std::vector<int64_t>{H, W})
                           .mode(torch::kBilinear)
                           .align_corners(true)),
           torch::nn::functional::interpolate(
               pyr[1], torch::nn::functional::InterpolateFuncOptions()
                           .size(std::vector<int64_t>{H, W})
                           .mode(torch::kBilinear)
                           .align_corners(true)),
           pyr[2]},
          1);
      feat_chw = compress_layer_->forward(fused);
    } else {
      feat_chw = chw;
    }
    auto sm = torch::nn::functional::interpolate(
        feat_chw, torch::nn::functional::InterpolateFuncOptions()
                      .size(std::vector<int64_t>{Hs, Ws})
                      .mode(torch::kBilinear)
                      .align_corners(false));
    fm.push_back(sm.permute({0, 2, 3, 1}).contiguous());
  }
  auto feat_hwc = torch::cat(fm, 0); // [V,Hs,Ws,C]

  // ---- per-voxel aggregation (init baked branch 와 동일) ----
  const int64_t Nv = vox_q_.size(0);
  auto centers_all = (vox_q_.to(torch::kFloat32) + 0.5f) * leaf_;
  std::vector<torch::Tensor> chunks;
  const int64_t CH = 50000;
  for (int64_t s0 = 0; s0 < Nv; s0 += CH) {
    int64_t e0 = std::min(s0 + CH, Nv);
    int64_t c = e0 - s0;
    auto ctr = centers_all.slice(0, s0, e0);
    auto cams = vox_cams_.slice(0, s0, e0);
    auto cam_ok = (cams >= 0);
    auto camc = cams.clamp_min(0);
    auto Rg = w2c_R_.index_select(0, camc.reshape(-1)).reshape({c, K, 3, 3});
    auto tg = w2c_t_.index_select(0, camc.reshape(-1)).reshape({c, K, 3});
    auto Xc = torch::einsum("nkij,nj->nki", {Rg, ctr}) + tg;
    auto z = Xc.select(2, 2);
    auto zc = torch::where(z.abs() < 1e-9f, torch::full_like(z, 1e-9f), z);
    auto upx = Xc.select(2, 0) * fxs_ / zc + cxs_;
    auto vpx = Xc.select(2, 1) * fys_ / zc + cys_;
    auto ok = (z > 0) & (upx >= 0) & (upx < (float)Ws) & (vpx >= 0) &
              (vpx < (float)Hs) & cam_ok;
    auto ci = camc.reshape(-1);
    auto uf = upx.reshape(-1), vf = vpx.reshape(-1);
    auto okf = ok.reshape(-1).to(torch::kFloat32);
    auto x0 = torch::floor(uf), y0 = torch::floor(vf);
    auto wx = (uf - x0).unsqueeze(1), wy = (vf - y0).unsqueeze(1);
    auto x0i = x0.to(torch::kLong).clamp(0, Ws - 1);
    auto x1i = (x0.to(torch::kLong) + 1).clamp(0, Ws - 1);
    auto y0i = y0.to(torch::kLong).clamp(0, Hs - 1);
    auto y1i = (y0.to(torch::kLong) + 1).clamp(0, Hs - 1);
    auto gg = [&](const torch::Tensor &yy, const torch::Tensor &xx) {
      return feat_hwc.index({ci, yy, xx});
    };
    auto feat = gg(y0i, x0i) * (1 - wx) * (1 - wy) +
                gg(y0i, x1i) * wx * (1 - wy) +
                gg(y1i, x0i) * (1 - wx) * wy + gg(y1i, x1i) * wx * wy;
    feat = (feat * okf.unsqueeze(1)).reshape({c, K, feat_dim_});
    auto cnt = ok.to(torch::kFloat32).sum(1);
    auto denom = cnt.clamp_min(1).unsqueeze(-1);
    auto mean = feat.sum(1) / denom;
    auto var = ((feat * feat).sum(1) / denom - mean * mean).clamp_min(0);
    auto cov = (cnt / (float)K).unsqueeze(-1);
    chunks.push_back(torch::cat({mean, var, cov}, -1));
  }
  vox_feat_ = torch::cat(chunks, 0).contiguous();
  feature_net_->train(ft_train); // BN 모드 복원 (후속 학습 보호)
  compress_layer_->train(ct_train);
  std::cout << "[img_feat] bake_volume: " << Nv
            << " voxels baked (fast render path)\n";
}

std::vector<torch::Tensor>
ImageFeatureField::sample_baked(const torch::Tensor &xyz) {
  torch::NoGradGuard ng;
  const int64_t M = xyz.size(0);
  const int OD = out_dim();
  if (!initialized() || !vox_feat_.defined())
    return {torch::zeros({M, OD}, xyz.options())};
  const int64_t Nv = keys_sorted_.size(0);

  // voxel 노드(center) 격자 좌표: center = (i+0.5)*leaf -> i = x/leaf - 0.5
  auto xl = xyz / leaf_ - 0.5f;                          // [M,3]
  auto base = torch::floor(xl).to(torch::kInt64);        // [M,3]
  auto frac = xl - base.to(torch::kFloat32);             // [M,3] in [0,1)
  auto offl = corner_off_.to(torch::kInt64);             // [8,3]

  auto nc = base.unsqueeze(1) + offl.unsqueeze(0);       // [M,8,3]
  auto key = encode_key(nc.reshape({-1, 3}));            // [M*8]
  auto pos = torch::searchsorted(keys_sorted_, key).clamp(0, Nv - 1);
  auto match = keys_sorted_.index_select(0, pos) == key; // [M*8]
  auto row = perm_.index_select(0, pos);
  row = torch::where(match, row, torch::zeros_like(row));
  auto fe = vox_feat_.index_select(0, row);              // [M*8, OD]
  fe = (fe * match.unsqueeze(-1).to(fe.dtype())).reshape({M, 8, OD});

  // trilinear 가중치
  auto fb = frac.unsqueeze(1);                           // [M,1,3]
  auto co = corner_off_.unsqueeze(0);                    // [1,8,3]
  auto w = (co * fb + (1 - co) * (1 - fb)).prod(2);      // [M,8]
  auto present = match.reshape({M, 8}).to(torch::kFloat32);
  auto wp = w * present;                                 // [M,8]
  auto denom = wp.sum(1, true).clamp_min(1e-8f);         // [M,1]
  auto out = (fe * wp.unsqueeze(-1)).sum(1) / denom;     // [M,OD]
  auto any = (present.sum(1, true) > 0).to(out.dtype()); // [M,1]
  return {out * any};
}

std::vector<torch::Tensor>
ImageFeatureField::sample_trainable(const torch::Tensor &xyz) {
  const int64_t M = xyz.size(0);
  const int OD = out_dim();
  auto dev = xyz.device();
  if (!initialized() || !train_color_.defined())
    return {torch::zeros({M, OD}, xyz.options())};
  const int64_t Nv = keys_sorted_.size(0);
  std::cerr << "[trainable] enter M=" << M << " K=" << views_cap_ << std::endl;

  auto xl = xyz / leaf_ - 0.5f;
  auto base = torch::floor(xl).to(torch::kInt64);
  auto frac = xl - base.to(torch::kFloat32);
  auto offl = corner_off_.to(torch::kInt64);

  auto nc = base.unsqueeze(1) + offl.unsqueeze(0);       // [M,8,3]
  auto key = encode_key(nc.reshape({-1, 3}));            // [M*8]
  auto pos = torch::searchsorted(keys_sorted_, key).clamp(0, Nv - 1);
  auto match = keys_sorted_.index_select(0, pos) == key; // [M*8]
  auto row = perm_.index_select(0, pos);
  row = torch::where(match, row, torch::zeros_like(row));

  const int64_t N = row.size(0);
  auto cams = vox_cams_.index_select(0, row);            // [N,K]
  cams = torch::where(match.unsqueeze(1), cams, torch::full_like(cams, -1));
  auto cams_flat_cpu = cams.reshape(-1).to(torch::kCPU).contiguous();
  const auto *cam_ptr = cams_flat_cpu.data_ptr<int64_t>();
  const int64_t cam_num = cams_flat_cpu.numel();

  std::vector<int64_t> unique_cams;
  unique_cams.reserve(std::min<int64_t>(cam_num, 256));
  std::unordered_map<int64_t, int64_t> cam_to_local;
  std::vector<int64_t> local_idx(cam_num, 0);
  for (int64_t i = 0; i < cam_num; ++i) {
    int64_t cam = cam_ptr[i];
    if (cam < 0)
      continue;
    auto it = cam_to_local.find(cam);
    if (it == cam_to_local.end()) {
      int64_t local = (int64_t)unique_cams.size();
      cam_to_local.emplace(cam, local);
      unique_cams.push_back(cam);
      local_idx[i] = local;
    } else {
      local_idx[i] = it->second;
    }
  }

  if (unique_cams.empty())
    return {torch::zeros({M, OD}, xyz.options())};
  std::cerr << "[trainable] N=" << N << " unique_cams U=" << unique_cams.size()
            << "  (fused=[U,56,Hs,Ws], gather=[N*K, C] N*K=" << (N * views_cap_)
            << ")" << std::endl;

  auto cam_ids = torch::from_blob(unique_cams.data(),
                                  {(int64_t)unique_cams.size()},
                                  torch::kInt64)
                     .clone()
                     .to(dev);
  auto local_cam = torch::from_blob(local_idx.data(), cams.sizes(), torch::kInt64)
                       .clone()
                       .to(dev);

  auto chw = train_color_.index_select(0, cam_ids.to(torch::kCPU))
                 .permute({0, 3, 1, 2})
                 .to(dev)
                 .to(torch::kFloat32)
                 .contiguous(); // [U,3,H,W]
  // 메모리: FeatureNet 을 다운샘플(Hs×Ws) 입력에서 실행. 풀해상도로 돌리면
  // fused[U,56,720,1280] 가 수십 GB -> OOM. 다운샘플로 ~16x 절감.
  chw = torch::nn::functional::interpolate(
      chw, torch::nn::functional::InterpolateFuncOptions()
               .size(std::vector<int64_t>{Hs_, Ws_})
               .mode(torch::kBilinear)
               .align_corners(false));
  torch::Tensor feat_chw;
  if (k_image_feature_backbone == 1) {
    auto pyr = feature_net_->forward(chw);
    auto to_s = [&](const torch::Tensor &p) {
      return torch::nn::functional::interpolate(
          p, torch::nn::functional::InterpolateFuncOptions()
                 .size(std::vector<int64_t>{Hs_, Ws_})
                 .mode(torch::kBilinear)
                 .align_corners(true));
    };
    auto fused = torch::cat({to_s(pyr[0]), to_s(pyr[1]), to_s(pyr[2])}, 1);
    feat_chw = compress_layer_->forward(fused);
  } else {
    feat_chw = chw;
  }
  auto feat_hwc = feat_chw.permute({0, 2, 3, 1}).contiguous(); // [U,Hs,Ws,C]
  std::cerr << "[trainable] featnet done feat_hwc=" << feat_hwc.sizes()
            << std::endl;

  auto centers = (vox_q_.index_select(0, row).to(torch::kFloat32) + 0.5f) * leaf_;
  auto cam_ok = cams >= 0;
  auto camc = cams.clamp_min(0);
  auto Rg = w2c_R_.index_select(0, camc.reshape(-1)).reshape(
      {N, views_cap_, 3, 3});
  auto tg = w2c_t_.index_select(0, camc.reshape(-1)).reshape(
      {N, views_cap_, 3});
  auto Xc = torch::einsum("nkij,nj->nki", {Rg, centers}) + tg;
  auto z = Xc.select(2, 2);
  auto zc = torch::where(z.abs() < 1e-9f, torch::full_like(z, 1e-9f), z);
  auto upx = Xc.select(2, 0) * fxs_ / zc + cxs_;
  auto vpx = Xc.select(2, 1) * fys_ / zc + cys_;
  auto ok = (z > 0) & (upx >= 0) & (upx < (float)Ws_) & (vpx >= 0) &
            (vpx < (float)Hs_) & cam_ok;
  std::cerr << "[trainable] proj done, gather 시작 (N*K="
            << (row.size(0) * views_cap_) << ")" << std::endl;

  auto ci = local_cam.reshape(-1);
  auto uf = upx.reshape(-1), vf = vpx.reshape(-1);
  auto okf = ok.reshape(-1).to(torch::kFloat32);
  auto x0 = torch::floor(uf), y0 = torch::floor(vf);
  auto wx = (uf - x0).unsqueeze(1), wy = (vf - y0).unsqueeze(1);
  auto x0i = x0.to(torch::kLong).clamp(0, Ws_ - 1);
  auto x1i = (x0.to(torch::kLong) + 1).clamp(0, Ws_ - 1);
  auto y0i = y0.to(torch::kLong).clamp(0, Hs_ - 1);
  auto y1i = (y0.to(torch::kLong) + 1).clamp(0, Hs_ - 1);
  auto gg = [&](const torch::Tensor &yy, const torch::Tensor &xx) {
    return feat_hwc.index({ci, yy, xx});
  };
  auto feat = gg(y0i, x0i) * (1 - wx) * (1 - wy) +
              gg(y0i, x1i) * wx * (1 - wy) +
              gg(y1i, x0i) * (1 - wx) * wy + gg(y1i, x1i) * wx * wy;
  feat = (feat * okf.unsqueeze(1)).reshape({N, views_cap_, feat_dim_});
  std::cerr << "[trainable] gather done" << std::endl;

  auto cnt = ok.to(torch::kFloat32).sum(1);
  auto denom = cnt.clamp_min(1).unsqueeze(-1);
  auto mean = feat.sum(1) / denom;
  auto var = ((feat * feat).sum(1) / denom - mean * mean).clamp_min(0);
  auto cov = (cnt / (float)views_cap_).unsqueeze(-1);
  auto fe = torch::cat({mean, var, cov}, -1);            // [M*8,OD]
  fe = (fe * match.unsqueeze(-1).to(fe.dtype())).reshape({M, 8, OD});

  auto fb = frac.unsqueeze(1);
  auto co = corner_off_.unsqueeze(0);
  auto w = (co * fb + (1 - co) * (1 - fb)).prod(2);
  auto present = match.reshape({M, 8}).to(torch::kFloat32);
  auto wp = w * present;
  auto denom_w = wp.sum(1, true).clamp_min(1e-8f);
  auto out = (fe * wp.unsqueeze(-1)).sum(1) / denom_w;
  auto any = (present.sum(1, true) > 0).to(out.dtype());
  return {out * any};
}
