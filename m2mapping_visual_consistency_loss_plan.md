# M2Mapping Multi-View Visual Consistency Loss 구현 계획

## 0. 한 줄 요약

이미지 feature를 SDF **입력으로 주입**(conditioning)하는 대신, **예측 표면점에서의 multi-view 시각 일관성(variance)을 loss로 추가**해서 geometry를 정련한다. SparseNeuS의 cost-variance 아이디어를 "조건"이 아니라 "감독 신호"로 쓰는 방식.

`get_sdf` / decoder / 데이터 파이프라인을 **건드리지 않는다**. 새 loss 항 하나만 추가.

---

## 1. 동기

- M2Mapping: LiDAR로 SDF를 학습 → coarse geometry는 좋지만 **얇은 구조/엣지/요철**에 한계.
- 이미지에는 그 디테일이 보임. 하지만 conditioning(입력 concat)은 get_sdf 공유·occlusion·free-space 등 설계 이슈가 많음 (별도 문서 참고).
- **더 단순한 길**: "올바른 표면이라면 여러 view에서 같은 곳이 보여야 한다"는 **multi-view photo-consistency**를 loss로 걸어 SDF가 표면을 일관된 위치로 밀게 한다.
- 이건 **differentiable MVS**를 LiDAR SDF 위에 얹는 것. Geo-NeuS / NeuralWarp 계열.

---

## 2. 핵심 아이디어

```
ray --(volume render)--> 예측 표면 깊이 D --> 표면점 p = o + D·d
p --(투영)--> view_1..view_K --> feature f_1..f_K
consistency loss = Var_k(f_k)        (낮을수록 일관 = 올바른 표면)
```

- **p는 SDF의 함수** (rendered depth가 alpha=σ(sdf) 가중합) → variance를 줄이면 **SDF가 표면을 photo-consistent한 곳으로 이동**.
- frozen feature(혹은 RGB)여도 **gradient는 투영 좌표 uv(=f(p)) 경유로 p까지** 흐른다 (`grid_sample`은 샘플 좌표에 대해 미분 가능). → encoder 학습 불필요.

---

## 3. Gradient 흐름

```
Var_k(f_k)
  │  d/d(uv)   ← grid_sample 좌표 미분 (feat_maps는 frozen, grad 안 감)
  ▼
uv_k = K·(w2c_k · p)
  │  d/dp
  ▼
p = origin + D·dir
  │  d/dD
  ▼
D = Σ w_i · depth_i,   w_i = render_weight(alpha_i),  alpha_i = σ(-sdf_i·isigma)
  │  d/d(sdf)
  ▼
sdf = decoder(hash(x))   → ✅ hash grid + decoder 업데이트
```

- ✅ grad 받음: **hash grid, SDF decoder** (rendered depth 경유)
- ❌ grad 안 감: image encoder / feat_maps (frozen) — 일관성은 "표면 위치"만 민다
- 즉 **이미지를 고정 reference로 두고, SDF가 거기에 맞춰 표면을 정렬**.

---

## 4. 수식

ray r의 예측 표면점:
\[
p_r = o_r + D_r \, d_r,\qquad D_r = \sum_i w_i\, t_i
\]

가시 view 집합 \(V_r\) (visibility mask)에서:
\[
\mu_r = \frac{1}{|V_r|}\sum_{k\in V_r} f_k(\pi_k(p_r)),\qquad
\mathcal{L}_{consist} = \frac{1}{|V_r|}\sum_{k\in V_r}\| f_k(\pi_k(p_r)) - \mu_r \|^2
\]

전체:
\[
\mathcal{L} = \mathcal{L}_{base} + \lambda_{consist}\,\mathcal{L}_{consist}
\]

- \(f_k\): view k의 시각 feature (Phase 0 = RGB 3채널, 이후 CNN/DINO)
- \(\pi_k\): world→view k 투영
- \(|V_r|<2\) 인 ray는 loss에서 제외 (variance 정의 불가)

---

## 5. M2Mapping 구현 위치

### 5.1 어디에 붙이나

`color_train_batch_iter` ([neural_mapping.cpp](include/neural_mapping/neural_mapping.cpp)) 안. 이미 여기서:
- ray를 만들고 (`color_ray_samples.origin/direction`)
- `tracer::render_ray(...)` → `trace_results`에 **rendered depth**가 들어있음.

→ rendered depth로 표면점 `p`를 만들고, consistency loss를 계산해 기존 color loss 옆에 더한다.

### 5.2 표면점 만들기

```cpp
// trace_results에서 rendered depth 추출 (render_depths)
auto D = trace_results[<depth_idx>];               // [N_ray, 1], grad 유지
auto p = color_ray_samples.origin + D * color_ray_samples.direction;  // [N_ray, 3]
```
※ `render_from_pts` 반환에서 depth 인덱스 확인 필요 ([tracer.cpp](include/tracer/tracer.cpp)). depth가 detach돼 있으면 grad 살리도록 수정.

### 5.3 variance 계산 — ImageFeatureField에 grad-enabled 메서드 추가

기존 `sample()`은 frozen/NoGrad라 loss에 못 씀. **새 메서드**를 추가:
```cpp
// grad가 xyz까지 흐르는 버전. detach/NoGrad 없음.
// returns: variance [M,1], valid_count [M,1]
std::vector<torch::Tensor> consistency_variance(const torch::Tensor &xyz);
```
내부는 `sample()`의 투영 루프 재사용하되:
- `torch::NoGradGuard` **제거** (grad 유지)
- mean/sq_sum으로 per-point variance 계산
- visibility mask로 `|V_r|<2`는 valid=false
- **stash/디버그 통계는 빼기** (.item() sync로 grad 끊기지 않게)

### 5.4 loss 합산

```cpp
auto cv = local_map_ptr->p_image_feat_field_->consistency_variance(p);
auto var = cv[0];                  // [N_ray,1]
auto valid = cv[1].squeeze() > 0;  // 2+ view
if (valid.any().item<bool>()) {
  auto consist_loss = var.index({valid}).mean();
  loss += k_consist_weight * consist_loss;
  llog::RecordValue("consist", consist_loss.item<float>());
}
```

---

## 6. Visibility / Occlusion

- **Phase 0**: frustum-only (`z>0` + uv 범위). 가림 무시. 빠른 검증.
- **Phase 1**: depth-band 추가 — view k의 depth와 `D` 비교해 가려진 view 제외. (per-view depth 필요 → bake 친화)
- variance는 **가려진 view가 섞이면 오염**되므로, depth-band가 일관성 loss 품질을 크게 좌우함. 단 Phase 0로 파이프라인/효과부터 확인.

---

## 7. 코드 변경 체크리스트

| 파일 | 변경 |
|---|---|
| `image_feature_field.h/.cpp` | `consistency_variance(xyz)` 추가 (grad 유지, NoGrad/stash 제외) |
| `neural_mapping.cpp` `color_train_batch_iter` | rendered depth → 표면점 p → consistency loss 합산 |
| `tracer.cpp` (필요시) | rendered depth가 detach면 grad 유지하도록 노출 |
| `params.h/.cpp` | `k_consist_weight` 추가 (config `consist_weight`) |
| `config/fast_livo/fast_livo.yaml` | `consist_weight: 0.0`(기본) — scene config임에 주의 |

**주의**: consistency loss는 `use_image_feature`(conditioning)와 **독립**. feat_maps(이미지)만 ImageFeatureField가 들고 있으면 되므로, conditioning은 꺼두고 consistency만 켤 수 있게 분리한다.
→ ImageFeatureField init 조건을 `k_use_image_feature || k_consist_weight>0`로.

---

## 8. Config

```yaml
# fast_livo.yaml (scene config — read_scene_params가 읽음)
consist_weight: 0.05        # 0 = 끔. 작게 시작 (base loss 압도 금지)
image_feature_dim: 3        # Phase 0 = RGB
image_feature_max_views: 64 # consistency도 view subsample 영향 받음 (coverage 주의)
# use_image_feature 는 0이어도 됨 (conditioning과 독립)
```

---

## 9. 학습 스케줄 권장

- **초반엔 끄고**, LiDAR로 coarse geometry가 잡힌 뒤(예: 30~50%) consistency를 점진 활성화.
  - 이유: 표면이 엉뚱한 곳에 있으면 variance gradient가 노이즈. rgb_weight warmup과 같은 사상.
  - 구현: `train_callback`에서 `k_consist_weight = k_consist_weight_end * smoothstep(k_t)` 형태.
- `k_truncated_dis` 정도의 band/스케일로 시작.

---

## 10. 평가

| 지표 | 기대 |
|---|---|
| depth 메트릭 (rendered vs GT LiDAR depth) | ↑ (geometry 직접 개선) |
| mesh Chamfer / F-score | ↑ |
| 얇은 구조 recall (탁구대 다리 등) | ↑ (정성) |
| 평면 roughness (normal variance) | ↓ |
| color PSNR/SSIM/LPIPS | 유지 또는 소폭 ↑ |
| `consist` loss 곡선 | 감소 추세 (수렴 확인) |

baseline(consist_weight=0) vs 켠 것 비교가 핵심. **total loss 말고 depth/mesh 지표로 판정** (rgb_weight 스케줄에 묻히지 않게).

---

## 11. 리스크 & 대응

| 리스크 | 원인 | 대응 |
|---|---|---|
| 표면이 더 망가짐 | 초반 표면 부정확 → variance grad 노이즈 | warmup 후 활성화, 작은 weight |
| view-dependent 오염 | RGB는 specular/조명 의존 | Phase 1에서 DINO/CNN feature로 교체 |
| 가림 오염 | occlusion 미처리 | depth-band 추가 (Phase 1) |
| coverage 부족 | max_views subsample | max_views↑ 또는 bake |
| rendered depth가 blurry | volume 기대깊이 평활 | sphere-trace depth 사용 검토 |
| grad 안 흐름 | depth/feature가 어디서 detach | 경로 점검 (render depth, grid_sample 좌표) |

---

## 12. 단계별 진행

1. **Phase 0**: RGB + frustum-only + warmup. `consistency_variance` 구현, color_train_batch_iter에 loss 추가. baseline 대비 depth/mesh 비교.
2. **Phase 1**: depth-band(occlusion) 추가 → variance 정제.
3. **Phase 2**: RGB → CNN/DINO feature 교체 (view-dependent 제거).
4. **Phase 3**: 효과 확인되면 conditioning(input 주입)과 결합 검토.

---

## 13. 핵심 포인트 재확인

- **get_sdf/decoder 안 건드림** → conditioning보다 안전·단순.
- **frozen feature여도 gradient OK** (투영 좌표 경유) → encoder 학습 불필요.
- gradient는 **rendered depth → SDF**로 흘러 표면을 photo-consistent하게 정렬.
- **depth-band / view coverage / warmup**이 품질의 3대 변수.
- 평가는 **depth·mesh 지표** 중심 (total loss 아님).
