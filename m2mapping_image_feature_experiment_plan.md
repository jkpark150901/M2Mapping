# M2Mapping Image Feature-Conditioned SDF 실험 및 코드 수정 계획

## 0. 목표

기존 M2Mapping의 geometry branch는 기본적으로 다음 구조에 가깝다.

```text
3D query point x
    ↓
Hash encoding H(x)
    ↓
SDF MLP
    ↓
sdf(x)
```

본 실험의 목표는 이미지 feature를 SDF branch에 직접 condition으로 넣었을 때, 기존 M2Mapping에서 관찰된 문제를 개선할 수 있는지 확인하는 것이다.

관찰된 문제:

- 박스 표면이 울퉁불퉁하게 복원됨
- 탁구대 다리 같은 얇은 구조물이 사라짐
- 학습을 오래 해도 SDF / rendering 품질 개선에 한계가 있음

핵심 가설:

```text
기존: sdf = f(H(x))
개선: sdf = f(H(x), F_img(x))
```

즉, LiDAR는 geometry supervision으로 유지하고, 이미지는 radiance 학습뿐 아니라 geometry branch의 입력 feature로 직접 활용한다.

---

## 1. 전체 실험 단계

| 단계 | 실험명 | 구조 | 목적 |
|---|---|---|---|
| 0 | Baseline | `sdf = f(H(x))` | 기존 M2Mapping 재현 |
| 1 | RGB concat | `sdf = f(H(x), RGB(x))` | 단순 픽셀 정보가 도움이 되는지 확인 |
| 2 | CNN feature concat | `sdf = f(H(x), F_cnn(x))` | 기존 image feature 방식 검증 |
| 3 | Multi-view feature concat | `sdf = f(H(x), Agg(F_i(pi_i(x))))` | SparseNeuS식 multi-view feature 효과 확인 |
| 4 | DINO / Depth feature | `sdf = f(H(x), F_dino/depth(x))` | geometry-aware feature 효과 확인 |
| 5 | Feature grid fusion | `sdf = f(H(x), G_img(x))` 또는 `sdf=f(G_img(x))` | GO-Surf식 3D feature grid 통합 |

우선 1차 목표는 **feature concat 방식으로 성능이 오르는지 확인**하는 것이다.  
처음부터 feature grid에 통합하면 실패 원인 분석이 어렵다.

---

## 2. 1차 실험 구조: Hash Feature + Image Feature Concat

### 2.1 Forward 구조

기존 M2Mapping:

```text
x
 ↓
HashEncoding H(x)
 ↓
SDF MLP
 ↓
sdf
```

수정 구조:

```text
image_i
 ↓
Image Encoder E
 ↓
2D feature map F_i

x
 ↓
projection pi_i(x)
 ↓
grid_sample(F_i, pi_i(x))
 ↓
multi-view aggregation
 ↓
F_img(x)

x
 ↓
HashEncoding H(x)

concat[H(x), F_img(x)]
 ↓
SDF MLP
 ↓
sdf
```

수식:

\[
z_x = H(x)
\]

\[
z_{img} = A\left(\{F_i(\pi_i(x))\}_{i=1}^{N}\right)
\]

\[
s = f_\theta([z_x, z_{img}])
\]

여기서:

- \(H(x)\): multi-resolution hash encoding
- \(F_i\): i번째 이미지의 2D feature map
- \(\pi_i(x)\): 3D query point를 i번째 image plane으로 projection
- \(A(\cdot)\): multi-view aggregation 함수
- \(f_\theta\): SDF MLP

---

## 3. Gradient 흐름 설계

### 3.1 1차 실험 권장 설정

처음에는 image encoder를 freeze한다.

```text
Image Encoder: frozen
Image Feature: stop-gradient
Hash Grid: trainable
SDF MLP: trainable
Radiance MLP: 기존 방식 유지
Camera Pose: fixed
```

즉, image feature는 학습 대상이 아니라 condition으로만 사용한다.

```python
with torch.no_grad():
    image_feat_maps = image_encoder(images)

z_img = sample_and_aggregate(image_feat_maps, x, poses, K)
z_img = z_img.detach()

z_hash = hash_encoding(x)
sdf = sdf_mlp(torch.cat([z_hash, z_img], dim=-1))
```

### 3.2 Gradient 흐름

| 모듈 | Gradient 업데이트 여부 | 설명 |
|---|---:|---|
| SDF MLP | O | SDF 예측 직접 학습 |
| Hash Grid | O | 기존 geometry latent feature 학습 |
| Image Encoder | X | feature extractor는 고정 |
| Projection / grid_sample | X | pose와 query point는 parameter가 아님 |
| Multi-view mean aggregation | X | parameter 없음 |
| Radiance MLP | O | 기존 rendering loss로 학습 |

### 3.3 End-to-end 학습은 나중에

image encoder까지 학습하면 LiDAR SDF loss가 image encoder까지 역전파된다.

문제:

```text
Image feature가 일반적인 visual representation을 유지하는 방향이 아니라,
현재 SDF loss만 줄이는 방향으로 망가질 수 있음.
```

따라서 실험 순서는 다음이 안전하다.

1. frozen encoder + concat
2. feature 효과 확인
3. 일부 layer fine-tuning
4. full end-to-end fine-tuning

---

## 4. Loss 구성

1차 실험에서는 기존 M2Mapping loss를 최대한 유지하고, SDF 입력만 바꾼다.

전체 loss:

\[
\mathcal{L}
=
\lambda_{surf}\mathcal{L}_{surf}
+
\lambda_{free}\mathcal{L}_{free}
+
\lambda_{eik}\mathcal{L}_{eik}
+
\lambda_{rgb}\mathcal{L}_{rgb}
\]

추가 실험에서만 feature consistency / plane / edge loss를 더한다.

---

## 5. Loss별 정리

| Loss | 수식 | 목적 | Gradient 흐름 |
|---|---|---|---|
| Surface Loss | \(\mathcal{L}_{surf}=|f_\theta([H(p_{hit}),F_{img}(p_{hit})])|\) | LiDAR hit point가 zero-level surface가 되게 함 | SDF MLP, Hash Grid |
| Free-space Loss | \(\mathcal{L}_{free}=\max(0,\delta-f_\theta([H(p_{ray}),F_{img}(p_{ray})]))\) | ray 중간을 free space로 만듦 | SDF MLP, Hash Grid |
| Eikonal Loss | \(\mathcal{L}_{eik}=(\|\nabla_x f_\theta(x)\|-1)^2\) | SDF 성질 유지 | SDF MLP, Hash Grid |
| RGB Rendering Loss | \(\mathcal{L}_{rgb}=\|C_{pred}-C_{gt}\|_1\) | 렌더링 색상 일치 | Radiance MLP, SDF branch 일부 |
| Feature Consistency | \(\mathcal{L}_{feat}=\|F_i(\pi_i(x))-F_j(\pi_j(x))\|_2^2\) | 같은 3D point의 multi-view feature 일치 | encoder freeze 시 업데이트 없음 |
| Normal Smoothness | \(\mathcal{L}_{normal}=\|n(x)-n(x+\epsilon)\|_2^2\) | 표면 요철 완화 | SDF MLP, Hash Grid |
| Plane Regularization | \(\mathcal{L}_{plane}=\sum_i(n^Tx_i+d)^2\) | 평면 영역 평탄화 | SDF MLP, Hash Grid |

---

## 6. Normal 관련 주의점

실제 GT normal이 없으므로 normal supervision은 사용할 수 없다.

불가능한 loss:

\[
\mathcal{L}_{normal\_gt}=1-n_{pred}^T n_{gt}
\]

가능한 loss:

\[
n(x)=\frac{\nabla f(x)}{\|\nabla f(x)\|}
\]

\[
\mathcal{L}_{normal}=\|n(x)-n(x+\epsilon)\|_2^2
\]

이 loss는 GT normal이 아니라 예측 normal끼리의 smoothness를 거는 것이다.

주의:

- 박스, 벽, 바닥 같은 평면에는 도움 가능
- 모서리, 얇은 구조물은 오히려 뭉갤 수 있음
- 따라서 1차 실험에는 넣지 않고, feature concat 성능 확인 후 추가하는 것이 좋음

---

## 7. Image Feature 추출 방식

### 7.1 가장 단순한 방식

```text
RGB image
 ↓
CNN encoder
 ↓
feature map [C, H', W']
```

3D query point x를 이미지에 투영한다.

```text
x_world
 ↓
T_cam_world
 ↓
x_cam
 ↓
K
 ↓
(u, v)
```

그리고 `grid_sample`로 feature를 뽑는다.

```python
feat = torch.nn.functional.grid_sample(
    feature_map,
    uv_grid,
    mode="bilinear",
    align_corners=True,
)
```

### 7.2 Multi-view aggregation

한 점 x가 여러 이미지에서 보이면:

\[
z_{img}(x)=A(f_1,f_2,...,f_N)
\]

1차 실험 추천:

```text
mean aggregation
```

후속 실험:

```text
visibility-weighted mean
max pooling
attention aggregation
```

---

## 8. Visibility 처리

이미지 feature를 사용할 때 모든 view를 쓰면 안 된다.

필터링 조건:

```text
1. projection된 uv가 image boundary 안에 있는가?
2. x_cam.z > 0 인가?
3. depth consistency가 맞는가?
4. M2Mapping visibility-aware occ grid상 unknown/free/occupied 상태가 어떤가?
```

1차 실험에서는 최소 조건만 사용한다.

```text
uv inside image
x_cam.z > 0
```

2차 실험에서 depth / occupancy 기반 visibility mask를 추가한다.

---

## 9. 코드 수정 계획

### 9.1 Dataset / DataLoader 수정

필요한 데이터:

```text
images
camera intrinsics K
camera extrinsics Tcw or Twc
query points x
LiDAR ray samples
RGB target
```

수정 사항:

- image tensor를 training batch에 포함
- camera pose와 intrinsics를 함께 전달
- query point가 어떤 image set에서 feature를 샘플링할지 결정

### 9.2 Image Encoder 추가

새 모듈 예시:

```python
class ImageFeatureEncoder(nn.Module):
    def __init__(self, backbone="resnet18", out_dim=32, freeze=True):
        super().__init__()
        self.encoder = build_backbone(backbone)
        self.proj = nn.Conv2d(backbone_dim, out_dim, kernel_size=1)
        if freeze:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, images):
        feat = self.encoder(images)
        feat = self.proj(feat)
        return feat
```

주의:

- 처음에는 `freeze=True`
- feature dimension은 16, 32 정도부터 시작
- 너무 큰 feature dimension은 SDF MLP를 불안정하게 만들 수 있음

### 9.3 Projection + Feature Sampling 모듈 추가

```python
class ImageFeatureSampler(nn.Module):
    def forward(self, points_world, feature_maps, intrinsics, extrinsics):
        # 1. world -> camera
        # 2. camera -> pixel
        # 3. normalize uv to [-1, 1]
        # 4. grid_sample
        # 5. visibility mask
        # 6. aggregate multi-view features
        return z_img, valid_mask
```

출력:

```text
z_img: [N_points, C_img]
valid_mask: [N_points]
```

### 9.4 SDF Network 입력 차원 수정

기존:

```python
sdf_in_dim = hash_feat_dim
```

수정:

```python
sdf_in_dim = hash_feat_dim + image_feat_dim
```

기존:

```python
z = hash_encoding(x)
sdf = sdf_net(z)
```

수정:

```python
z_hash = hash_encoding(x)
z_img, mask = image_feature_sampler(x, feat_maps, K, Tcw)
z = torch.cat([z_hash, z_img], dim=-1)
sdf = sdf_net(z)
```

### 9.5 Config 추가

예시:

```yaml
use_image_feature_sdf: true
image_feature:
  backbone: resnet18
  pretrained: true
  freeze: true
  out_dim: 32
  aggregation: mean
  detach: true
  use_visibility_mask: false
sdf_network:
  input_mode: hash_plus_image
```

---

## 10. 실험 순서

### Experiment 0: Baseline 재현

목표:

```text
기존 M2Mapping 결과 재현
```

확인 지표:

- mesh 품질
- 박스 표면 요철
- 탁구대 다리 유무
- RGB rendering PSNR / SSIM / LPIPS
- Chamfer distance 가능하면 측정

---

### Experiment 1: RGB Pixel Concat

구조:

\[
sdf=f(H(x),RGB(\pi(x)))
\]

목적:

```text
projection / sampling pipeline이 정상 작동하는지 확인
```

RGB 자체는 강한 feature가 아니므로 성능보다 코드 검증 목적이 큼.

---

### Experiment 2: Frozen CNN Feature Concat

구조:

\[
sdf=f(H(x),F_{cnn}(\pi(x)))
\]

설정:

```text
encoder frozen
feature detach
mean aggregation
기존 loss 유지
```

목적:

```text
이미지 feature가 SDF branch에 도움 되는지 확인
```

---

### Experiment 3: Multi-view Aggregation 개선

비교:

| Aggregation | 설명 |
|---|---|
| mean | 가장 단순 |
| max | 강한 edge/activation 보존 가능 |
| visibility-weighted mean | 관측 가능성이 높은 view 가중 |
| attention | 학습 가능하지만 불안정 가능 |

---

### Experiment 4: DINO / Depth Feature 교체

후보:

```text
DINOv2 feature
DepthAnything encoder feature
normal/depth prior feature
edge feature
```

목적:

```text
blurred CNN feature보다 geometry-aware visual feature가 더 좋은지 확인
```

---

### Experiment 5: Feature Grid Fusion

concat 실험에서 효과가 확인된 후 진행한다.

구조 후보:

```text
A. sdf = f(H(x), G_img(x))
B. sdf = f(G_img(x))
C. sdf = f(H(x) + Project(G_img(x)))
```

GO-Surf와 유사하게 image feature를 3D grid에 누적한 뒤 query point에서 interpolation한다.

---

## 11. 평가 항목

정량 평가:

| 항목 | 설명 |
|---|---|
| PSNR / SSIM / LPIPS | novel view rendering 품질 |
| Chamfer Distance | GT mesh / point cloud가 있을 때 geometry 오차 |
| F-score | geometry reconstruction 품질 |
| Normal Consistency | GT normal이 있을 때만 |
| Surface roughness | 평면 영역의 normal variance 또는 point-to-plane variance |
| Thin structure recall | 탁구대 다리 등 얇은 구조물 검출 여부 |

정성 평가:

```text
1. 박스 평면이 덜 울퉁불퉁한가?
2. 탁구대 다리가 살아나는가?
3. 모서리가 더 선명한가?
4. floating artifact가 늘어나지는 않는가?
5. RGB rendering이 오히려 나빠지지는 않는가?
```

---

## 12. 예상 리스크와 대응

| 리스크 | 원인 | 대응 |
|---|---|---|
| 성능 변화 없음 | image feature가 SDF와 연관 약함 | DINO/depth/edge feature로 교체 |
| geometry가 더 뭉개짐 | CNN feature가 blur됨 | high-res feature, shallow feature 사용 |
| 학습 불안정 | image feature dimension이 큼 | out_dim 16/32로 축소 |
| projection 오류 | pose/K mismatch | RGB concat 실험으로 먼저 검증 |
| occluded view feature 오염 | visibility mask 없음 | depth / occ grid 기반 visibility 추가 |
| thin structure 여전히 누락 | LiDAR supervision 부족 | edge-guided sampling, silhouette loss 추가 |

---

## 13. 최종 권장 구현 순서

1. 기존 M2Mapping baseline 결과 저장
2. RGB pixel projection + sampling pipeline 구현
3. `sdf=f(H(x), RGB(x))`로 코드 동작 확인
4. Frozen CNN encoder 추가
5. `sdf=f(H(x), F_cnn(x))` 실험
6. mean / max / visibility-weighted aggregation 비교
7. DINOv2 또는 DepthAnything feature로 교체
8. 성능 상승 확인 후 GO-Surf식 3D feature grid 통합 검토

---

## 14. 핵심 결론

지금 단계에서 가장 중요한 것은 좋은 구조를 한 번에 만드는 것이 아니라,

```text
이미지 feature가 SDF geometry branch에 실제로 도움이 되는가?
```

를 가장 작은 코드 수정으로 검증하는 것이다.

따라서 1차 실험은 반드시 다음 구조로 시작한다.

\[
sdf=f(H(x),F_{img}(x))
\]

그리고 image encoder는 freeze한다.

성능 향상이 확인되면 그 다음 단계로 GO-Surf식 feature grid fusion을 시도한다.
