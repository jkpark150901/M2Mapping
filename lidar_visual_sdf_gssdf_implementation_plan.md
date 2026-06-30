# LiDAR-Visual Feature 기반 SDF / Gaussian 실험 구현 계획

## 0. 목표

Sparse LiDAR만으로 geometry를 학습할 때 생기는 평면 요철, thin structure 누락을 보완하기 위해 multi-view RGB feature를 활용한다.

실험 축은 두 개로 분리한다.

| 축 | Backbone | 도입 방법 | 확인할 질문 |
|---|---|---|---|
| A | M2Mapping | IBRNet식 multi-view feature aggregation, 이후 MVSNeRF식 3D visual volume | visual feature가 SDF geometry를 개선하는가 |
| B | GS-SDF | Feature 3DGS식 Gaussian feature distillation | splat feature가 geometry-aware rendering과 thin structure 표현을 개선하는가 |

두 축은 동시에 변경하지 않는다.

---

# Part A. M2Mapping + IBRNet / MVSNeRF Feature Encoding

## A-0. Baseline 고정

### 수행 항목
- 기존 M2Mapping baseline 학습 및 checkpoint 저장
- 동일 scene, pose, train/val/test split 고정
- 박스 평면, 탁구대 다리 등 실패 ROI 지정
- LiDAR-RGB projection overlay로 extrinsic/timestamp 점검

### 저장 구조

```text
runs/
  baseline/
    config.yaml
    checkpoint/
    mesh/
    render_train/
    render_val/
    metrics.json
    roi_metrics.json
```

### 지표

| 대상 | 지표 |
|---|---|
| RGB | PSNR, SSIM, LPIPS |
| Geometry | held-out LiDAR depth MAE, point-to-surface distance |
| Mesh | Chamfer 또는 point-to-mesh distance |
| 평면 ROI | plane fitting residual / roughness |
| thin ROI | point completeness, silhouette overlap |

---

## A-1. 공통 projection / visibility interface

### 입력
- RGB image \(I_i\)
- camera intrinsic \(K_i\)
- camera pose \(T_{cw}^{(i)}\)
- world query point \(x_w\)

### Projection

\[
x_c=T_{cw}^{(i)}x_w
\]

\[
(u,v,1)^T \sim K_i x_c
\]

### valid-view mask

\[
m_i(x)=
\mathbb{1}[z_c>0]
\cdot
\mathbb{1}[(u,v)\in image]
\cdot
\mathbb{1}[visible/depth\ consistent]
\]

초기 버전은 positive depth + image bounds만 사용한다. 이후 depth consistency를 추가한다.

### 파일 제안

```text
data/
  camera_utils.py
  feature_projection.py
  visibility.py
```

### 핵심 함수

```python
def project_world_to_image(points_w, K, T_cw, image_hw):
    # uv_norm [N,2], camera_depth [N], in_bounds [N]
    ...
```

```python
def gather_valid_views(points_w, camera_batch, max_views):
    # view_ids [N,V], valid_mask [N,V], uv_norm [N,V,2]
    ...
```

### Unit test
- LiDAR point cloud를 RGB에 overlay
- box edge, table leg에서 projection pixel error 확인
- FOV 밖/카메라 뒤/occlusion sample을 별도 테스트

---

## A-2. IBRNet식 projection + multi-view feature aggregation

### 목적
가장 작은 구조 변경으로 visual feature가 SDF에 유효한지 확인한다.

### Forward

\[
F_i=E_{img}(I_i)
\]

\[
f_i(x)=grid\_sample(F_i,\pi_i(x))
\]

\[
f_{mv}(x)=A(\{f_i(x),m_i(x)\}_{i=1}^{V})
\]

\[
\phi(x)=f_{sdf}([H(x),f_{mv}(x),c(x)])
\]

- \(H(x)\): 기존 hash encoding
- \(c(x)\): valid view ratio 또는 confidence
- \(A\): mean, max, weighted mean, attention

### A-2-1. Image encoder

초기 권장:
- pretrained ResNet-18/FPN
- intermediate feature map 하나 사용
- **frozen**
- feature dim은 16~64로 projection

```python
image_encoder.eval()
for p in image_encoder.parameters():
    p.requires_grad = False
```

### A-2-2. Feature cache

각 image의 feature map을 사전 생성한다.

```text
cache/
  scene_x/
    frame_000123.pt
```

```python
{
  "feature": Tensor[C,Hf,Wf],
  "K": Tensor[3,3],
  "T_cw": Tensor[4,4],
  "image_size": (H,W)
}
```

### A-2-3. Feature aggregation

초기 mean aggregation:

\[
f_{mv}(x)=
rac{\sum_i m_i(x) f_i(x)}
{\sum_i m_i(x)+\epsilon}
\]

confidence:

\[
c(x)=rac{1}{V}\sum_i m_i(x)
\]

feature가 없으면:

```python
f_mv = torch.zeros(feature_dim)
confidence = 0.0
```

### A-2-4. SDF MLP 수정

기존:

```python
sdf = sdf_mlp(hash_feature)
```

변경:

```python
sdf_input = torch.cat(
    [hash_feature, mv_feature, confidence[..., None]], dim=-1
)
sdf = sdf_mlp(sdf_input)
```

### Gradient 정책

| 모듈 | 학습 |
|---|---|
| Hash grid | O |
| SDF MLP | O |
| Image encoder | X |
| Feature cache | X |
| Camera pose | X |

\[
rac{\partial\mathcal L}{\partial E_{img}}=0
\]

### Loss

초기에는 기존 M2Mapping loss만 유지한다.

\[
\mathcal L=
\lambda_{sdf}\mathcal L_{sdf}+
\lambda_{eik}\mathcal L_{eik}+
\lambda_{rgb}\mathcal L_{rgb}+
\lambda_{curv}\mathcal L_{curv}
\]

새 loss는 추가하지 않는다. 먼저 condition feature 자체의 효과를 확인한다.

### Ablation

| ID | Hash | MV feature | Encoder | Aggregation |
|---|---:|---:|---|---|
| A0 | O | X | - | - |
| A1 | O | O | frozen ResNet | mean |
| A2 | O | O | frozen ResNet | max |
| A3 | O | O | frozen ResNet | confidence-weighted mean |
| A4 | O | O | fine-tuned encoder | learned weighting |

### 성공 기준
- global geometry metric 유지 또는 개선
- plane roughness 감소
- thin-structure ROI completeness 향상
- image feature 없는 영역에서 성능 악화 없음
- VRAM/training time 증가량 기록

---

## A-3. IBRNet식 view weighting

### 목적
단순 평균이 occlusion, blur, view angle 차이를 섞는 문제를 완화한다.

각 view의 입력:

\[
q_i(x)=[f_i(x), d_i(x), 	heta_i(x),m_i(x),\Delta t_i]
\]

가중치:

\[
w_i(x)=softmax_i(g_\psi(q_i(x)))
\]

\[
f_{mv}(x)=\sum_i w_i(x)f_i(x)
\]

### 구현 순서
1. depth/angle 기반 deterministic weight
2. 작은 MLP weight predictor
3. transformer/attention은 마지막

### 주의
- RGB rendering 학습에서 source image와 target image를 분리하는 ablation 수행
- SDF 쪽은 valid mask와 calibration을 먼저 안정화

---

## A-4. MVSNeRF식 3D visual volume

### 목적
query마다 projection하는 비용과 feature missing을 줄이고, multi-view visual evidence를 3D 공간에 축적한다.

### 구조

\[
V_{vis}(x)=Fuse(\{F_i(\pi_i(x))\}_{i=1}^{V})
\]

\[
\phi(x)=f_{sdf}([H(x),V_{vis}(x),c(x)])
\]

### 권장 구현: sparse visual voxel grid

M2Mapping visibility-aware occupancy grid 또는 active voxel 영역을 scaffold로 사용한다.

각 active voxel center \(v\)에서:

1. 여러 view에 투영  
2. valid feature sample  
3. weighted aggregation  
4. visual feature와 confidence 저장  

\[
V_{vis}(v)=
rac{\sum_i m_i(v)w_i(v)F_i(\pi_i(v))}
{\sum_i m_i(v)w_i(v)+\epsilon}
\]

저장값:

```python
{
  "feature": float16[C],
  "confidence": float16[1],
  "num_views": uint8,
  "last_update": int32,
}
```

Query:

\[
f_{vis}(x)=trilinear\_interpolation(V_{vis},x)
\]

### Residual 확장

\[
V(x)=V_{obs}(x)+\Delta V_\eta(x)
\]

- \(V_{obs}\): frozen image-derived volume
- \(\Delta V\): trainable residual grid 또는 MLP

### Ablation

| ID | Hash | Pixel MV feature | Visual voxel grid |
|---|---:|---:|---:|
| A0 | O | X | X |
| A1 | O | O | X |
| A5 | O | X | O frozen |
| A6 | O | X | O + residual |
| A7 | O | O | O |

---

## A-5. M2Mapping split / 평가

### Data split
- 연속 frame 랜덤 split 금지
- trajectory 또는 viewpoint block 단위 분할
- test에는 held-out viewpoint 포함
- 가능하면 held-out LiDAR frame으로 geometry 평가

| Split | 권장 |
|---|---|
| Train | trajectory 60~70% |
| Validation | 다른 viewpoint 포함 10~20% |
| Test | 별도 trajectory block 15~20% |
| ROI test | box, table leg, edge 별도 |

---

# Part B. GS-SDF + Feature 3DGS

## B-0. 목적

GS-SDF의 SDF-constrained Gaussian geometry에 multi-view image-derived feature를 부여한다.

핵심 질문:

> Gaussian feature distillation이 thin structure와 geometry-aware rendering을 개선하는가?

---

## B-1. GS-SDF baseline

- 동일 scene에서 GS-SDF baseline 재현
- Gaussian count, densification/pruning schedule 기록
- SDF mesh와 Gaussian rendering 동시 저장

```text
runs/
  gssdf_baseline/
    gaussians.ply
    sdf_mesh.ply
    render_val/
    metrics.json
```

---

## B-2. 2D feature extractor

초기 후보:

| 후보 | 초기 권장 |
|---|---:|
| ResNet/FPN | O |
| DINOv2 | 후속 |
| DepthAnything feature | 후속 |
| diffusion U-Net latent | 마지막 |

초기에는 frozen encoder 한 종만 쓴다.

---

## B-3. Gaussian feature 초기화

각 Gaussian:

\[
g_k=\{\mu_k,\Sigma_k,lpha_k,c_k,z_k\}
\]

Gaussian center를 여러 image에 투영:

\[
(u_{ik},v_{ik})=\pi_i(\mu_k)
\]

\[
f_{ik}=F_i(u_{ik},v_{ik})
\]

초기 feature:

\[
z_k^{obs}=
rac{\sum_i m_{ik}w_{ik}f_{ik}}
{\sum_i m_{ik}w_{ik}+\epsilon}
\]

권장 parameterization:

\[
z_k=z_k^{obs}+\Delta z_k
\]

- \(z_k^{obs}\): frozen observation prior
- \(\Delta z_k\): trainable residual
- 초기에는 \(\Delta z_k=0\)

feature 없는 Gaussian은 zero feature + confidence 0으로 둔다.

---

## B-4. Feature Gaussian rendering

RGB splatting weight를 그대로 재사용해 feature를 렌더링한다.

\[
\hat F(p)=\sum_k T_k(p)lpha_k(p)z_k
\]

- \(p\): target pixel
- \(T_k\): transmittance
- \(lpha_k\): projected Gaussian opacity

구현 원칙:
- RGB renderer blending weight 재사용
- feature channel만 추가
- feature map 해상도는 RGB보다 낮게 시작
- VRAM 사용량 기록

---

## B-5. Feature distillation loss

\[
\mathcal L_{feat}=
rac{1}{|\Omega|}
\sum_{p\in\Omega}
m(p)\|\hat F(p)-F_{target}(p)\|_1
\]

또는 cosine:

\[
\mathcal L_{feat-cos}=
1-rac{\hat F(p)^TF_{target}(p)}
{\|\hat F(p)\|\|F_{target}(p)\|+\epsilon}
\]

전체 loss:

\[
\mathcal L=
\lambda_{rgb}\mathcal L_{rgb}+
\lambda_{sdf}\mathcal L_{sdf}+
\lambda_{eik}\mathcal L_{eik}+
\lambda_{align}\mathcal L_{align}+
\lambda_{feat}\mathcal L_{feat}
\]

SDF alignment:

\[
\mathcal L_{align}=
rac{1}{K}\sum_k |\phi(\mu_k)|
\]

### Shortcut 방지
- source view: Gaussian feature 초기화
- target view: feature rendering loss
- validation: 학습에 쓰지 않은 held-out camera frame
- image encoder: frozen

---

## B-6. GS-SDF 구현 단계

| 단계 | 설정 | 목적 |
|---|---|---|
| B1 | baseline GS-SDF | 재현 기준 |
| B2 | feature 초기화만 | projection/feature visualization 점검 |
| B3 | frozen feature + feature loss | rendering pipeline 검증 |
| B4 | residual feature 학습 | observation + adaptation |
| B5 | feature loss가 geometry에 주는 gradient ablation | geometry 기여 확인 |

### Geometry gradient ablation

| ID | feature loss → Gaussian position/scale/opacity gradient |
|---|---:|
| B4a | 차단 |
| B4b | position만 허용 |
| B4c | full 허용 |

초기 순서는 `B4a → B4b → B4c`.

---

## B-7. GS-SDF 지표

| 영역 | 지표 |
|---|---|
| RGB/NVS | PSNR, SSIM, LPIPS |
| Feature rendering | feature L1, cosine similarity |
| SDF/mesh | Chamfer, point-to-mesh distance |
| Geometry consistency | SDF-Gaussian alignment residual |
| Thin structure | ROI completeness, silhouette overlap |
| Cost | FPS, training time, VRAM, Gaussian count |

정성 결과:
- held-out view RGB
- rendered feature PCA map
- SDF mesh와 Gaussian overlay
- box plane close-up
- table leg close-up
- depth/normal map

---

# Part C. 공통 위험 요소

## C-1. Calibration / timestamp error

증상:
- projection feature가 물체 경계와 불일치
- visual feature 추가 후 geometry 악화
- multi-view aggregation이 흐림

대응:
- LiDAR point RGB overlay 생성
- box edge/table leg 기준 pixel error 확인
- timestamp offset 확인
- depth consistency/angle weighting 추가

## C-2. Feature missing

\[
\phi(x)=f(H(x),f_{vis}(x),c(x))
\]

- zero-fill + confidence 입력
- no-feature sample 비율 로그
- LiDAR-only 영역에서 hash-only fallback 확인

## C-3. Shortcut / overfit

증상:
- train rendering만 개선
- held-out view 악화
- feature loss만 감소하고 geometry 개선 없음

대응:
- source/target view 분리
- held-out viewpoint 평가
- frozen encoder 유지
- geometry 지표를 별도로 보고 판단

## C-4. Compute / VRAM

우선순위:
1. frozen feature cache
2. low-resolution feature map
3. max source view 제한
4. query batch chunking
5. float16 feature 저장
6. occupancy-aligned sparse volume
7. feature dim projection

---

# Part D. 권장 실행 순서

## M2Mapping
1. A0 baseline 고정
2. projection/valid mask unit test
3. A1 frozen ResNet + mean aggregation
4. A2/A3 aggregation 및 confidence ablation
5. learned view weighting
6. sparse visual voxel grid
7. visual grid residual
8. MVSNeRF식 cost volume은 마지막

## GS-SDF
1. B1 baseline 고정
2. B2 Gaussian feature 초기화 및 투영 시각화
3. B3 frozen feature rendering loss
4. B4 residual feature 학습
5. B5 geometry gradient ablation
6. DINO/depth/diffusion feature 교체 실험

---

# Part E. 실험 로그 템플릿

```markdown
## Experiment ID
- Date:
- Backbone:
- Scene:
- Seed:
- Commit:
- GPU / VRAM:

## Configuration
- Image encoder:
- Encoder frozen:
- Feature dimension:
- Source views:
- Aggregation:
- Visibility rule:
- SDF input:
- Loss weights:
- Train / Val / Test frames:

## Metrics
| Metric | Baseline | Experiment | Delta |
|---|---:|---:|---:|
| PSNR | | | |
| SSIM | | | |
| LPIPS | | | |
| Depth MAE | | | |
| Plane residual | | | |
| Thin ROI completeness | | | |
| Training time | | | |
| Peak VRAM | | | |

## Observation
- 개선:
- 악화:
- Projection / visibility 오류:
- 다음 수정:
```

---

# 최종 원칙

1. projection + aggregation부터 검증한다.
2. image feature 유효성 확인 전에는 3D volume/cost volume으로 가지 않는다.
3. M2Mapping과 GS-SDF는 독립 실험한다.
4. geometry 개선과 RGB/feature rendering 개선을 분리해 평가한다.
5. calibration, visibility mask, source-target 분리가 성능보다 우선이다.
