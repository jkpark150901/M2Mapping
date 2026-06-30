# M2Mapping: Visual Feature Conditioned SDF 구현 계획

## 목표
기존 M2Mapping의 좌표/hash 기반 SDF에 IBRNet식 multi-view image feature를 condition으로 추가한다.

\[
\phi(x)=f_{\mathrm{sdf}}([H(x), f_{\mathrm{mv}}(x), c(x)])
\]

- \(H(x)\): 기존 hash encoding
- \(f_{\mathrm{mv}}(x)\): 여러 이미지에서 projection/sampling한 feature
- \(c(x)\): valid-view confidence

Full MVSNeRF cost volume은 계산량 때문에 제외한다.

## Phase 0. Baseline 고정
- 기존 M2Mapping checkpoint, config, seed 저장
- box plane / table leg ROI 지정
- mesh, RGB render, depth, normal, runtime 저장
- train/val/test trajectory block 고정

| 평가 | 지표 |
|---|---|
| RGB | PSNR, SSIM, LPIPS |
| Geometry | held-out LiDAR depth MAE, point-to-surface distance |
| Plane | plane fitting residual |
| Thin structure | completeness, silhouette overlap |
| Cost | iteration time, VRAM |

## Phase 1. Projection / visibility unit test
입력: RGB image \(I_i\), intrinsic \(K_i\), pose \(T_{cw}^{(i)}\), world query point \(x_w\).

\[
x_c=T_{cw}^{(i)}x_w,\qquad (u,v,1)^T\sim K_i x_c
\]

초기 valid mask:

\[
m_i(x)=\mathbb{1}[z_c>0]\cdot\mathbb{1}[(u,v)\in image]
\]

후속 확장:

\[
m_i(x)\leftarrow m_i(x)\cdot\mathbb{1}[|D_i(u,v)-z_c|<\tau]
\]

구현 파일:
- `camera_utils.py`
- `feature_projection.py`
- `visibility.py`

완료 조건:
- LiDAR point cloud를 RGB에 overlay
- box edge/table leg에서 projection 위치 확인
- image 밖, camera 뒤, occlusion point 확인
- timestamp offset 점검

## Phase 2. Frozen image feature cache
초기 encoder:
- pretrained ResNet-18/FPN
- 중간 feature map 하나
- frozen
- feature dim 16~64로 projection

각 image별 저장:
- `feature: [C,Hf,Wf]`
- `K`, `T_cw`, image size, timestamp

권장 cache:
`feature_cache/<scene>/frame_xxxxxx.pt`

## Phase 3. IBRNet식 query-time aggregation
\[
F_i=E_{\mathrm{img}}(I_i)
\]

\[
f_i(x)=\mathrm{grid\_sample}(F_i,\pi_i(x))
\]

초기 aggregation:

\[
f_{\mathrm{mv}}(x)=
\frac{\sum_i m_i(x)f_i(x)}
{\sum_i m_i(x)+\epsilon}
\]

\[
c(x)=\frac{1}{V}\sum_i m_i(x)
\]

feature가 없으면 zero feature와 `confidence=0`을 입력한다.

기존:
`SDF = MLP(hash_feature)`

변경:
`SDF = MLP(concat(hash_feature, mv_feature, confidence))`

### Gradient 정책
| 모듈 | 업데이트 |
|---|---:|
| Hash grid | O |
| SDF MLP | O |
| Image encoder | X |
| Feature cache | X |
| Camera pose | X |

\[
\frac{\partial\mathcal L}{\partial E_{\mathrm{img}}}=0
\]

Loss는 초기에 기존 M2Mapping loss만 사용한다.

\[
\mathcal L=
\lambda_{sdf}\mathcal L_{sdf}+
\lambda_{eik}\mathcal L_{eik}+
\lambda_{rgb}\mathcal L_{rgb}+
\lambda_{curv}\mathcal L_{curv}
\]

### Ablation
| ID | MV feature | Aggregation |
|---|---:|---|
| M0 | X | - |
| M1 | O | masked mean |
| M2 | O | max |
| M3 | O | confidence-weighted mean |

## Phase 4. View weighting
단순 평균 이후에만 추가한다.

\[
w_i(x)=\mathrm{softmax}_i(g_\psi([f_i(x),d_i,\theta_i,m_i,\Delta t_i]))
\]

\[
f_{\mathrm{mv}}(x)=\sum_i w_i(x)f_i(x)
\]

구현 순서:
1. angle/depth deterministic weight
2. small MLP weight predictor
3. attention은 마지막

## Phase 5. Occupancy-aligned sparse visual cache
M2Mapping의 active occupancy voxel에만 visual feature를 저장한다.

\[
V_{\mathrm{obs}}(v)=
\frac{\sum_i m_i(v)w_i(v)F_i(\pi_i(v))}
{\sum_i m_i(v)w_i(v)+\epsilon}
\]

query 시:

\[
f_{\mathrm{vis}}(x)=\mathrm{trilinear\_interpolation}(V_{\mathrm{obs}},x)
\]

\[
\phi(x)=f_{\mathrm{sdf}}([H(x),f_{\mathrm{vis}}(x),c(x)])
\]

저장값:
- feature float16
- confidence
- num_views
- last_update

후속 residual:
\[
V(x)=V_{\mathrm{obs}}(x)+\Delta V_\eta(x)
\]

### Cache Ablation
| ID | Query MV | Sparse visual cache |
|---|---:|---:|
| M0 | X | X |
| M1 | O | X |
| M4 | X | frozen |
| M5 | X | frozen + residual |
| M6 | O | frozen |

## 권장 실행 순서
1. baseline 고정
2. projection/visibility test
3. frozen cache
4. M1 masked mean
5. M2/M3 aggregation
6. deterministic weighting
7. learned weighting
8. sparse visual cache
9. residual cache
10. DINO/Depth feature 교체

## 주의점
- full MVS cost volume은 제외
- feature 없는 query는 hash-only fallback 유지
- calibration/visibility를 성능보다 먼저 검증
- geometry metric과 RGB metric을 분리해 판단
