# M2Mapping × SparseNeuS: OGM 기반 multi-view feature 적용 계획

> 근거: [SparseNeuS_voxel_summary.md](3rdparty(reference)/SparseNeuS/docs/SparseNeuS_voxel_summary.md)
> 목표: SparseNeuS의 "voxel→이미지 back-project → 보이는 뷰에서 feature 추출 → mean/var 통합
> → conditional volume → 임의 포인트 trilinear query"를 M2Mapping OGM 위에 이식.
> **핵심 차별점:** SparseNeuS는 frustum visibility만 쓰지만, 우리는 이미 만든
> **occlusion-aware visibility LUT** 로 가려진 뷰를 제외한다(= 문서가 지목한 "핵심 추가분").

---

## 0. SparseNeuS → M2Mapping 매핑 (이미 있는 것 / 만들 것)

| SparseNeuS 요소 | M2Mapping 현황 | 작업 |
|---|---|---|
| lod0 voxel grid + frustum culling | **OGM occupied(+visible_unknown) voxel** (occgrid_cache `qpts.pt`) | 이미 sparse — culling 불필요. 그대로 voxel set으로 사용 |
| visibility mask (frustum만) | **`visibility_lut.bin`** (per-voxel×camera, **occlusion 반영**) | 이미 있음 ✅ — SparseNeuS보다 정확 |
| back_project_sparse_type (voxel→UV→grid_sample) | C++ 투영(`visibility_lut.cpp`), Python(`camera_utils.project`) | 2D feature map에 grid_sample 추가 |
| 2D feature (FPN+compress) | 없음 | Phase별로 RGB→frozen CNN→DINO |
| aggregate_multiview_features (mean/var) | (롤백된 `ImageFeatureField` out_dim=2C+1과 동일 개념) | mean/var/coverage 통합 재사용 |
| SparseCostRegNet (sparse 3D conv) | 없음 | Phase 3(옵션, spconv/Minkowski 필요) |
| sparse→dense + trilinear query | hash grid + octree(`p_acc_strcut_`) 존재 | sparse trilinear(8-이웃 voxel gather) 구현 |
| conditional volume → SDF MLP | **get_sdf 주입 훅** (이전에 만들었다 롤백) | concat 주입부 부활 |
| volume rendering / eikonal / sparse loss | sphere tracing + `loss::sdf_loss`/eikonal/rgb 존재 | 기존 loss 유지 + (옵션) consistency |

**한 줄 구조:** `sdf = decoder( concat[ hash(x), trilinear(V_feat, x), coverage(x) ] )`
여기서 `V_feat`는 OGM voxel에 bake된 multi-view 통합 feature.

---

## 1. 전체 파이프라인

```
[전처리 1회 bake]  (occ-build 직후 또는 Python)
  OGM voxel 중심 ──proj──> 각 카메라 2D feature map ──grid_sample──> f_i (보이는 뷰만, LUT)
                                       │
                                mean / var / coverage 통합  (뷰수 무관)
                                       ▼
                          V_feat: voxel별 [2C+1] feature  → voxel_features.bin
   (옵션 Phase3) sparse 3D conv 로 이웃 전파 → conditional volume

[학습/쿼리]  get_sdf(x):
  z_hash = hash(x)
  z_img, cov = trilinear(V_feat, x)        # 8-이웃 occupied voxel gather
  sdf = decoder( concat[z_hash, z_img, cov] )
  (cov=0 인 곳은 hash-only fallback)

[loss]  기존 LiDAR sdf + eikonal + rgb (+ 옵션: var 기반 photo-consistency, sparse loss)
```

---

## 2. 단계별 계획 (위험 낮은 순)

### Phase 0 — baseline 고정 (이미 가능)
- 현 M2Mapping(feature 없음) mesh/PSNR/depth/runtime 저장. ROI(박스면, 얇은 구조) 지정.

### Phase 1 — RGB mean/var bake + frozen 주입  ★ 최소 변경, 먼저
- **bake (Python, `scripts/visual_feature/`)**: `visibility_lut.bin`(occlusion) + `color_poses.txt` + 이미지로
  각 voxel을 보이는 뷰에 투영 → RGB를 `grid_sample` → **mean(3)+var(3)+coverage(1)=7ch** → `voxel_features.bin`.
  - 이미 만든 LUT/projection/리더를 그대로 재사용. 인코더 없음.
- **주입 (C++)**: `LocalMap::get_sdf`에서 voxel_features를 trilinear interp → decoder 입력 concat
  (롤백한 ImageFeatureField 주입부를 "on-the-fly 투영" 대신 "baked grid trilinear"로 바꿔 부활).
  - decoder 입력차원 += 7. feature는 **frozen**(grad 안 감) — prior 붕괴 방지.
- **검증**: baseline 대비 depth MAE / mesh Chamfer / 박스 roughness / 얇은 구조 recall.
- 목적: "occlusion-aware multi-view feature가 geometry에 도움 되나"를 가장 싸게 확인.

### Phase 2 — frozen CNN/FPN feature로 교체
- per-view 2D feature map을 frozen ResNet/FPN(+1×1 compress, 16~32ch)으로. bake/주입 구조 동일.
- RGB→CNN으로 정보량↑. 여전히 frozen, var는 photo-consistency 신호 유지.

### Phase 3 — 공간 전파 (sparse 3D conv, 옵션)
- aggregate된 voxel feature를 sparse 3D U-Net(spconv/Minkowski 또는 경량 3D conv)으로 정규화 → conditional volume.
- 의존성 큼 → Python 전처리(bake 단계)에서 수행해 결과만 `voxel_features.bin`에 저장하면 C++ 무변경.

### Phase 4 — trainable feature (prior 붕괴 방지)
- voxel feature를 학습 대상으로 풀되, **bake값으로 당기는 anchor regularizer** `‖V-V_bake‖`.
- 또는 작은 embedding MLP만 학습(feature 고정) — 우리가 이전에 합의한 형태.

### Phase 5 — coarse→fine voxel (SparseNeuS lod1)
- 표면 근처 occupied voxel을 octree 8분할(`spc_ops`/`points_to_corners` 활용)해 feature 해상도↑.

---

## 3. 핵심 구현 포인트 / 결정사항

1. **trilinear over sparse voxels (가장 중요한 신규 구현)**
   - dense volume은 bbox/leaf³ 라 메모리 폭발(map_0410 ~ 수억 voxel) → **불가**.
   - 점 x → 주변 8 voxel 좌표 양자화 → 각 voxel을 voxel→feature 테이블에서 조회
     (octree `p_acc_strcut_->query` 또는 voxel-key 해시) → valid weight로 trilinear blend.
   - 빠진 이웃(occupied 아님)은 weight 0, 전부 0이면 coverage=0 → hash-only fallback.
   - Phase 1은 우선 **nearest-voxel**(불연속)로 빠르게 검증 후 trilinear로 승격 가능.

2. **bake 위치**: Python(이미 LUT/이미지 도구 있음)에서 `voxel_features.bin` 생성이 가장 빠름.
   포맷은 `visibility_lut.bin`과 같은 스타일(voxel xyz + feature). 학습용으로 안정화되면 C++ occ-build로 포팅.

3. **occlusion mask 결합**: `back_project`의 frustum mask × **LUT visibility** → 가려진 뷰 제외.
   mean/var는 뷰수 무관(`1/cnt` 정규화)이라 가변 뷰·occlusion mask와 곱셈 결합이 자연스러움.

4. **gradient 정책 (초기)**: encoder/feature frozen, hash grid·SDF MLP만 학습. (SparseNeuS finetune은 Phase 4)

5. **loss**: 기존 `sdf+eikonal+rgb` 유지. 추가 후보 —
   - var 기반 **photo-consistency**(다른 plan 문서의 consistency loss와 동일 아이디어),
   - SparseNeuS **sparse_loss** `exp(-|sdf|·decay)`로 빈 공간 SDF 밀기(얇은 구조 보존에 도움 가능).

---

## 4. 평가
- geometry: held-out LiDAR depth MAE, mesh Chamfer/F-score, 평면 roughness, 얇은 구조 recall.
- appearance: PSNR/SSIM/LPIPS.
- cost: iter time, VRAM, bake 시간/용량.
- **반드시 baseline(feature off) vs on을 같은 seed/iter로 비교**, total loss 말고 depth/mesh로 판정.

## 5. 권장 진행 순서
1. Phase 1 bake(Python, RGB mean/var, occlusion LUT) → `voxel_features.bin`
2. trilinear(or nearest) query + get_sdf concat(frozen) 부활
3. baseline 대비 geometry/RGB 비교
4. 효과 확인 시 Phase 2(CNN) → Phase 4(trainable+anchor) → Phase 3/5
