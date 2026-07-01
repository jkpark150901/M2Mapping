# SparseNeuS 정리 (voxel 중심)

> 목적: SparseNeuS의 샘플링 / feature 추출 / 통합 / 최적화 구조를 voxel 단위로 정리.
> M2Mapping의 OGM에 "카메라 occlusion 체크 → 보이는 뷰에서 feature 추출 → 통합" 구조를
> 이식하기 위한 레퍼런스.

전제: **모든 것이 "voxel grid에 feature를 채워 conditional volume을 만들고, 임의의 3D
포인트는 그 volume을 보간해서 값을 얻는다"** 구조. 포인트는 volume의 소비자일 뿐,
geometry의 표현 단위는 voxel이다.

---

## 1. 샘플링 방식 (두 종류를 구분해야 함)

### (A) Voxel grid 샘플링 — feature volume을 만들기 위한 공간 샘플링
- coarse-to-fine LoD 구조. `self.lod`로 단계 제어. (`models/sparse_sdf_network.py:279` `get_conditional_volume`)
- **lod0 (coarse):** `generate_grid(vol_dims, 1)`로 공간 전체에 균일 voxel grid를 깐 뒤,
  카메라 frustum 밖 voxel을 `back_project_sparse_type(..., only_mask=True)`로 제거 → sparse해짐.
  (`sparse_sdf_network.py:312-327`)
- **lod1 (fine):** lod0에서 살아남은 voxel을 `upsample`로 8분할(octree subdivision)하고,
  다시 visibility로 필터링. (`sparse_sdf_network.py:329-346`, `upsample`은 L198)
- 핵심: 표면은 공간 대부분이 빈 얇은 manifold이므로, coarse에서 빈 공간을 쳐내고
  fine에서 표면 근처 voxel에만 연산을 집중 → O(N^3) 메모리 회피.

### (B) Ray 위 포인트 샘플링 — 렌더링 시
- NeuS 방식. ray마다 z_vals를 잡고, SDF 기반 importance sampling
  (`sparse_neus_renderer.py:78` `up_sample`)으로 표면 근처를 조밀하게 재샘플
  (`sparse_neus_renderer.py:122` `cat_z_vals`).
- **중요:** 샘플된 포인트는 항상 `get_pts_mask_for_conditional_volume`로
  valid voxel(=volume이 채워진 영역) 안에 있는지 마스킹됨. volume 밖 포인트는
  sdf=100(빈 공간)으로 처리. (`sparse_neus_renderer.py:159, 222-237`)

---

## 2. Feature 추출 방식 (voxel ← 이미지)

각 voxel 중심 좌표를 모든 입력 뷰에 투영해서 2D feature map을 샘플하는
**unprojection / back-projection**. (`ops/back_project.py:5` `back_project_sparse_type`)
- 입력: voxel 좌표 → world 좌표(`coords * voxel_size + origin`) → projection matrix로
  image UV → `grid_sample`로 뷰별 feature 추출.
- 출력: `(num_voxels, num_views, C)` feature + `(num_voxels, num_views)` visibility mask
  (`im_z>0` 그리고 UV가 [-1,1] 안).
- 2D feature는 사전에 FPN류 pyramid feature를 `compress_layer`로 압축한 것
  (`sparse_sdf_network.py:171, 304`).
- **OGM 아이디어와 직결:** "voxel을 카메라에 투영 → 보이는 뷰에서만 feature 추출"이
  정확히 이 함수의 동작. mask가 곧 visibility.
- **주의:** 이 코드의 mask는 frustum visibility일 뿐 **occlusion(앞에 다른 표면이 가림)은
  체크 안 함** — OGM에 추가하려는 부분이 바로 이것.

---

## 3. 통합 방식 (multi-view → 하나의 voxel feature)

### (A) Multi-view aggregation — variance/mean cost
(`sparse_sdf_network.py:221` `aggregate_multiview_features`)
- 한 voxel에 대해 여러 뷰 feature의 **평균과 분산**을 계산해 concat.
- 직관: 분산이 작다 = 뷰 간 일관(photo-consistency) = 그 voxel에 진짜 표면일 가능성↑. MVS 원리.
- `counts = 1/(보이는 뷰 수)`로 정규화하므로 **뷰 개수에 무관하게** 통합됨
  (permutation-invariant). 뷰 수가 가변이어도 OK — OGM 적용 시 장점.

### (B) 공간적 통합 — sparse 3D conv
(`sparse_sdf_network.py:177` `SparseCostRegNet`, `sparse_costreg_net`)
- aggregate된 voxel feature를 `SparseTensor`로 만들어 sparse 3D U-Net에 통과 →
  이웃 voxel 간 정보 전파(빈 곳 메우고 노이즈 정규화). 출력이 `regnet_d_out` 채널의 conditional volume.
- lod1에서는 이전 lod feature를 concat해서 입력(`cat([volume, up_feat])`, L353) →
  coarse context가 fine으로 전달.

### (C) sparse → dense + 보간
(`sparse_sdf_network.py:245` `sparse_to_dense_volume`)
- trilinear 보간을 쓰기 위해 dense volume `[1,C,X,Y,Z]`로 변환. 임의 포인트는
  `sdf()` (L376)에서 `grid_sample_3d(volume, pts)`로 feature 보간 후
  `LatentSDFLayer`(조건부 MLP)에 넣어 SDF + feature 출력.

---

## 4. 최적화 방식

**Volume rendering 기반, 미분가능.** SDF→alpha 변환은 NeuS 방식.
Loss는 `models/trainer_generic.py:715` `cal_losses_sdf`, 합산은 L827:

| Loss | 역할 |
|---|---|
| `color_fine_loss` (L1) | volume rendering 색 vs GT. 주 신호. |
| `color_mlp_loss` | blending rendering network 색(finetune 시). |
| `eikonal` (`gradient_error * sdf_igr_weight`) | \|∇SDF\|=1 강제. SDF를 진짜 distance field로. |
| `sparse_loss` (L800-803) | `exp(-\|sdf\|*decay)` 형태. 빈 공간 voxel의 SDF를 크게 밀어 표면을 희소하게. weight는 학습 후반에 점증(`get_weight`). |
| `fg_bg_loss` | (옵션) 전경/배경 마스크. |

- **학습 순서도 coarse-to-fine:** lod0 loss로 학습 후 lod1 추가(`trainer_generic.py:253, 308`).
- depth_loss는 inference 통계용, total loss에 미포함(`trainer_generic.py:794`).
- 두 가지 운용: **generic(generalizable)** 네트워크 사전학습 → 새 scene은 **finetune**으로
  conditional volume을 직접 최적화(`sparse_sdf_network.py:520` `FinetuneOctreeSdfNetwork`,
  TV regularizer 포함).

---

## OGM 적용 관점 메모 (목표와 매핑)

- **OGM voxel = SparseNeuS voxel grid**로 직접 대응. 이미 occupancy로 sparse하니
  frustum culling 단계를 OGM occupancy로 대체 가능.
- **occlusion 체크**가 이 코드엔 없음(frustum visibility만 있음). OGM은 occupancy를 알고
  있으니 **ray-marching/voxel traversal로 voxel→카메라 사이에 occupied voxel이 있으면
  그 뷰는 invisible** 처리하는 마스크를 `back_project_sparse_type`의 mask에 곱하면 됨.
  이게 핵심 추가분.
- 통합(variance/mean)은 뷰 수 무관 + occlusion mask와 자연스럽게 곱셈 결합되므로 그대로 재사용 가능.

---

## 핵심 파일/함수 인덱스

- `models/sparse_sdf_network.py`
  - `SparseSdfNetwork.get_conditional_volume` (L279): voxel volume 구축 메인
  - `aggregate_multiview_features` (L221): multi-view variance/mean 통합
  - `upsample` (L198): coarse→fine octree 8분할
  - `sparse_to_dense_volume` (L245): trilinear sampling용 변환
  - `sdf` (L376): 포인트 query (volume 보간 + MLP)
- `ops/back_project.py:5` `back_project_sparse_type`: voxel→이미지 feature 추출 + visibility mask
- `models/sparse_neus_renderer.py`: ray 샘플링 + volume rendering
- `models/trainer_generic.py:715` `cal_losses_sdf`: loss 정의
