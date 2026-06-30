#!/usr/bin/env python3
"""
Phase 1 - 단일 ray raymarch 샘플 점들의 전체-이미지 coverage 검사

흐름 (기존 LocalMap::sample 의 voxel raymarch 를 단순 재현):
  1. occ_occupied.ply 를 leaf_size voxel 로 양자화 -> occupied voxel set.
  2. occ grid 의 한 점 P 를 target 으로 고르고, 그 점을 (앞쪽+이미지안)으로 보는
     카메라 중 가장 가까운 것을 ray origin O 로 잡는다.  dir = (P-O)/|P-O|.
  3. O 에서 dir 로 raymarch: step 마다 점을 만들고, 그 점이 occupied voxel 안일
     때만 채택 (= occupied voxel 통과 점 샘플링).  이것이 sample 점들.
  4. 각 sample 점을 **전체 이미지(모든 pose)** 에 투영해 valid(z>0 & in-image)
     인 이미지 수를 센다 (subsample 없음).
  5. ray 를 따라 coverage 분포를 출력하고, target 카메라 이미지에 overlay 저장.

사용:
  venv/bin/python scripts/visual_feature/ray_sampling_coverage_test.py \
      --root data/iae_map --config config/fast_livo/iae.yaml --leaf-size 0.2
"""
import argparse
import datetime
import os
import re
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import camera_utils as cu          # noqa: E402
from visibility import valid_mask  # noqa: E402


def parse_leaf_size(config_path):
    txt = open(config_path).read()
    m = re.search(r"^\s*leaf_sizes\s*:\s*([-\d.eE+]+)", txt, re.MULTILINE)
    return float(m.group(1)) if m else 0.2


def voxel_key(pts, leaf):
    """[N,3] -> int64 voxel key (occupied set 용)."""
    q = np.floor(pts / leaf).astype(np.int64) + (1 << 20)
    return (q[:, 0] << 42) | (q[:, 1] << 21) | q[:, 2]


def coverage_all_images(P, poses, intr, distort=False):
    """점 P[M,3] 를 모든 pose 에 투영 -> 각 점이 valid 인 이미지 수 [M]."""
    M = P.shape[0]
    count = np.zeros(M, dtype=np.int32)
    for c2w in poses:
        u, v, z = cu.project(P, c2w, intr, distort)
        count += valid_mask(u, v, z, intr["W"], intr["H"])
    return count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/iae_map")
    ap.add_argument("--config", default="config/fast_livo/iae.yaml")
    ap.add_argument("--xyz", default=None,
                    help="occ ply (기본 <root>/occgrid_cache/occ_occupied.ply)")
    ap.add_argument("--leaf-size", type=float, default=None,
                    help="voxel 크기 (기본: config 의 leaf_sizes)")
    ap.add_argument("--point-idx", type=int, default=-1,
                    help="target occ 점 인덱스 (음수면 랜덤)")
    ap.add_argument("--step-frac", type=float, default=0.25,
                    help="raymarch step = step_frac * leaf_size")
    ap.add_argument("--voxel-sample-num", type=int, default=8,
                    help="occupied voxel 하나당 샘플 점 수 (LocalMap::sample 유사)")
    ap.add_argument("--distort", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--visibility", default=None,
                    help="precompute_visibility.py 의 npz. 주면 occlusion 반영 "
                         "(기본 <root>/occgrid_cache/visibility.npz 있으면 자동 사용)")
    ap.add_argument("--no-visibility", action="store_true",
                    help="visibility 캐시 무시하고 frustum-only 로")
    args = ap.parse_args()

    np.random.seed(args.seed)
    out = args.out or os.path.join(args.root, "phase1_viz")
    os.makedirs(out, exist_ok=True)
    leaf = args.leaf_size or parse_leaf_size(args.config)
    xyz_path = args.xyz or os.path.join(args.root, "occgrid_cache",
                                        "occ_occupied.ply")

    intr = cu.parse_intrinsics(args.config)
    poses = cu.load_poses(os.path.join(args.root, "color_poses.txt"))
    imgs = cu.list_images(os.path.join(args.root, "images"))
    n = min(len(poses), len(imgs))
    poses, imgs = poses[:n], imgs[:n]
    X, _ = cu.read_ply_xyz(xyz_path)
    print(f"[setup] occ pts={len(X)}  poses={len(poses)}  leaf={leaf}")

    # occupied voxel set
    occ_set = set(voxel_key(X, leaf).tolist())

    # occlusion-aware visibility 캐시 (precompute_visibility.py)
    vis_cache = None
    vis_path = args.visibility or os.path.join(args.root, "occgrid_cache",
                                               "visibility.npz")
    if not args.no_visibility and os.path.exists(vis_path):
        npz = np.load(vis_path)
        vis_cache = {"key2idx": {int(k): i for i, k in enumerate(npz["keys"])},
                     "vis": npz["vis"], "n_img": int(npz["n_img"])}
        print(f"[vis] occlusion 캐시 로드: {vis_path}  "
              f"(voxels={len(npz['keys'])}, n_img={vis_cache['n_img']})")
    else:
        print("[vis] occlusion 캐시 없음 -> frustum-only (화각 안이면 통과)")

    # ---- 1) target occ 점 P, ray origin O 선정 ----
    pidx = args.point_idx if args.point_idx >= 0 else \
        np.random.randint(len(X))
    P_target = X[pidx]

    # P_target 을 앞(+z)·이미지안 으로 보는 카메라 중 가장 가까운 것을 origin 으로
    best = None
    for i, c2w in enumerate(poses):
        u, v, z = cu.project(P_target[None], c2w, intr, args.distort)
        if z[0] > 0 and 0 <= u[0] < intr["W"] and 0 <= v[0] < intr["H"]:
            if best is None or z[0] < best[1]:
                best = (i, float(z[0]))
    if best is None:
        sys.exit("target 점을 보는 카메라가 없음 (occ 점이 아닐 수 있음).")
    cam_i, dist = best
    O = poses[cam_i][:3, 3]
    d = P_target - O
    d = d / np.linalg.norm(d)
    print(f"[ray] target occ idx={pidx} P={np.round(P_target,2)}")
    print(f"      origin cam frame={cam_i} O={np.round(O,2)} dist={dist:.2f}m")

    # ---- 2) raymarch: 통과하는 occupied voxel 들을 찾고, voxel 당 N점 샘플 ----
    step = args.step_frac * leaf
    ts = np.arange(0.0, dist * 1.2, step)
    ray_pts = O[None] + ts[:, None] * d[None]
    keys = voxel_key(ray_pts, leaf)
    occ_hit = np.array([k in occ_set for k in keys.tolist()])
    # ray 가 통과한 distinct occupied voxel (등장 순서 유지)
    hit_keys, first = np.unique(keys[occ_hit], return_index=True)
    hit_keys = hit_keys[np.argsort(first)]
    hit_centers = ray_pts[occ_hit][np.argsort(first)]
    n_vox = len(hit_keys)
    # 각 occupied voxel 안에서 voxel_sample_num 개 랜덤 점 (LocalMap::sample 유사)
    vox_origin = (np.floor(hit_centers / leaf)) * leaf  # voxel 하단 코너
    K = args.voxel_sample_num
    rand = np.random.rand(n_vox, K, 3) * leaf
    samples = (vox_origin[:, None, :] + rand).reshape(-1, 3)
    print(f"[raymarch] step={step:.3f}m  통과 occupied voxel {n_vox}개  "
          f"x {K}점/voxel = 샘플 {len(samples)}")
    if len(samples) == 0:
        sys.exit("occupied voxel 통과 샘플이 없음. step/leaf 확인.")

    # ---- 3) 각 샘플 점 coverage: frustum 전수검사 + (있으면) occlusion 반영 ----
    cov = coverage_all_images(samples, poses, intr, args.distort)  # frustum-only
    depth_along = np.linalg.norm(samples - O, axis=1)
    order = np.argsort(depth_along)

    # occlusion-aware: 각 샘플의 voxel -> 캐시의 image bitset
    vis_rows = None  # [num_samples, n_img] bool
    if vis_cache is not None:
        skeys = voxel_key(samples, leaf)
        idxs = np.array([vis_cache["key2idx"].get(int(k), -1) for k in skeys])
        vis_rows = np.zeros((len(samples), len(poses)), dtype=bool)
        known = idxs >= 0
        if known.any():
            bits = np.unpackbits(vis_cache["vis"][idxs[known]], axis=1)
            vis_rows[known] = bits[:, :len(poses)].astype(bool)
    cov_occ = vis_rows.sum(1) if vis_rows is not None else None

    print("\n========= 샘플 점별 coverage (전수검사) =========")
    print(f"전체 이미지 수      : {len(poses)}")
    print(f"샘플 점 수          : {len(samples)}")
    print(f"frustum coverage    min/median/max : "
          f"{cov.min()} / {np.median(cov):.0f} / {cov.max()}")
    if cov_occ is not None:
        print(f"occlusion coverage  min/median/max : "
              f"{cov_occ.min()} / {np.median(cov_occ):.0f} / {cov_occ.max()}  "
              f"(가림 통과)")
    print("\n  ray 진행순(가까운→먼) 점별 coverage:")
    for k in order:
        extra = f"  occl {cov_occ[k]:4d}" if cov_occ is not None else ""
        print(f"    t={depth_along[k]:6.2f}m  xyz={np.round(samples[k],2)}  "
              f"-> frustum {cov[k]:4d}{extra} 장")
    print("==========================================================\n")

    # ---- 4) 실행별 폴더 생성 + 점이 투영된(=포함된) 모든 이미지 저장 ----
    ts_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(out, f"pt{pidx}_{ts_str}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"[run] 출력 폴더: {run_dir}")

    import matplotlib.cm as cm
    cov_disp = cov_occ if cov_occ is not None else cov  # 컬러 기준
    col_all = (cm.turbo(np.clip(cov_disp / max(cov_disp.max(), 1), 0, 1))[:, :3]
               * 255).astype(np.uint8)  # 점별 coverage 컬러

    def draw_markers(im, ui, vi, col, r=4):
        for du in range(-r, r + 1):
            for dv in range(-r, r + 1):
                edge = max(abs(du), abs(dv)) >= r - 1
                xx = np.clip(ui + du, 0, im.shape[1] - 1)
                yy = np.clip(vi + dv, 0, im.shape[0] - 1)
                im[yy, xx] = (0, 0, 0) if edge else col
        return im

    # 샘플 점 ply 도 같은 폴더에
    cu.write_ply(os.path.join(run_dir, f"ray_samples_pt{pidx}.ply"),
                 samples, col_all)

    mode = "occlusion" if vis_rows is not None else "frustum"
    saved = 0
    for i in range(len(poses)):
        u, v, z = cu.project(samples, poses[i], intr, args.distort)
        m = valid_mask(u, v, z, intr["W"], intr["H"])    # 화각 안
        if vis_rows is not None:
            m = m & vis_rows[:, i]                        # + 가림 통과한 샘플만
        if not m.any():
            continue
        img = np.asarray(Image.open(imgs[i]).convert("RGB")).copy()
        ui, vi = np.round(u[m]).astype(int), np.round(v[m]).astype(int)
        draw_markers(img, ui, vi, col_all[m])
        Image.fromarray(img).save(
            os.path.join(run_dir, f"frame_{i:05d}_n{int(m.sum())}.png"))
        saved += 1
        if saved % 50 == 0:
            print(f"  saved {saved} ...")

    print(f"[done] [{mode}] 점이 실제로 보이는 이미지 {saved}장 저장 -> {run_dir}/")
    print(f"       (파일명 frame_<프레임idx>_n<보이는 샘플 수>.png)")


if __name__ == "__main__":
    main()
