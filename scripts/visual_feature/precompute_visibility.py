#!/usr/bin/env python3
"""
Phase 1.x - occ voxel x image visibility 마스크 사전 계산 + 캐시

아이디어(사용자 제안):
  occ grid 의 각 occupied voxel 에 대해 "어느 이미지에서 (가림 고려) 보이는지" 를
  미리 계산해 둔다. raymarch 때 샘플 점의 voxel 을 찾아 이 마스크를 조회하면,
  frustum-only(화각 안이면 다 통과) 가 아니라 occlusion 을 근사 반영한 visibility
  를 얻는다.

방법: per-camera z-buffer (occlusion 근사)
  각 카메라에 모든 occupied voxel 중심을 투영 -> 다운샘플 격자에서 픽셀당 최근접
  voxel 만 visible 로 표시. (가까운 표면이 먼 voxel 을 가린다)
    visible(v, i)  <=>  z_v <= min_cell_depth(픽셀셀) + tol

출력 (npz): <root>/occgrid_cache/visibility.npz
  keys      [Nv] int64   voxel key (= voxel_key(center))
  centers   [Nv,3] f32   voxel 중심 (world)
  vis       [Nv, ceil(Nimg/8)] uint8   packbits image visibility
  leaf, n_img, downscale, tol, W, H

C++ 포팅 메모: build_occ_map 의 occupied voxel 확보 직후 같은 z-buffer 를
occgrid_cache 에 함께 저장하면 학습에서도 동일 마스크를 쓸 수 있다.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import camera_utils as cu  # noqa: E402


def voxel_key(qv):
    """[N,3] int64 voxel coords -> int64 key (precompute/test 공통 규약)."""
    q = qv + (1 << 20)
    return (q[:, 0] << 42) | (q[:, 1] << 21) | q[:, 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/iae_map")
    ap.add_argument("--config", default="config/fast_livo/iae.yaml")
    ap.add_argument("--xyz", default=None)
    ap.add_argument("--leaf-size", type=float, default=None)
    ap.add_argument("--downscale", type=int, default=4,
                    help="z-buffer 격자 다운샘플(클수록 빠르고 occlusion 보수적)")
    ap.add_argument("--tol-frac", type=float, default=1.0,
                    help="depth tolerance = tol_frac * leaf_size")
    ap.add_argument("--distort", action="store_true")
    ap.add_argument("--max-cams", type=int, default=0, help="0=전체, >0 디버그용")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import re
    leaf = args.leaf_size
    if leaf is None:
        m = re.search(r"^\s*leaf_sizes\s*:\s*([-\d.eE+]+)",
                      open(args.config).read(), re.MULTILINE)
        leaf = float(m.group(1)) if m else 0.2

    xyz_path = args.xyz or os.path.join(args.root, "occgrid_cache",
                                        "occ_occupied.ply")
    out = args.out or os.path.join(args.root, "occgrid_cache", "visibility.npz")

    intr = cu.parse_intrinsics(args.config)
    poses = cu.load_poses(os.path.join(args.root, "color_poses.txt"))
    if args.max_cams > 0:
        poses = poses[:args.max_cams]
    X, _ = cu.read_ply_xyz(xyz_path)

    # occ 점 -> 고유 voxel (중심, key)
    qv = np.floor(X / leaf).astype(np.int64)
    keys_all = voxel_key(qv)
    keys, first = np.unique(keys_all, return_index=True)
    centers = ((qv[first].astype(np.float64)) + 0.5) * leaf
    Nv, Ni = len(centers), len(poses)
    print(f"[setup] occ pts={len(X)} -> voxels={Nv}  images={Ni}  leaf={leaf}")
    print(f"[setup] z-buffer downscale={args.downscale} tol={args.tol_frac*leaf:.3f}m")

    W, H, s = intr["W"], intr["H"], args.downscale
    Wc, Hc = (W + s - 1) // s, (H + s - 1) // s
    tol = args.tol_frac * leaf
    nbytes = (Ni + 7) // 8
    vis = np.zeros((Nv, nbytes), dtype=np.uint8)

    for i, c2w in enumerate(poses):
        u, v, z = cu.project(centers, c2w, intr, args.distort)
        valid = (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if not valid.any():
            continue
        vi = np.nonzero(valid)[0]
        cu_ = (u[vi] / s).astype(np.int64)
        cv_ = (v[vi] / s).astype(np.int64)
        cell = cv_ * Wc + cu_
        # z-buffer: 셀별 최근접 depth
        zbuf = np.full(Hc * Wc, np.inf)
        np.minimum.at(zbuf, cell, z[vi])
        # 가려지지 않은(=셀 최근접 ± tol) voxel 만 visible
        vis_local = z[vi] <= zbuf[cell] + tol
        vidx = vi[vis_local]
        vis[vidx, i >> 3] |= np.uint8(1 << (i & 7))
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{Ni} cams")

    seen_cnt = np.unpackbits(vis, axis=1)[:, :Ni].sum(1)
    print(f"[stat] voxel당 visible 이미지 수: "
          f"median {np.median(seen_cnt):.0f}  mean {seen_cnt.mean():.1f}  "
          f"max {seen_cnt.max()}")
    print(f"[stat] 한 번도 안 보인 voxel: {(seen_cnt==0).sum()} / {Nv}")

    np.savez_compressed(out, keys=keys, centers=centers.astype(np.float32),
                        vis=vis, leaf=leaf, n_img=Ni, downscale=s, tol=tol,
                        W=W, H=H)
    print(f"[save] {out}  ({os.path.getsize(out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
