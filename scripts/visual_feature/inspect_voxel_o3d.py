#!/usr/bin/env python3
"""
visibility_lut.bin 의 한 voxel 을 Open3D 로 3D 검증.

보여주는 것:
  - OGM 점군 (전체 occ voxel, 회색, subsample)
  - 선택 voxel = 와이어프레임 정육면체 (빨강 선)
  - 그 voxel 이 LUT 상 'visible' 인 카메라들:
      * frustum (초록 선)         - 카메라가 어디서 어느 방향으로 보는지
      * 카메라중심 -> voxel 시선 (파랑 선) - 관측 각도 확인 (정면/측면 의심 검증)

선이 보이게(LineSet) 그린다. 화면 없으면 geometry 를 .ply 로 저장.

사용:
  venv/bin/python scripts/visual_feature/inspect_voxel_o3d.py \
      --root /datasets/iae_5f/map_0410 --config config/fast_livo/iae.yaml \
      --voxel-idx 12345
의존성: pip install open3d
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import camera_utils as cu  # noqa: E402
import inspect_voxel_lut as ivl  # noqa: E402  (read_lut, read_llff, train_subset...)


def cube_lineset(center, leaf, color):
    import open3d as o3d
    c = ivl.voxel_corners(center, leaf)  # [8,3]
    edges = []
    for i in range(8):
        for b in (1, 2, 4):
            j = i ^ b
            if i < j:
                edges.append([i, j])
    ls = o3d.geometry.LineSet(
        o3d.utility.Vector3dVector(c),
        o3d.utility.Vector2iVector(np.array(edges)))
    ls.colors = o3d.utility.Vector3dVector(np.tile(color, (len(edges), 1)))
    return ls


def frustum_lineset(c2w, intr, scale, color):
    """카메라 frustum(중심+4코너) LineSet. c2w [4,4] camera-to-world."""
    import open3d as o3d
    R, t = c2w[:3, :3], c2w[:3, 3]
    fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
    W, H = intr["W"], intr["H"]
    px = [(0, 0), (W, 0), (W, H), (0, H)]
    dirs = np.array([[(p[0] - cx) / fx, (p[1] - cy) / fy, 1.0] for p in px])
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    far = (R @ (dirs.T * scale)).T + t          # [4,3] world
    pts = np.vstack([t[None], far])              # 0=center, 1..4=corners
    lines = [[0, 1], [0, 2], [0, 3], [0, 4],
             [1, 2], [2, 3], [3, 4], [4, 1]]
    ls = o3d.geometry.LineSet(
        o3d.utility.Vector3dVector(pts),
        o3d.utility.Vector2iVector(np.array(lines)))
    ls.colors = o3d.utility.Vector3dVector(np.tile(color, (len(lines), 1)))
    return ls


def sightline_lineset(centers, voxel_center, color):
    """각 카메라중심 -> voxel 중심 선."""
    import open3d as o3d
    n = len(centers)
    pts = np.vstack([centers, voxel_center[None]])
    lines = [[i, n] for i in range(n)]
    ls = o3d.geometry.LineSet(
        o3d.utility.Vector3dVector(pts),
        o3d.utility.Vector2iVector(np.array(lines)))
    ls.colors = o3d.utility.Vector3dVector(np.tile(color, (n, 1)))
    return ls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/iae_map")
    ap.add_argument("--config", default="config/fast_livo/iae.yaml")
    ap.add_argument("--lut", default=None)
    ap.add_argument("--voxel-idx", type=int, default=-1)
    ap.add_argument("--max-cams", type=int, default=30,
                    help="그릴 visible 카메라 수(0=전부)")
    ap.add_argument("--frustum-scale", type=float, default=0.0,
                    help="frustum 길이(m). 0=카메라-voxel 거리의 0.4배 자동")
    ap.add_argument("--leaf-size", type=float, default=None)
    ap.add_argument("--llff", type=int, default=-1)
    ap.add_argument("--no-window", action="store_true",
                    help="창 안 띄우고 ply 만 저장")
    args = ap.parse_args()

    try:
        import open3d as o3d
    except ImportError:
        sys.exit("open3d 필요: pip install open3d")

    lut_path = args.lut or os.path.join(args.root, "occgrid_cache",
                                        "visibility_lut.bin")
    L = ivl.read_lut(lut_path)
    N, C = L["N"], L["C"]
    v = args.voxel_idx if 0 <= args.voxel_idx < N else np.random.randint(N)
    center = L["xyz"][v]
    cams = ivl.visible_cams(L["lut"][v], C)
    leaf = args.leaf_size or ivl.parse_leaf_size(args.config)
    print(f"[lut] voxels={N} cameras={C}")
    print(f"[voxel] idx={v} center={np.round(center,3)} leaf={leaf}")
    print(f"[voxel] visible {len(cams)}/{C} cams -> {cams[:20]}"
          f"{' ...' if len(cams)>20 else ''}")

    out = os.path.join(args.root, "phase1_viz", f"voxel_{v}")
    os.makedirs(out, exist_ok=True)

    geoms = []

    # OGM 점군 (회색)
    X = L["xyz"]
    if N > 400000:
        X = X[np.random.choice(N, 400000, replace=False)]
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(X))
    pcd.paint_uniform_color([0.6, 0.6, 0.6])
    geoms.append(pcd)
    o3d.io.write_point_cloud(os.path.join(out, "ogm.ply"), pcd)

    # 선택 voxel 박스 (빨강 선)
    box = cube_lineset(center, leaf, [1, 0, 0])
    geoms.append(box)

    # 카메라 frustum + 시선
    poses_path = os.path.join(args.root, "color_poses.txt")
    if os.path.exists(poses_path):
        intr = cu.parse_intrinsics(args.config)
        raw = cu.load_poses(poses_path)
        n_raw = len(raw)
        llff = args.llff if args.llff >= 0 else ivl.read_llff(args.config)
        keep = ivl.train_subset(n_raw, llff)
        train = raw[keep]
        ok = len(keep) == C
        print(f"[align] raw={n_raw} llff={llff} -> train={len(keep)} "
              f"(C={C}) {'OK' if ok else 'MISMATCH!'}")
        sel = cams if args.max_cams <= 0 else cams[:args.max_cams]
        sel = [c for c in sel if c < len(train)]
        cam_centers = np.array([train[c][:3, 3] for c in sel]) if sel else \
            np.zeros((0, 3))
        all_lines = []
        for c in sel:
            c2w = train[c]
            dist = np.linalg.norm(c2w[:3, 3] - center)
            scale = args.frustum_scale if args.frustum_scale > 0 else 0.4 * dist
            fr = frustum_lineset(c2w, intr, scale, [0, 0.8, 0])
            geoms.append(fr)
            all_lines.append(fr)
        if len(cam_centers):
            sl = sightline_lineset(cam_centers, center, [0, 0.3, 1])
            geoms.append(sl)
            all_lines.append(sl)
        print(f"[draw] frustum {len(sel)}개 + 시선")
    else:
        print(f"[skip] {poses_path} 없음 -> 카메라 frustum 생략")

    # 좌표축
    geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=leaf * 5, origin=center - leaf * 2.5))

    # 선 geometry 저장(합쳐서 하나의 LineSet)
    line_geoms = [g for g in geoms if isinstance(g, o3d.geometry.LineSet)]
    if line_geoms:
        merged = line_geoms[0]
        for g in line_geoms[1:]:
            merged += g
        o3d.io.write_line_set(os.path.join(out, "voxel_cams_lines.ply"), merged)
    print(f"[save] {out}/ogm.ply, voxel_cams_lines.ply")

    if args.no_window:
        print("[done] --no-window: ply 저장만. 디스플레이 있는 곳에서 열어보세요.")
        return
    try:
        o3d.visualization.draw_geometries(
            geoms, window_name=f"voxel {v}  visible {len(cams)}/{C}")
    except Exception as e:
        print(f"[warn] 창 띄우기 실패({e}). ply 는 저장됨 -> {out}/")
        print("  디스플레이 있는 곳에서: ogm.ply + voxel_cams_lines.ply 열기")


if __name__ == "__main__":
    main()
