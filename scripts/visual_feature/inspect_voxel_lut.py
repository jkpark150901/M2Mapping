#!/usr/bin/env python3
"""
visibility_lut.bin 리더 + voxel 검사.

하는 일 (voxel 인덱스를 직접 선택):
  1. occgrid_cache/visibility_lut.bin 읽기 (voxel xyz + voxel×camera 비트마스크).
  2. 선택한 voxel 이 전체 OGM 에서 어디에 있는지:
       - occ_with_voxel.ply (전체 voxel 회색 + 선택 voxel 빨강)
       - topdown.png (XY 평면 산점도, 선택 voxel 강조)
  3. LUT 상 그 voxel 이 '보인다'고 된 카메라들에 voxel 중심을 투영해 overlay 저장
     -> 마커가 실제 표면 위에 얹히면 occlusion-aware LUT 가 맞다는 검증.

카메라 정렬: LUT 의 camera i = train 프레임(llff 로 매 8번째 제외)의 i 번째.
color_poses.txt(raw 전체)에서 동일 규칙으로 subset 을 복원해 매핑한다.

사용:
  venv/bin/python scripts/visual_feature/inspect_voxel_lut.py \
      --root /datasets/iae_5f/map_0410 --config config/fast_livo/iae.yaml \
      --voxel-idx 12345
"""
import argparse
import os
import re
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import camera_utils as cu  # noqa: E402
from visibility import valid_mask  # noqa: E402


def read_lut(path):
    with open(path, "rb") as f:
        N = int(np.fromfile(f, "<i8", 1)[0])
        C = int(np.fromfile(f, "<i8", 1)[0])
        nbytes = int(np.fromfile(f, "<i8", 1)[0])
        downscale = int(np.fromfile(f, "<i4", 1)[0])
        xyz = np.fromfile(f, "<f4", N * 3).reshape(N, 3).astype(np.float64)
        lut = np.fromfile(f, "<u1", N * nbytes).reshape(N, nbytes)
    return {"N": N, "C": C, "nbytes": nbytes, "downscale": downscale,
            "xyz": xyz, "lut": lut}


def visible_cams(lut_row, C):
    bits = np.unpackbits(lut_row, bitorder="little")[:C]  # C++: 1<<(i%8) in byte i//8
    return np.nonzero(bits)[0]


def read_llff(config_path):
    """config(및 scene_config)에서 llff 값을 읽는다. 못 찾으면 1."""
    def find(p):
        if not os.path.exists(p):
            return None
        txt = open(p).read()
        m = re.search(r"^\s*llff\s*:\s*(\d)", txt, re.MULTILINE)
        if m:
            return int(m.group(1))
        sc = re.search(r'scene_config\s*:\s*"?([^"\s]+)', txt)
        if sc:
            return find(os.path.join(os.path.dirname(p), sc.group(1)))
        return None
    v = find(config_path)
    return 1 if v is None else v


def train_subset(n_raw, llff):
    """C++ load_colors 규칙: llff 면 1-indexed i%8==0 프레임 제외(=0-idx p%8==7)."""
    if not llff:
        return list(range(n_raw))
    return [p for p in range(n_raw) if (p + 1) % 8 != 0]


def parse_leaf_size(config_path, default=0.2):
    """config(및 scene_config)에서 leaf_sizes 를 읽는다."""
    def find(p):
        if not os.path.exists(p):
            return None
        txt = open(p).read()
        m = re.search(r"^\s*leaf_sizes\s*:\s*([-\d.eE+]+)", txt, re.MULTILINE)
        if m:
            return float(m.group(1))
        sc = re.search(r'scene_config\s*:\s*"?([^"\s]+)', txt)
        if sc:
            return find(os.path.join(os.path.dirname(p), sc.group(1)))
        return None
    v = find(config_path)
    return default if v is None else v


def voxel_corners(center, leaf):
    """voxel(정육면체) 8 코너 [8,3]."""
    h = leaf / 2.0
    o = np.array([[sx, sy, sz] for sx in (-h, h) for sy in (-h, h)
                  for sz in (-h, h)])
    return center[None] + o


def convex_hull_2d(pts):
    """2D 점들의 convex hull (monotone chain), 순서대로 반환."""
    pts = sorted(map(tuple, pts))
    if len(pts) <= 2:
        return np.array(pts)
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lo = []
    for p in pts:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    hi = []
    for p in reversed(pts):
        while len(hi) >= 2 and cross(hi[-2], hi[-1], p) <= 0:
            hi.pop()
        hi.append(p)
    return np.array(lo[:-1] + hi[:-1])


def render_overlay(img, hull, save_path, color=(1, 0, 0), xlim=None, ylim=None):
    """이미지 위에 투영된 voxel 사각형(hull)을 채워서 저장. xlim/ylim 주면 확대."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    H, W = img.shape[:2]
    fig, ax = plt.subplots(figsize=(W / 160.0, H / 160.0), dpi=160)
    ax.imshow(img)
    ax.add_patch(Polygon(hull, closed=True, facecolor=color, edgecolor="yellow",
                         alpha=0.45, linewidth=1.5))
    ax.add_patch(Polygon(hull, closed=True, fill=False, edgecolor="yellow",
                         linewidth=1.5))
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)  # (ymax, ymin) for image coords
    ax.set_axis_off()
    fig.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/iae_map", help="dataset 경로")
    ap.add_argument("--config", default="config/fast_livo/iae.yaml")
    ap.add_argument("--lut", default=None,
                    help="기본 <root>/occgrid_cache/visibility_lut.bin")
    ap.add_argument("--voxel-idx", type=int, default=-1, help="음수면 랜덤")
    ap.add_argument("--max-overlay", type=int, default=20,
                    help="overlay 저장할 visible 카메라 최대 수 (0=전부)")
    ap.add_argument("--leaf-size", type=float, default=None,
                    help="voxel 크기(기본: config 의 leaf_sizes)")
    ap.add_argument("--zoom-threshold", type=float, default=40.0,
                    help="투영 사각형 최대변(px)이 이보다 작으면 확대본도 저장")
    ap.add_argument("--distort", action="store_true")
    ap.add_argument("--llff", type=int, default=-1,
                    help="train subset 규칙(-1=config 에서 자동)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    lut_path = args.lut or os.path.join(args.root, "occgrid_cache",
                                        "visibility_lut.bin")
    L = read_lut(lut_path)
    N, C = L["N"], L["C"]
    print(f"[lut] {lut_path}\n      voxels={N} cameras={C} "
          f"downscale={L['downscale']}")

    v = args.voxel_idx if 0 <= args.voxel_idx < N else np.random.randint(N)
    center = L["xyz"][v]
    cams = visible_cams(L["lut"][v], C)
    print(f"[voxel] idx={v}  center(world)={np.round(center,3)}")
    print(f"[voxel] visible 카메라 {len(cams)}/{C}  -> {cams[:20]}"
          f"{' ...' if len(cams)>20 else ''}")

    out = args.out or os.path.join(args.root, "phase1_viz",
                                   f"voxel_{v}")
    os.makedirs(out, exist_ok=True)

    # ---- (2) OGM 내 위치 ----
    # 전체 voxel 회색 + 선택 voxel 빨강 (subsample 해서 ply 크기 관리)
    X = L["xyz"]
    idx = np.arange(N)
    if N > 300000:
        idx = np.random.choice(N, 300000, replace=False)
    Xs = X[idx]
    col = np.tile([150, 150, 150], (len(Xs), 1)).astype(np.uint8)
    Xs = np.concatenate([Xs, center[None]], 0)
    col = np.concatenate([col, np.array([[255, 0, 0]], np.uint8)], 0)
    cu.write_ply(os.path.join(out, "occ_with_voxel.ply"), Xs, col)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 8))
        s = X[np.random.choice(N, min(N, 80000), replace=False)]
        plt.scatter(s[:, 0], s[:, 1], s=0.3, c="lightgray")
        plt.scatter([center[0]], [center[1]], s=120, c="red", marker="*")
        plt.gca().set_aspect("equal")
        plt.title(f"voxel {v}  (visible {len(cams)}/{C} cams)")
        plt.savefig(os.path.join(out, "topdown.png"), dpi=110,
                    bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"[warn] topdown 렌더 실패: {e}")

    # ---- (3) visible 카메라에 voxel 투영 overlay ----
    poses_path = os.path.join(args.root, "color_poses.txt")
    img_dir = os.path.join(args.root, "images")
    if not (os.path.exists(poses_path) and os.path.isdir(img_dir)):
        print(f"[skip] overlay: {poses_path} 또는 {img_dir} 없음 "
              f"(LUT 읽기/OGM 위치만 완료)")
        print(f"[done] -> {out}/")
        return

    intr = cu.parse_intrinsics(args.config)
    raw_poses = cu.load_poses(poses_path)
    raw_imgs = cu.list_images(img_dir)
    n_raw = min(len(raw_poses), len(raw_imgs))
    llff = args.llff if args.llff >= 0 else read_llff(args.config)
    keep = train_subset(n_raw, llff)
    train_poses = raw_poses[keep]
    train_imgs = [raw_imgs[p] for p in keep]
    print(f"[align] raw={n_raw} llff={llff} -> train={len(keep)} "
          f"(LUT C={C}) {'OK' if len(keep)==C else 'MISMATCH!'}")
    if len(keep) != C:
        print("[warn] train 프레임 수가 LUT C와 불일치 -> 카메라 인덱스 매핑이 "
              "어긋날 수 있음. --llff 확인.")

    leaf = args.leaf_size or parse_leaf_size(args.config)
    corners = voxel_corners(center, leaf)  # [8,3]
    print(f"[voxel] leaf_size={leaf}  -> 투영 사각형으로 표시")

    sel = cams if args.max_overlay <= 0 else cams[:args.max_overlay]
    saved = 0
    for ci in sel:
        if ci >= len(train_poses):
            continue
        u, v_, z = cu.project(corners, train_poses[ci], intr, args.distort)
        front = z > 0
        if front.sum() < 3:
            continue
        uv = np.stack([u[front], v_[front]], 1)
        # 중심이 화면 안인지(검증용)
        uc, vc, zc = cu.project(center[None], train_poses[ci], intr, args.distort)
        if not valid_mask(uc, vc, zc, intr["W"], intr["H"])[0]:
            print(f"[warn] cam {ci}: LUT엔 visible인데 중심이 frustum 밖")
        hull = convex_hull_2d(uv)
        img = np.asarray(Image.open(train_imgs[ci]).convert("RGB"))

        base = os.path.join(out, f"cam{ci:05d}.png")
        render_overlay(img, hull, base)

        # 투영 크기가 작으면 확대본
        w = uv[:, 0].max() - uv[:, 0].min()
        h = uv[:, 1].max() - uv[:, 1].min()
        if max(w, h) < args.zoom_threshold:
            cxp, cyp = uv[:, 0].mean(), uv[:, 1].mean()
            pad = max(40.0, 4.0 * max(w, h))
            render_overlay(img, hull,
                           os.path.join(out, f"cam{ci:05d}_zoom.png"),
                           xlim=(cxp - pad, cxp + pad),
                           ylim=(cyp + pad, cyp - pad))  # y 뒤집힘(이미지 좌표)
        saved += 1

    print(f"[done] voxel {v}: visible {len(cams)}개 중 {saved}장 overlay 저장")
    print(f"       -> {out}/")
    print("  occ_with_voxel.ply: 전체 OGM(회색)+해당 voxel(빨강)")
    print("  topdown.png       : XY 평면 위치")
    print("  cam*.png          : voxel 정육면체를 실제 크기로 투영한 사각형")
    print("  cam*_zoom.png     : 투영이 작을 때 확대본")


if __name__ == "__main__":
    main()
