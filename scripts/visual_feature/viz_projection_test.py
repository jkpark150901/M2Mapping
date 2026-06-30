#!/usr/bin/env python3
"""
Phase 1 시각화 테스트 (M2Mapping visual feature 계획서 "완료 조건" 검증용)

하는 일:
  1. 기존 샘플링이 raymarch 하는 occupied 점(occgrid_cache/occ_occupied.ply)을
     query xyz 로 로드. (= 기존 샘플링 방식과 동일한 xyz 소스)
  2. 각 점을 모든(샘플된) 카메라 view 에 투영, valid mask 로 "그 점을 포함한
     이미지" 를 탐색하고 multi-view coverage 를 센다.
  3. 시각화:
     (a) 몇 개 프레임에 투영점을 depth 컬러로 overlay -> PNG  (LiDAR/occ overlay)
     (b) 각 점의 coverage(보인 view 수) 를 컬러로 ply 저장 -> 3D 뷰어 점검
  4. 캘리브/visibility 통계 출력: behind-camera / out-of-image / valid 카운트.

사용:
  venv/bin/python scripts/visual_feature/viz_projection_test.py \
      --root data/iae_map --config config/fast_livo/iae.yaml

의존성: numpy, pillow, matplotlib (venv 에 이미 있음)
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import camera_utils as cu          # noqa: E402
import visibility as vis           # noqa: E402


def turbo(x):
    """[0,1] -> RGB uint8, matplotlib turbo."""
    import matplotlib.cm as cm
    return (cm.turbo(np.clip(x, 0, 1))[:, :3] * 255).astype(np.uint8)


def overlay(img, u, v, colors, radius=2):
    """이미지 numpy[H,W,3] 에 (u,v) 점들을 colors 로 splat (작은 사각형)."""
    H, W = img.shape[:2]
    ui, vi = np.round(u).astype(int), np.round(v).astype(int)
    for du in range(-radius, radius + 1):
        for dv in range(-radius, radius + 1):
            x = np.clip(ui + du, 0, W - 1)
            y = np.clip(vi + dv, 0, H - 1)
            img[y, x] = colors
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/iae_map", help="dataset 경로")
    ap.add_argument("--config", default="config/fast_livo/iae.yaml",
                    help="intrinsic 이 든 scene config")
    ap.add_argument("--xyz", default=None,
                    help="query xyz ply (기본: <root>/occgrid_cache/occ_occupied.ply)")
    ap.add_argument("--frames", type=int, default=200, help="coverage 계산 프레임 수")
    ap.add_argument("--overlay-frames", type=int, default=6, help="overlay PNG 장수")
    ap.add_argument("--max-pts", type=int, default=80000, help="xyz subsample 상한")
    ap.add_argument("--distort", action="store_true",
                    help="OpenCV radtan 왜곡 적용(원본 왜곡 이미지일 때)")
    ap.add_argument("--out", default=None, help="출력 폴더(기본 <root>/phase1_viz)")
    args = ap.parse_args()

    out = args.out or os.path.join(args.root, "phase1_viz")
    os.makedirs(out, exist_ok=True)
    xyz_path = args.xyz or os.path.join(args.root, "occgrid_cache",
                                        "occ_occupied.ply")

    # ---- 로드 ----
    intr = cu.parse_intrinsics(args.config)
    print(f"[intr] fx={intr['fx']:.2f} fy={intr['fy']:.2f} "
          f"cx={intr['cx']:.2f} cy={intr['cy']:.2f} {intr['W']}x{intr['H']} "
          f"model={intr['model']} distort={args.distort}")

    poses = cu.load_poses(os.path.join(args.root, "color_poses.txt"))
    imgs = cu.list_images(os.path.join(args.root, "images"))
    n = min(len(poses), len(imgs))
    poses, imgs = poses[:n], imgs[:n]
    print(f"[data] poses={len(poses)} images={len(imgs)}  (aligned n={n})")

    X, _ = cu.read_ply_xyz(xyz_path)
    if len(X) > args.max_pts:
        np.random.seed(0)
        X = X[np.random.choice(len(X), args.max_pts, replace=False)]
    print(f"[xyz] {xyz_path}\n      query points = {len(X)}")

    # ---- (2) multi-view coverage ----
    fidx = np.linspace(0, n - 1, min(args.frames, n)).astype(int)
    count, first_seen, per_frame = vis.multiview_coverage(
        X, poses, intr, fidx, distort=args.distort)

    seen = count > 0
    print("\n================ Phase1 visibility 통계 ================")
    print(f"query 점수            : {len(X)}")
    print(f">=1 view 에서 보임     : {seen.sum()} ({100*seen.mean():.1f}%)")
    print(f">=2 view 에서 보임     : {(count>=2).sum()} "
          f"({100*(count>=2).mean():.1f}%)")
    print(f"coverage(view수) median/mean/max : "
          f"{np.median(count):.0f} / {count.mean():.1f} / {count.max()}")
    pf = per_frame
    print(f"프레임당 평균 valid    : {np.mean([p['valid'] for p in pf]):.0f}")
    print(f"프레임당 평균 behind   : {np.mean([p['behind'] for p in pf]):.0f} "
          f"(카메라 뒤)")
    print(f"프레임당 평균 out      : {np.mean([p['out'] for p in pf]):.0f} "
          f"(앞이지만 이미지 밖)")
    print("=======================================================\n")

    # ---- (3b) coverage ply ----
    cov_norm = count / max(count.max(), 1)
    cu.write_ply(os.path.join(out, "coverage.ply"), X, turbo(cov_norm))
    print(f"[save] coverage.ply  (turbo: 파랑=적게 보임, 빨강=많이 보임)")

    # ---- (3a) overlay PNG: 점을 카메라-depth 컬러로 ----
    ov_idx = np.linspace(0, n - 1, args.overlay_frames).astype(int)
    for fi in ov_idx:
        img = np.asarray(Image.open(imgs[fi]).convert("RGB")).copy()
        u, v, z, m = vis.project_and_mask(X, poses[fi], intr, args.distort)
        if m.sum() == 0:
            print(f"[overlay] frame {fi}: valid 0 -> skip")
            continue
        zc = z[m]
        zn = (zc - zc.min()) / (np.ptp(zc) + 1e-9)
        img = overlay(img, u[m], v[m], turbo(zn))
        name = os.path.join(out, f"overlay_{fi:05d}.png")
        Image.fromarray(img).save(name)
        print(f"[overlay] frame {fi}: valid {int(m.sum())} -> "
              f"{os.path.basename(name)}")

    print(f"\n완료. 결과: {out}/")
    print("  - overlay_*.png : 점이 박스 엣지/구조물 위에 정확히 얹히는지 육안 확인")
    print("  - coverage.ply  : 멀티뷰 coverage 공간 분포 확인")


if __name__ == "__main__":
    main()
