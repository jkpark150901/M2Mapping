#!/usr/bin/env python3
"""
occ grid 를 state(type) 별로 색을 입혀 저장.

C++ build_occ_map 의 4-color export 가 occgrid_cache 에 xyz-only(무색) 로 저장한
type별 ply 를 읽어, 문서화된 색을 입혀 (a) type별 colored ply, (b) 하나로 합친
colored ply 를 만든다.  free / invisible_unknown 은 수천만 점이라 type별 subsample.

색 규약 (neural_mapping.cpp 의 4-color export 주석과 일치):
  occupied=red, free=green, visible_unknown=cyan, invisible_unknown=purple

사용:
  venv/bin/python scripts/visual_feature/export_occ_by_type.py --root data/iae_map
  venv/bin/python scripts/visual_feature/export_occ_by_type.py \
      --types occupied visible_unknown --max-pts 0   # 전체(서브샘플 없음)
"""
import argparse
import os
import re
import sys
from itertools import islice

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from camera_utils import write_ply  # noqa: E402

# type -> (파일 stem, RGB)
TYPES = {
    "occupied":          ("occ_occupied",          (220, 40, 40)),    # red
    "free":              ("occ_free",              (40, 200, 40)),    # green
    "visible_unknown":   ("occ_visible_unknown",   (40, 200, 220)),   # cyan
    "invisible_unknown": ("occ_invisible_unknown", (160, 60, 200)),   # purple
}


def read_ply_sub(path, max_pts):
    """ascii/binary ply 의 xyz 를 stride subsample 해서 [M,3] 로. return xyz, n_full."""
    f = open(path, "rb")
    header = b""
    while b"end_header" not in header:
        line = f.readline()
        if not line:
            raise RuntimeError(f"ply header 끝 못 찾음: {path}")
        header += line
    htxt = header.decode("ascii", "ignore")
    n = int(re.search(r"element vertex (\d+)", htxt).group(1))
    fmt = re.search(r"format\s+(\S+)", htxt).group(1)
    props = re.findall(r"property\s+\S+\s+(\S+)", htxt)

    stride = 1 if (max_pts <= 0 or n <= max_pts) else (n // max_pts + 1)

    if fmt == "ascii":
        rows = islice(f, 0, None, stride)
        xyz = np.array([ln.split()[:3] for ln in rows], dtype=np.float64)
    else:  # binary_little_endian, 모든 property float32 가정
        data = np.frombuffer(f.read(n * len(props) * 4), dtype="<f4")
        arr = data.reshape(n, len(props))[::stride]
        xi, yi, zi = (props.index(c) for c in ("x", "y", "z"))
        xyz = arr[:, [xi, yi, zi]].astype(np.float64)
    return xyz, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/iae_map")
    ap.add_argument("--cache", default=None,
                    help="occgrid_cache 경로(기본 <root>/occgrid_cache)")
    ap.add_argument("--types", nargs="+", default=list(TYPES),
                    choices=list(TYPES), help="저장할 type 들")
    ap.add_argument("--max-pts", type=int, default=500000,
                    help="type별 subsample 상한 (0=전체)")
    ap.add_argument("--no-combined", action="store_true",
                    help="합친 ply 생략(type별만)")
    ap.add_argument("--no-per-type", action="store_true",
                    help="type별 ply 생략(합친 것만)")
    ap.add_argument("--out", default=None,
                    help="출력 폴더(기본 <cache>/colored)")
    args = ap.parse_args()

    cache = args.cache or os.path.join(args.root, "occgrid_cache")
    out = args.out or os.path.join(cache, "colored")
    os.makedirs(out, exist_ok=True)

    all_xyz, all_rgb = [], []
    print(f"{'type':18} {'full':>12} {'saved':>10}  color")
    for t in args.types:
        stem, color = TYPES[t]
        path = os.path.join(cache, stem + ".ply")
        if not os.path.exists(path):
            print(f"{t:18} {'MISSING':>12}")
            continue
        xyz, n = read_ply_sub(path, args.max_pts)
        rgb = np.tile(color, (len(xyz), 1))
        print(f"{t:18} {n:>12,} {len(xyz):>10,}  {color}")

        if not args.no_per_type:
            write_ply(os.path.join(out, f"{stem}_colored.ply"), xyz, rgb)
        all_xyz.append(xyz)
        all_rgb.append(rgb)

    if not args.no_combined and all_xyz:
        X = np.concatenate(all_xyz, 0)
        C = np.concatenate(all_rgb, 0)
        cname = os.path.join(out, "occ_states_colored.ply")
        write_ply(cname, X, C)
        print(f"\n[combined] {len(X):,} pts -> {cname}")

    print(f"[done] -> {out}/")
    print("  red=occupied, green=free, cyan=visible_unknown, purple=invisible_unknown")


if __name__ == "__main__":
    main()
