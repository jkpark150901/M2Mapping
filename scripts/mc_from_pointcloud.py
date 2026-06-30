#!/usr/bin/env python3
"""
Point cloud (.ply / .pcd) -> Marching Cubes mesh.

M2Mapping 프로젝트의 SDF marching cubes( LocalMap::meshing_ )와 동일한 규약을 따른다:
  - voxel grid 해상도 = export_resolution (기본 0.04)
  - level set = 0.0
  - truncation = 3 * leaf_size  (params.cpp: k_truncated_dis = 3 * k_leaf_size)
  - grid 바깥/빈 voxel 은 +truncation(양수)로 채워 표면 밖으로 밀어냄
    (프로젝트는 1e-6 양수로 채우지만, 여기선 신경 SDF가 없으므로 +trunc 사용)

프로젝트와의 차이(불가피):
  프로젝트는 학습된 neural SDF( get_sdf )를 샘플링하지만, 여기엔 그 네트워크가
  없으므로 입력 point cloud로부터 oriented-normal 기반 signed distance field를
  직접 만들어 동일한 해상도/level/truncation 으로 marching cubes 한다.

의존성:
  pip install open3d scikit-image scipy numpy
"""
import argparse
import sys
import numpy as np


# ---- 프로젝트 기본값 (config/fast_livo/fast_livo.yaml, *.yaml, params.cpp) ----
DEFAULT_EXPORT_RES = 0.04   # export_resolution (fast_livo). replica=0.01
DEFAULT_LEAF_SIZE = 0.2     # leaf_sizes (iae/station/cbd/sysu/campus). 씬마다 다름
DEFAULT_LEVEL = 0.0         # marching cubes iso-level
TRUNC_MULT = 3.0            # k_truncated_dis = 3 * k_leaf_size


def load_points(path):
    import open3d as o3d
    if path.lower().endswith(".pcd"):
        pcd = o3d.io.read_point_cloud(path)
    else:
        pcd = o3d.io.read_point_cloud(path)  # open3d auto-detects ply/pcd/xyz...
    if len(pcd.points) == 0:
        raise RuntimeError(f"빈 point cloud 이거나 읽기 실패: {path}")
    return pcd


def ensure_oriented_normals(pcd, leaf_size, knn=30):
    import open3d as o3d
    has_normals = pcd.has_normals() and len(pcd.normals) == len(pcd.points)
    if not has_normals:
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=2.0 * leaf_size, max_nn=knn))
    # MC가 닫힌 표면을 내려면 normal 방향이 일관돼야 함
    try:
        pcd.orient_normals_consistent_tangent_plane(knn)
    except Exception as e:
        print(f"[warn] normal orientation 실패({e}); 방향 비일관 → 표면 구멍 가능",
              file=sys.stderr)
    return pcd


def build_sdf_grid(points, normals, res, trunc, margin):
    """oriented normal 기반 signed distance field를 voxel grid에 채운다."""
    from scipy.spatial import cKDTree

    lo = points.min(0) - margin
    hi = points.max(0) + margin
    dims = np.ceil((hi - lo) / res).astype(int) + 1  # (nx, ny, nz)
    print(f"[grid] bbox={lo} ~ {hi}  res={res}  dims={tuple(dims)} "
          f"(voxels={np.prod(dims):,})")

    # grid vertex 좌표
    xs = lo[0] + np.arange(dims[0]) * res
    ys = lo[1] + np.arange(dims[1]) * res
    zs = lo[2] + np.arange(dims[2]) * res
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    verts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)

    tree = cKDTree(points)
    sdf = np.full(verts.shape[0], trunc, dtype=np.float32)  # 기본 = 표면 밖(+)

    chunk = 2_000_000
    for s in range(0, verts.shape[0], chunk):
        e = min(s + chunk, verts.shape[0])
        d, idx = tree.query(verts[s:e], k=1, workers=-1)
        near = d <= trunc                      # truncation band 안만 갱신
        ii = idx[near]
        diff = verts[s:e][near] - points[ii]   # vertex - nearest point
        signed = np.einsum("ij,ij->i", diff, normals[ii])  # point-to-plane
        sdf[s:e][near] = np.clip(signed, -trunc, trunc).astype(np.float32)

    return sdf.reshape(dims), lo


def main():
    ap = argparse.ArgumentParser(
        description="point cloud(.ply/.pcd) -> marching cubes mesh "
                    "(M2Mapping 기본 파라미터)")
    ap.add_argument("input", help="입력 point cloud (.ply / .pcd)")
    ap.add_argument("-o", "--output", default=None,
                    help="출력 mesh .ply (기본: 입력명_mesh.ply)")
    ap.add_argument("--res", type=float, default=DEFAULT_EXPORT_RES,
                    help=f"marching cubes 해상도 = export_resolution "
                         f"(기본 {DEFAULT_EXPORT_RES})")
    ap.add_argument("--leaf-size", type=float, default=DEFAULT_LEAF_SIZE,
                    help=f"leaf_sizes. truncation=3*leaf_size 계산용 "
                         f"(기본 {DEFAULT_LEAF_SIZE})")
    ap.add_argument("--level", type=float, default=DEFAULT_LEVEL,
                    help=f"iso-level (기본 {DEFAULT_LEVEL})")
    ap.add_argument("--knn", type=int, default=30,
                    help="normal 추정/orientation 이웃 수 (기본 30)")
    args = ap.parse_args()

    try:
        import open3d as o3d
        from skimage import measure
    except ImportError as e:
        sys.exit(f"의존성 누락: {e}\n  pip install open3d scikit-image scipy numpy")

    trunc = TRUNC_MULT * args.leaf_size
    margin = args.leaf_size
    out = args.output or (args.input.rsplit(".", 1)[0] + "_mesh.ply")

    print(f"[load] {args.input}")
    pcd = load_points(args.input)
    pcd = ensure_oriented_normals(pcd, args.leaf_size, args.knn)

    points = np.asarray(pcd.points, dtype=np.float64)
    normals = np.asarray(pcd.normals, dtype=np.float64)
    print(f"[load] points={len(points):,}  trunc={trunc:.3f}  margin={margin:.3f}")

    sdf, lo = build_sdf_grid(points, normals, args.res, trunc, margin)

    # level 이 grid 값 범위 밖이면 MC가 빈 mesh를 냄 -> 사전 점검
    if not (sdf.min() < args.level < sdf.max()):
        sys.exit(f"[error] level={args.level} 가 SDF 범위 "
                 f"[{sdf.min():.3f}, {sdf.max():.3f}] 안에서 교차하지 않음. "
                 f"--res / --leaf-size 조정 필요.")

    verts, faces, normals_v, _ = measure.marching_cubes(
        sdf, level=args.level, spacing=(args.res, args.res, args.res))
    verts += lo  # grid local -> world

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()

    o3d.io.write_triangle_mesh(out, mesh)
    print(f"[done] vertices={len(verts):,} faces={len(faces):,} -> {out}")


if __name__ == "__main__":
    main()
