"""
Phase 1 - 카메라 투영 유틸 (M2Mapping visual feature 계획서 camera_utils.py)

규약(C++ 코드/노트북과 일치):
  - color_poses.txt : [N,4,4] camera-to-world (T_wc). world = R @ p_cam + t
  - intrinsic       : OpenCV pinhole.  u = fx*Xc/Zc + cx,  v = fy*Yc/Zc + cy
  - 좌표계          : OpenCV (x right, y down, z forward).  z_c > 0 이 카메라 앞.
  - 이미지 i  <->  pose i  (파일명 숫자 정렬 순서로 1:1 정렬)

intrinsic은 config/fast_livo/iae.yaml 의 camera 블록에서 직접 파싱한다
(cv::FileStorage YAML 이라 PyYAML 로는 못 읽으므로 정규식으로 추출).
"""
import glob
import os
import re

import numpy as np


# ----------------------------------------------------------------------
# intrinsics
# ----------------------------------------------------------------------
def parse_intrinsics(config_path):
    """iae.yaml 류의 camera 블록 -> dict(fx,fy,cx,cy,W,H,model,dist[5])."""
    txt = open(config_path, "r").read()
    cam = txt[txt.find("camera:"):]  # camera 블록부터

    def g(key, default=0.0):
        m = re.search(rf"^\s*{key}\s*:\s*([-\d.eE+]+)", cam, re.MULTILINE)
        return float(m.group(1)) if m else float(default)

    intr = {
        "fx": g("fx"), "fy": g("fy"), "cx": g("cx"), "cy": g("cy"),
        "W": int(g("width")), "H": int(g("height")),
        "model": int(g("model", 0)),
        "dist": np.array([g("d0"), g("d1"), g("d2"), g("d3"), g("d4")],
                         dtype=np.float64),  # OpenCV 순서 (k1,k2,p1,p2,k3)
    }
    assert intr["fx"] > 0 and intr["W"] > 0, f"intrinsic 파싱 실패: {intr}"
    return intr


# ----------------------------------------------------------------------
# poses / images
# ----------------------------------------------------------------------
def load_poses(path):
    """color_poses.txt -> [N,4,4] camera-to-world."""
    return np.loadtxt(path).reshape(-1, 4, 4)


def _numkey(p):
    return int(re.search(r"(\d+)", os.path.basename(p)).group(1))


def list_images(img_dir, ext=("png", "jpg", "jpeg")):
    """숫자 정렬된 이미지 경로 리스트 (pose 인덱스와 1:1)."""
    files = []
    for e in ext:
        files += glob.glob(os.path.join(img_dir, f"*.{e}"))
    return sorted(files, key=_numkey)


# ----------------------------------------------------------------------
# ply IO (ascii / binary_little_endian, float xyz [+ rgb])
# ----------------------------------------------------------------------
def read_ply_xyz(path, want_rgb=False):
    f = open(path, "rb")
    header = b""
    while b"end_header" not in header:
        line = f.readline()
        if not line:
            raise RuntimeError(f"ply header 끝을 못 찾음: {path}")
        header += line
    htxt = header.decode("ascii", "ignore")
    n = int(re.search(r"element vertex (\d+)", htxt).group(1))
    props = re.findall(r"property\s+\S+\s+(\S+)", htxt)
    fmt = re.search(r"format\s+(\S+)", htxt).group(1)
    xi, yi, zi = props.index("x"), props.index("y"), props.index("z")

    if fmt == "ascii":
        skip = htxt[:htxt.find("end_header")].count("\n") + 1
        arr = np.loadtxt(path, skiprows=skip, max_rows=n).reshape(n, -1)
    else:  # binary_little_endian, 모든 property float32 가정
        data = np.frombuffer(f.read(n * len(props) * 4), dtype="<f4")
        arr = data.reshape(n, len(props))

    xyz = arr[:, [xi, yi, zi]].astype(np.float64)
    if want_rgb and "red" in props:
        ri, gi, bi = (props.index(c) for c in ("red", "green", "blue"))
        return xyz, arr[:, [ri, gi, bi]].astype(np.int32)
    return xyz, None


def write_ply(path, xyz, rgb=None):
    xyz = np.asarray(xyz, np.float32)
    with open(path, "wb") as f:
        hdr = ["ply", "format binary_little_endian 1.0",
               f"element vertex {len(xyz)}",
               "property float x", "property float y", "property float z"]
        if rgb is not None:
            hdr += ["property uchar red", "property uchar green",
                    "property uchar blue"]
        hdr.append("end_header\n")
        f.write(("\n".join(hdr)).encode())
        if rgb is None:
            f.write(xyz.tobytes())
        else:
            rgb = np.clip(np.asarray(rgb), 0, 255).astype(np.uint8)
            rec = np.empty(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"),
                                            ("z", "<f4"), ("r", "u1"),
                                            ("g", "u1"), ("b", "u1")])
            rec["x"], rec["y"], rec["z"] = xyz.T
            rec["r"], rec["g"], rec["b"] = rgb.T
            f.write(rec.tobytes())


# ----------------------------------------------------------------------
# projection: world xyz -> pixel (u, v) + camera-z
# ----------------------------------------------------------------------
def project(Xw, c2w, intr, distort=False):
    """
    Xw   : [N,3] world points
    c2w  : [4,4] camera-to-world (pose)
    return u[N], v[N], z[N]  (z = camera-frame depth, z>0 == 카메라 앞)
    distort=True 면 OpenCV radtan 왜곡을 적용해 '원본(왜곡) 이미지' 픽셀로 매핑.
    """
    w2c = np.linalg.inv(c2w)
    N = Xw.shape[0]
    Xh = np.concatenate([Xw, np.ones((N, 1))], 1)
    cam = (w2c @ Xh.T).T[:, :3]
    z = cam[:, 2]
    zs = np.where(np.abs(z) < 1e-9, 1e-9, z)
    xn, yn = cam[:, 0] / zs, cam[:, 1] / zs  # normalized image plane

    if distort:
        k1, k2, p1, p2, k3 = intr["dist"]
        r2 = xn * xn + yn * yn
        radial = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
        xd = xn * radial + 2 * p1 * xn * yn + p2 * (r2 + 2 * xn * xn)
        yd = yn * radial + p1 * (r2 + 2 * yn * yn) + 2 * p2 * xn * yn
        xn, yn = xd, yd

    u = intr["fx"] * xn + intr["cx"]
    v = intr["fy"] * yn + intr["cy"]
    return u, v, z
