"""
Phase 1 - visibility / valid mask (계획서 visibility.py)

valid mask (Phase 1):
    m_i(x) = 1[z_c > 0] * 1[(u,v) in image]

후속 확장 (Phase 1.x, depth_test_mask):
    m_i(x) <- m_i(x) * 1[|D_i(u,v) - z_c| < tau]
    (per-frame LiDAR/render depth 가 있어야 하므로 여기선 hook 만 둠)
"""
import numpy as np

from camera_utils import project


def valid_mask(u, v, z, W, H):
    """z>0 이고 픽셀이 이미지 안 -> True. [N] bool."""
    return (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)


def project_and_mask(Xw, c2w, intr, distort=False):
    """한 프레임 투영 + valid mask. return u, v, z, mask."""
    u, v, z = project(Xw, c2w, intr, distort)
    m = valid_mask(u, v, z, intr["W"], intr["H"])
    return u, v, z, m


def multiview_coverage(Xw, poses, intr, frame_idx, distort=False, verbose=True):
    """
    각 점이 몇 개 view 에서 보이는지 센다.
    return:
      count      [N] int   : valid view 수
      first_seen [N] int   : 처음 보인 frame_idx (없으면 -1)
      per_frame  list of dict(behind, out, valid)  프레임별 카운트(캘리브 점검용)
    """
    N = Xw.shape[0]
    count = np.zeros(N, dtype=np.int32)
    first_seen = np.full(N, -1, dtype=np.int32)
    per_frame = []

    for fi in frame_idx:
        u, v, z = project(Xw, poses[fi], intr, distort)
        in_img = (u >= 0) & (u < intr["W"]) & (v >= 0) & (v < intr["H"])
        behind = z <= 0
        m = (~behind) & in_img

        count[m] += 1
        newly = m & (first_seen < 0)
        first_seen[newly] = fi

        per_frame.append({
            "frame": int(fi),
            "behind": int(behind.sum()),                 # 카메라 뒤
            "out": int((~behind & ~in_img).sum()),       # 앞이지만 이미지 밖
            "valid": int(m.sum()),                       # 보임
        })
        if verbose and len(per_frame) % 50 == 0:
            print(f"  coverage {len(per_frame)}/{len(frame_idx)} frames")

    return count, first_seen, per_frame


def depth_test_mask(z, sampled_depth, tau):
    """Phase 1.x hook: per-frame depth 와 비교해 occlusion 제거.
    sampled_depth: project 한 (u,v) 에서 grid_sample 한 D_i(u,v). z: camera-z."""
    return np.abs(sampled_depth - z) < tau
