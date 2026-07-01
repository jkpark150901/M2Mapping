#!/usr/bin/env python3
"""
DINOv2 (또는 다른 백본) 로 train 뷰별 feature map 을 미리 뽑아 저장.
C++ ImageFeatureField(init, backbone==2) 가 이 파일을 로드해 conditional volume 을 bake 한다.

- 대상 뷰 = train split (llff 로 매 8프레임 held-out 제외). LUT/train_color_ 와 동일 순서.
- DINOv2 patch feature -> PCA 로 image_feature_dim(예 16) 차원 압축 -> patch 해상도(H/14,W/14) 저장.
- C++ 는 파일의 (Hs,Ws) 로 intrinsic 스케일을 자동 유도하므로 해상도는 여기서 자유.

출력 (occgrid_cache/image_features.bin, little-endian):
  int64 V; int64 C; int64 Hs; int64 Ws;   float32 data[V*Hs*Ws*C]  (C-order: V,Hs,Ws,C)

사용:
  venv/bin/python scripts/visual_feature/precompute_image_features.py \
      --root /datasets/iae_5f/map_0410 --config config/fast_livo/iae.yaml --dim 16
의존성: torch, torchvision, pillow, numpy  (DINOv2 는 torch.hub 로 자동 다운로드)
"""
import argparse
import os
import struct
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import camera_utils as cu          # noqa: E402
import inspect_voxel_lut as ivl    # read_llff, train_subset  # noqa: E402


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def load_dino(model_name, device):
    import torch
    model = torch.hub.load("facebookresearch/dinov2", model_name)
    model.eval().to(device)
    return model


def dino_feat(model, img_path, H14, W14, patch, device):
    """이미지 -> DINOv2 patch feature [Hp, Wp, Cdino] (torch, device)."""
    import torch
    im = Image.open(img_path).convert("RGB").resize((W14, H14), Image.BILINEAR)
    arr = (np.asarray(im, np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)  # [1,3,H,W]
    with torch.no_grad():
        out = model.forward_features(t)["x_norm_patchtokens"]  # [1, Np, C]
    Hp, Wp = H14 // patch, W14 // patch
    return out.reshape(Hp, Wp, -1)  # [Hp,Wp,Cdino]


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/iae_map")
    ap.add_argument("--config", default="config/fast_livo/iae.yaml")
    ap.add_argument("--out", default=None,
                    help="기본 <root>/occgrid_cache/image_features.bin")
    ap.add_argument("--dim", type=int, default=16, help="PCA 압축 차원 (=image_feature_dim)")
    ap.add_argument("--model", default="dinov2_vits14")
    ap.add_argument("--patch", type=int, default=14)
    ap.add_argument("--pca-views", type=int, default=80, help="PCA fit 에 쓸 뷰 수")
    ap.add_argument("--pca-patches", type=int, default=800, help="뷰당 PCA 샘플 patch 수")
    ap.add_argument("--llff", type=int, default=-1)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = args.out or os.path.join(args.root, "occgrid_cache", "image_features.bin")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    # train 뷰 (LUT/train_color_ 와 동일 순서)
    raw_imgs = cu.list_images(os.path.join(args.root, "images"))
    llff = args.llff if args.llff >= 0 else ivl.read_llff(args.config)
    keep = ivl.train_subset(len(raw_imgs), llff)
    train_imgs = [raw_imgs[p] for p in keep]
    V = len(train_imgs)
    print(f"[precompute] raw={len(raw_imgs)} llff={llff} -> train views V={V}")

    # 입력 해상도를 patch 배수로 맞춤
    w0, h0 = Image.open(train_imgs[0]).size
    H14 = max(args.patch, (h0 // args.patch) * args.patch)
    W14 = max(args.patch, (w0 // args.patch) * args.patch)
    Hp, Wp = H14 // args.patch, W14 // args.patch
    print(f"[precompute] img {w0}x{h0} -> dino in {W14}x{H14} -> patch {Wp}x{Hp}")

    model = load_dino(args.model, device)

    # ---- PCA fit (뷰/patch subsample) ----
    print("[precompute] PCA fit ...")
    samp = []
    idxs = np.linspace(0, V - 1, min(args.pca_views, V)).astype(int)
    for i in idxs:
        f = dino_feat(model, train_imgs[i], H14, W14, args.patch, device)
        f = f.reshape(-1, f.shape[-1])
        sel = torch.randint(0, f.shape[0], (min(args.pca_patches, f.shape[0]),),
                            device=f.device)
        samp.append(f.index_select(0, sel).float().cpu())
    X = torch.cat(samp, 0)                       # [Ns, Cdino]
    mean = X.mean(0, keepdim=True)               # [1, Cdino]
    U, S, Vt = torch.linalg.svd(X - mean, full_matrices=False)
    comps = Vt[:args.dim].contiguous()           # [dim, Cdino]
    mean = mean.to(device); comps = comps.to(device)
    print(f"[precompute] Cdino={X.shape[1]} -> dim={args.dim}, "
          f"explained {float(S[:args.dim].pow(2).sum()/S.pow(2).sum()):.3f}")

    # ---- 전체 뷰 project + 저장 ----
    with open(out, "wb") as fout:
        fout.write(struct.pack("<qqqq", V, args.dim, Hp, Wp))
        for j, p in enumerate(train_imgs):
            f = dino_feat(model, p, H14, W14, args.patch, device)  # [Hp,Wp,Cdino]
            proj = torch.matmul(f.reshape(-1, f.shape[-1]) - mean, comps.t())
            proj = proj.reshape(Hp, Wp, args.dim).float().cpu().numpy()
            fout.write(np.ascontiguousarray(proj, np.float32).tobytes())
            if (j + 1) % 100 == 0:
                print(f"  {j+1}/{V}")

    sz = os.path.getsize(out) / 1e6
    print(f"[precompute] saved -> {out}  ({sz:.1f} MB)  [V={V},C={args.dim},"
          f"{Wp}x{Hp}]")
    print("이제 config: image_feature_backbone: 2, image_feature_dim: "
          f"{args.dim} 로 학습하세요.")


if __name__ == "__main__":
    main()
