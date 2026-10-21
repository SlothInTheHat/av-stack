"""Minimal 3DGS fit with gsplat on the COLMAP output (no densification): proves the CUDA kernels run on this GPU.
Usage: python train_gsplat.py SCENE_DIR [steps]"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from gsplat import rasterization

scene, steps = Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 800
d = np.load(scene / "colmap.npz")
dev = "cuda"
imgs = torch.tensor(np.stack([cv2.imread(str(scene / "images" / n))[:, :, ::-1] for n in d["names"]]),
                    dtype=torch.float32, device=dev) / 255.0
viewmats = torch.tensor(d["w2c"], dtype=torch.float32, device=dev)
K = torch.tensor(d["K"], dtype=torch.float32, device=dev)[None].repeat(len(imgs), 1, 1)
W, H = int(d["W"]), int(d["H"])

pts = torch.tensor(d["pts"], dtype=torch.float32, device=dev)
n = len(pts)
spacing = torch.cdist(pts[:2000], pts).kthvalue(4, dim=1).values.mean().item()  # typical neighbour distance
means = torch.nn.Parameter(pts.clone())
log_scales = torch.nn.Parameter(torch.full((n, 3), np.log(spacing), device=dev))
quats = torch.nn.Parameter(torch.nn.functional.normalize(torch.randn(n, 4, device=dev), dim=-1))
opac = torch.nn.Parameter(torch.full((n,), 0.0, device=dev))  # logit
cols = torch.nn.Parameter(torch.logit(torch.tensor(d["cols"], dtype=torch.float32, device=dev).clamp(0.02, 0.98)))
opt = torch.optim.Adam([{"params": [means], "lr": 2e-3}, {"params": [log_scales], "lr": 5e-3},
                        {"params": [quats], "lr": 2e-3}, {"params": [opac], "lr": 5e-2}, {"params": [cols], "lr": 2e-2}])


def render(idx):
    out, _, _ = rasterization(means, torch.nn.functional.normalize(quats, dim=-1), log_scales.exp(),
                              torch.sigmoid(opac), torch.sigmoid(cols), viewmats[idx:idx + 1], K[idx:idx + 1], W, H)
    return out[0]


def psnr():
    with torch.no_grad():
        mse = torch.stack([((render(i) - imgs[i]) ** 2).mean() for i in range(len(imgs))]).mean()
    return -10 * torch.log10(mse).item()


print(f"{n} gaussians, init PSNR {psnr():.2f} dB")
t = time.time()
for s in range(steps):
    i = np.random.randint(len(imgs))
    loss = ((render(i) - imgs[i]) ** 2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    if (s + 1) % 200 == 0:
        print(f"step {s + 1}: PSNR {psnr():.2f} dB")
torch.cuda.synchronize()
print(f"{steps} steps in {time.time() - t:.1f}s, peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")
