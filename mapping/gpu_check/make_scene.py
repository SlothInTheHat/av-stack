"""Render a synthetic scene (floor + two walls, noise textures) from known camera poses.
Usage: python make_scene.py OUT_DIR   ->  OUT_DIR/images/*.png, OUT_DIR/gt.json (c2w poses, intrinsics)"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

W, H, F = 640, 480, 500.0
HALF = 1.5  # plane half-extent


def texture(seed, size=1024):
    rng = np.random.default_rng(seed)
    img = sum(cv2.resize(rng.random((s, s, 3)).astype(np.float32), (size, size), interpolation=cv2.INTER_CUBIC)
              for s in (8, 16, 32, 64, 128, 256))
    img = (img - img.min()) / (img.max() - img.min())
    return (img * 255).astype(np.uint8)


# (origin, u axis, v axis): floor, back wall (y=1.5), side wall (x=-1.5)
PLANES = [((0, 0, 0), (1, 0, 0), (0, 1, 0)),
          ((0, 1.5, 1.5), (1, 0, 0), (0, 0, 1)),
          ((-1.5, 0, 1.5), (0, 1, 0), (0, 0, 1))]
TEX = [texture(i) for i in range(3)]


def look_at(pos, target, up=(0, 0, 1)):
    f = np.asarray(target, float) - pos
    f /= np.linalg.norm(f)
    right = np.cross(f, up)
    right /= np.linalg.norm(right)
    down = np.cross(f, right)
    c2w = np.eye(4)
    c2w[:3, :3] = np.stack([right, down, f], axis=1)  # OpenCV camera: x right, y down, z forward
    c2w[:3, 3] = pos
    return c2w


def render(c2w):
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    d = np.stack([(u - W / 2) / F, (v - H / 2) / F, np.ones_like(u)], -1) @ c2w[:3, :3].T
    p = c2w[:3, 3]
    img = np.full((H, W, 3), 90, np.uint8)
    best = np.full((H, W), np.inf)
    for (o, a, b), tex in zip(PLANES, TEX):
        o, a, b = map(np.asarray, (o, a, b))
        n = np.cross(a, b)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (n @ (o - p)) / (d @ n)
        q = p + t[..., None] * d
        ca, cb = (q - o) @ a, (q - o) @ b
        ok = (t > 0) & (np.abs(ca) <= HALF) & (np.abs(cb) <= HALF) & (t < best)
        mx = ((ca + HALF) / (2 * HALF) * (tex.shape[1] - 1)).astype(np.float32)
        my = ((HALF - cb if a[2] == 0 and b[2] == 1 else cb + HALF) / (2 * HALF) * (tex.shape[0] - 1)).astype(np.float32)
        samp = cv2.remap(tex, np.nan_to_num(mx), np.nan_to_num(my), cv2.INTER_LINEAR)
        img[ok], best[ok] = samp[ok], t[ok]
    return img


def main(out):
    out = Path(out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    poses, target = [], np.array([0, 0, 0.8])
    for el in (25, 42):
        for az in np.linspace(-95, 5, 14):
            e, a = np.radians(el), np.radians(az)
            pos = target + 4.5 * np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
            poses.append(look_at(pos, target))
    for i, c2w in enumerate(poses):
        cv2.imwrite(str(out / "images" / f"{i:03d}.png"), render(c2w)[:, :, ::-1])
    json.dump({"W": W, "H": H, "f": F, "c2w": [p.tolist() for p in poses]}, open(out / "gt.json", "w"))
    print(f"rendered {len(poses)} views to {out}")


if __name__ == "__main__":
    main(sys.argv[1])
