"""SfM with pycolmap on the synthetic scene, then compare camera centres to ground truth.
Usage: python run_colmap.py SCENE_DIR  ->  SCENE_DIR/colmap.npz (poses, points for the 3DGS step)"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pycolmap


def pose_w2c(im):
    m = im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world
    return np.vstack([m.matrix(), [0, 0, 0, 1]])


def umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s, d = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(d.T @ s / len(src))
    D = np.diag([1, 1, np.sign(np.linalg.det(U) * np.linalg.det(Vt))])
    R = U @ D @ Vt
    scale = np.trace(np.diag(S) @ D) / s.var(0).sum()
    return scale, R, mu_d - scale * R @ mu_s


def main(scene):
    scene = Path(scene)
    db, sparse = scene / "colmap.db", scene / "sparse"
    db.unlink(missing_ok=True)
    sparse.mkdir(exist_ok=True)
    print("pycolmap", pycolmap.__version__, "| CUDA build:", pycolmap.has_cuda)
    t = time.time()
    pycolmap.extract_features(db, scene / "images")
    pycolmap.match_exhaustive(db)
    maps = pycolmap.incremental_mapping(db, scene / "images", sparse)
    print(f"SfM took {time.time() - t:.1f}s, {len(maps)} model(s)")
    rec = max(maps.values(), key=lambda r: r.num_reg_images())
    gt = json.load(open(scene / "gt.json"))
    names = sorted(im.name for im in rec.images.values())
    est, true, w2c = [], [], []
    for im in sorted(rec.images.values(), key=lambda i: i.name):
        w = pose_w2c(im)
        est.append(-w[:3, :3].T @ w[:3, 3])
        true.append(np.array(gt["c2w"][int(im.name[:3])])[:3, 3])
        w2c.append(w)
    est, true = np.array(est), np.array(true)
    s, R, tr = umeyama(est, true)
    err = np.linalg.norm((s * est @ R.T + tr) - true, axis=1)
    print(f"registered {rec.num_reg_images()}/{len(gt['c2w'])} images, {rec.num_points3D()} 3D points, "
          f"mean reproj err {rec.compute_mean_reprojection_error():.2f}px")
    print(f"camera position error after similarity alignment: mean {err.mean()*100:.2f} cm, max {err.max()*100:.2f} cm "
          f"(cameras ~4.5 m from scene)")
    pts = np.array([p.xyz for p in rec.points3D.values()])
    cols = np.array([p.color for p in rec.points3D.values()]) / 255.0
    cam = next(iter(rec.cameras.values()))
    np.savez(scene / "colmap.npz", w2c=np.array(w2c), names=names, pts=pts, cols=cols,
             K=cam.calibration_matrix(), W=cam.width, H=cam.height)


if __name__ == "__main__":
    main(sys.argv[1])
