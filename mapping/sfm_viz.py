"""Pictures of what each COLMAP stage did, read from a workspace made by nuscenes_sfm.py.
One function per visual."""
import shutil
import sqlite3
from pathlib import Path

from nuscenes_sfm import align_similarity

import cv2
import matplotlib
import numpy as np
import pycolmap

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def draw_keypoints(workspace, image_name, out_png, crop=(1000, 380, 1400, 605)):
    """Stage 1 (feature extraction): every SIFT keypoint COLMAP found in one image.

    Each keypoint is drawn as a circle whose radius is its detected scale, with a tick for its
    orientation. Left: the whole image. Right: a zoomed crop (x0, y0, x1, y1) where the circles
    and ticks are readable."""
    ws = Path(workspace)
    db = sqlite3.connect(ws / "database.db")
    image_id = db.execute("SELECT image_id FROM images WHERE name = ?", (image_name,)).fetchone()[0]
    rows, cols, blob = db.execute("SELECT rows, cols, data FROM keypoints WHERE image_id = ?", (image_id,)).fetchone()
    kp = np.frombuffer(blob, np.float32).reshape(rows, cols)
    # COLMAP stores each keypoint as x, y and a 2x2 affine shape [a11 a12; a21 a22]:
    # scale = length of the first column, orientation = angle of the first column
    x, y, a11, a21 = kp[:, 0], kp[:, 1], kp[:, 2], kp[:, 4]
    scale, angle = np.hypot(a11, a21), np.arctan2(a21, a11)

    img = cv2.imread(str(ws / "images" / image_name))
    full = img.copy()
    for xi, yi in zip(x, y):
        cv2.circle(full, (int(xi), int(yi)), 2, (0, 255, 255), -1, cv2.LINE_AA)

    x0, y0, x1, y1 = crop
    zoom = 3
    detail = cv2.resize(img[y0:y1, x0:x1], None, fx=zoom, fy=zoom, interpolation=cv2.INTER_CUBIC)
    rng = np.random.default_rng(0)
    for xi, yi, si, ai in zip(x, y, scale, angle):
        if not (x0 <= xi < x1 and y0 <= yi < y1):
            continue
        c = tuple(int(v) for v in rng.integers(80, 256, 3))
        cx, cy, r = (xi - x0) * zoom, (yi - y0) * zoom, max(si * zoom, 3)
        cv2.circle(detail, (int(cx), int(cy)), int(r), c, 1, cv2.LINE_AA)
        cv2.line(detail, (int(cx), int(cy)), (int(cx + r * np.cos(ai)), int(cy + r * np.sin(ai))), c, 1, cv2.LINE_AA)
    cv2.rectangle(full, (x0, y0), (x1, y1), (255, 255, 255), 3)

    # stack side by side at equal height, with captions
    h = 900
    left = cv2.resize(full, (int(full.shape[1] * h / full.shape[0]), h))
    right = cv2.resize(detail, (int(detail.shape[1] * h / detail.shape[0]), h))
    canvas = np.hstack([left, np.full((h, 12, 3), 255, np.uint8), right])
    for text, xpos in ((f"{image_name}: {rows} SIFT keypoints", 20), (f"zoom x{zoom}: circle = scale, tick = orientation", left.shape[1] + 32)):
        cv2.putText(canvas, text, (xpos, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 6, cv2.LINE_AA)
        cv2.putText(canvas, text, (xpos, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 2, cv2.LINE_AA)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), canvas)
    return rows


def _pair_matches(db, table, id1, id2):
    """Matched keypoint indices for an image pair from `matches` (raw) or `two_view_geometries` (verified).
    COLMAP stores each pair once, smaller id first, under pair_id = id1 * 2147483647 + id2."""
    swap = id1 > id2
    a, b = (id2, id1) if swap else (id1, id2)
    row = db.execute(f"SELECT rows, data FROM {table} WHERE pair_id = ?", (a * 2147483647 + b,)).fetchone()
    if row is None or row[0] == 0:
        return np.zeros((0, 2), int)
    m = np.frombuffer(row[1], np.uint32).reshape(-1, 2).astype(int)
    return m[:, ::-1] if swap else m


def draw_matches(workspace, name1, name2, out_png, max_green=250):
    """Stage 2 (matching): two images stacked, lines joining matched keypoints.

    Red = descriptor matches that geometric verification (RANSAC) rejected.
    Green = inliers kept, a random sample of at most `max_green` so the picture stays readable."""
    ws = Path(workspace)
    db = sqlite3.connect(ws / "database.db")
    ids = {n: db.execute("SELECT image_id FROM images WHERE name = ?", (n,)).fetchone()[0] for n in (name1, name2)}
    kps = {}
    for n, i in ids.items():
        rows, cols, blob = db.execute("SELECT rows, cols, data FROM keypoints WHERE image_id = ?", (i,)).fetchone()
        kps[n] = np.frombuffer(blob, np.float32).reshape(rows, cols)[:, :2]
    raw = _pair_matches(db, "matches", ids[name1], ids[name2])
    inl = _pair_matches(db, "two_view_geometries", ids[name1], ids[name2])
    inlier_set = {tuple(m) for m in inl}
    rejected = np.array([m for m in raw if tuple(m) not in inlier_set]).reshape(-1, 2)

    top, bottom = (cv2.imread(str(ws / "images" / n)) for n in (name1, name2))
    canvas = np.vstack([top, np.full((10, top.shape[1], 3), 255, np.uint8), bottom])
    dy = top.shape[0] + 10

    def line(m, color, width):
        p, q = kps[name1][m[0]], kps[name2][m[1]]
        cv2.line(canvas, (int(p[0]), int(p[1])), (int(q[0]), int(q[1]) + dy), color, width, cv2.LINE_AA)

    for m in rejected:
        line(m, (0, 0, 255), 1)
    rng = np.random.default_rng(0)
    for m in inl[rng.permutation(len(inl))[:max_green]]:
        line(m, (0, 220, 0), 1)
    text = f"{name1} -> {name2}: {len(raw)} descriptor matches, {len(inl)} kept (green, {min(max_green, len(inl))} drawn), {len(rejected)} rejected (red)"
    cv2.putText(canvas, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 6, cv2.LINE_AA)
    cv2.putText(canvas, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(out_png), canvas)
    return len(raw), len(inl)


def _camera_center(image):
    """Camera position in the reconstruction's frame: C = -R^T t, from the world->camera pose (R, t)."""
    pose = image.cam_from_world() if callable(image.cam_from_world) else image.cam_from_world
    m = pose.matrix()
    return -m[:, :3].T @ m[:, 3]


def draw_track(workspace, out_png, point_id=None, max_patches=12):
    """Stage 3 (mapping): the life of one 3D point.

    Top: the small patch around this point's keypoint in every image that saw it (its "track").
    Bottom: top-down view of all cameras, the observing ones highlighted, with their viewing rays
    meeting at the triangulated point. If point_id is None, picks a long track with a wide
    triangulation angle (the rays from its first and last observing camera disagree the most)."""
    ws = Path(workspace)
    rec = pycolmap.Reconstruction(str(ws / "sparse" / "0"))
    centers = {iid: _camera_center(im) for iid, im in rec.images.items()}

    def spread(pid):  # angle between the rays from the two most separated observing cameras
        p = rec.points3D[pid]
        ids = sorted(e.image_id for e in p.track.elements)
        a, b = p.xyz - centers[ids[0]], p.xyz - centers[ids[-1]]
        return np.degrees(np.arccos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1)))
    if point_id is None:
        candidates = [pid for pid, p in rec.points3D.items() if p.track.length() >= 10 and p.error < 1.0]
        point_id = max(candidates, key=spread)
    pt = rec.points3D[point_id]
    obs = sorted(((rec.images[e.image_id].name, e.image_id, rec.images[e.image_id].points2D[e.point2D_idx].xy)
                  for e in pt.track.elements))

    # "up" = opposite of the cameras' average y axis (camera y points down); "forward" = direction of travel
    R_rows = []
    for im in rec.images.values():
        pose = im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world
        R_rows.append(pose.matrix()[:, :3])
    up = -np.mean([R[1] for R in R_rows], axis=0)
    up /= np.linalg.norm(up)
    path = np.array([centers[i] for i in sorted(centers, key=lambda i: rec.images[i].name)])
    fwd = path[-1] - path[0]
    fwd -= (fwd @ up) * up
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, up)
    to2d = lambda X: np.stack([np.asarray(X) @ fwd, np.asarray(X) @ -right], -1)  # noqa: E731  road runs left->right

    n = min(len(obs), max_patches)
    pick = np.linspace(0, len(obs) - 1, n).round().astype(int)
    fig = plt.figure(figsize=(16, 7))
    gs = fig.add_gridspec(2, n, height_ratios=[1, 1.7])
    for k, j in enumerate(pick):
        name, _, xy = obs[j]
        img = cv2.imread(str(ws / "images" / name))[:, :, ::-1]
        r = 40
        x, y = int(round(xy[0])), int(round(xy[1]))
        patch = np.zeros((2 * r, 2 * r, 3), np.uint8)
        y0, y1, x0, x1 = max(y - r, 0), min(y + r, img.shape[0]), max(x - r, 0), min(x + r, img.shape[1])
        patch[y0 - (y - r):y1 - (y - r), x0 - (x - r):x1 - (x - r)] = img[y0:y1, x0:x1]
        ax = fig.add_subplot(gs[0, k])
        ax.imshow(patch)
        ax.plot([r - 0.5], [r - 0.5], "r+", ms=14, mew=1.5)
        ax.set_title(name, fontsize=9)
        ax.axis("off")

    ax = fig.add_subplot(gs[1, :])
    allc = to2d(path)
    ax.plot(allc[:, 0], allc[:, 1], ".", color="0.75", ms=5, label="all 89 camera positions")
    P = to2d(pt.xyz)
    for name, iid, _ in obs:
        c = to2d(centers[iid])
        ax.plot([c[0], P[0]], [c[1], P[1]], "-", color="tab:green", lw=0.8, alpha=0.7)
        ax.plot(*c, "o", color="tab:green", ms=5)
    ax.plot(*P, "*", color="red", ms=18, label=f"triangulated point {point_id}")
    ax.plot([], [], "-o", color="tab:green", ms=5, label=f"{len(obs)} cameras that saw it, and their rays")
    # zoom to the stretch of road where the point was visible; equal axes keep the ray angles true
    seen = np.array([to2d(centers[iid]) for _, iid, _ in obs] + [P])
    pad = 0.08 * np.ptp(seen[:, 0])
    ax.set_xlim(seen[:, 0].min() - pad, seen[:, 0].max() + pad)
    mid = seen[:, 1].mean()
    ax.set_ylim(mid - 0.12 * np.ptp(seen[:, 0]), mid + 0.12 * np.ptp(seen[:, 0]))
    ax.set_aspect("equal")
    ax.set_xlabel("along the road, direction of travel -> (reconstruction units, scale unknown)")
    ax.set_ylabel("sideways (left +)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.suptitle(f"One 3D point's track: seen in {len(obs)} images ({n} shown), reprojection error {pt.error:.2f} px, "
                 f"rays from first and last camera {spread(point_id):.1f} deg apart", fontsize=13)
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return point_id, len(obs)


def animate_mapping(workspace, out_gif, fps=6):
    """Stage 3 (mapping) as a GIF: the reconstruction growing one registered image at a time.

    Re-runs only the mapping stage (features and matches are reused from database.db) with COLMAP
    writing a snapshot after every registered image, then draws each snapshot top-down. COLMAP may
    rescale/shift the model between snapshots, so each is aligned to the final model first
    (align_similarity on the shared camera centres) to keep the view steady."""
    ws = Path(workspace)
    tmp = ws / "anim"
    shutil.rmtree(tmp, ignore_errors=True)
    (tmp / "snapshots").mkdir(parents=True)
    (tmp / "sparse").mkdir()
    opts = pycolmap.IncrementalPipelineOptions()
    opts.ba_refine_focal_length = opts.ba_refine_principal_point = opts.ba_refine_extra_params = False
    opts.snapshot_path = str(tmp / "snapshots")
    opts.snapshot_frames_freq = 1
    pycolmap.incremental_mapping(ws / "database.db", ws / "images", tmp / "sparse", options=opts)
    snaps = [pycolmap.Reconstruction(str(d)) for d in sorted((tmp / "snapshots").iterdir())]
    final = pycolmap.Reconstruction(str(ws / "sparse" / "0"))

    # top-down basis from the final model: up = minus the cameras' mean y axis, forward = start->end of path
    def poses(rec):
        return {im.name: (im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world).matrix()
                for im in rec.images.values() if im.has_pose}
    fin = poses(final)
    centre = {n: -m[:, :3].T @ m[:, 3] for n, m in fin.items()}
    up = -np.mean([m[1, :3] for m in fin.values()], axis=0)
    up /= np.linalg.norm(up)
    names = sorted(centre)
    fwd = centre[names[-1]] - centre[names[0]]
    fwd -= (fwd @ up) * up
    fwd /= np.linalg.norm(fwd)
    left = np.cross(up, fwd)
    to2d = lambda X: np.stack([X @ fwd, X @ left], -1)  # noqa: E731
    path2d = to2d(np.array([centre[n] for n in names]))
    fpts = to2d(np.array([p.xyz for p in final.points3D.values()]))
    xlo, xhi = np.percentile(fpts[:, 0], [1, 99])
    ylo, yhi = np.percentile(fpts[:, 1], [2, 98])
    xlo, xhi = min(xlo, path2d[:, 0].min()) - 0.3, max(xhi, path2d[:, 0].max()) + 0.3

    frames, prev = [], set()
    for k, rec in enumerate(snaps):
        P = poses(rec)
        shared = [n for n in P if n in centre]
        c_snap = np.array([-P[n][:, :3].T @ P[n][:, 3] for n in shared])
        s, R, t = align_similarity(c_snap, np.array([centre[n] for n in shared])) if len(shared) >= 3 else (1, np.eye(3), 0)
        pts = np.array([p.xyz for p in rec.points3D.values()]).reshape(-1, 3)
        cols = np.array([p.color for p in rec.points3D.values()]).reshape(-1, 3) / 255.0
        pts2d = to2d(s * pts @ R.T + t)
        cams2d = to2d(s * c_snap @ R.T + t)
        new = [i for i, n in enumerate(shared) if n not in prev]
        prev = set(shared)

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.scatter(pts2d[:, 0], pts2d[:, 1], s=1.5, c=cols, linewidths=0)
        ax.plot(path2d[:, 0], path2d[:, 1], "-", color="0.85", lw=1, zorder=0)
        ax.plot(cams2d[:, 0], cams2d[:, 1], "o", color="tab:green", ms=4)
        if k > 0:
            ax.plot(cams2d[new, 0], cams2d[new, 1], "o", color="red", ms=9, label="just added")
        else:
            # COLMAP's first snapshot comes after the initial pair plus the next registered image
            ax.plot(cams2d[:, 0], cams2d[:, 1], "o", color="red", ms=9, label=f"first {len(shared)} cameras")
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(ylo, yhi)
        ax.set_aspect("equal")
        ax.set_facecolor("0.12")
        ax.legend(loc="upper left")
        ax.set_title(f"step {k + 1}: {len(shared)} cameras registered, {len(pts)} 3D points "
                     f"(top-down, direction of travel ->)")
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
        plt.close(fig)

    import imageio.v2 as imageio
    imageio.mimsave(out_gif, frames + [frames[-1]] * fps * 2, duration=1 / fps, loop=0)  # hold the last frame
    shutil.rmtree(tmp)
    return len(frames)
