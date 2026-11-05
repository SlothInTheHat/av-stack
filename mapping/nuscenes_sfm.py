"""Reconstruct one nuScenes scene with COLMAP and compare the result with nuScenes' own camera poses.
Built up one method at a time."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pycolmap
from nuscenes.nuscenes import NuScenes
from PIL import Image
from pyquaternion import Quaternion

DATAROOT = "/home/nikhil/data/nuscenes"


def camera_frames(nusc, scene_name, channel="CAM_FRONT"):
    """Every frame one camera recorded during a scene, in time order (keyframes and sweeps).

    Returns a list of dicts: image path, timestamp (microseconds), whether it is a keyframe,
    the ego pose record (where the car was in the world) and the calibrated sensor record
    (where the camera is mounted on the car, plus its intrinsics)."""
    scene = next(s for s in nusc.scene if s["name"] == scene_name)
    first_sample = nusc.get("sample", scene["first_sample_token"])
    sd = nusc.get("sample_data", first_sample["data"][channel])
    frames = []
    while sd is not None:
        frames.append({
            "path": f"{nusc.dataroot}/{sd['filename']}",
            "timestamp": sd["timestamp"],
            "is_key_frame": sd["is_key_frame"],
            "ego_pose": nusc.get("ego_pose", sd["ego_pose_token"]),
            "calib": nusc.get("calibrated_sensor", sd["calibrated_sensor_token"]),
        })
        sd = nusc.get("sample_data", sd["next"]) if sd["next"] else None
    return frames


def camera_to_world(frame):
    """4x4 matrix that takes a point in this frame's camera coordinates to world (map) coordinates.

    Two hops: camera -> car (calibrated_sensor), then car -> world (ego_pose)."""
    def to_matrix(record):  # translation + [w, x, y, z] quaternion -> 4x4 rigid transform
        T = np.eye(4)
        T[:3, :3] = Quaternion(record["rotation"]).rotation_matrix
        T[:3, 3] = record["translation"]
        return T
    return to_matrix(frame["ego_pose"]) @ to_matrix(frame["calib"])


def select_frames(frames, spacing=1.0):
    """Keep frames roughly `spacing` metres apart along the camera's path.

    Video frames are evenly spaced in time, not distance: when the car crawls they are near-duplicates
    (little new information for COLMAP), when it is stopped they are exact duplicates (no baseline to
    triangulate from). Spacing by distance gives COLMAP a steady amount of motion between images."""
    kept = [frames[0]]
    last = camera_to_world(frames[0])[:3, 3]
    for f in frames[1:]:
        pos = camera_to_world(f)[:3, 3]
        if np.linalg.norm(pos - last) >= spacing:
            kept.append(f)
            last = pos
    return kept


def prepare_workspace(frames, out_dir):
    """Lay out a COLMAP workspace: out_dir/images/000.jpg, 001.jpg, ... plus out_dir/gt.json.

    Images are symlinks to the nuScenes files (no copy). Numbering them in time order matters:
    COLMAP's sequential matcher pairs each image with its next few neighbours by name order.
    gt.json keeps what COLMAP must not see but we need later: each image's true camera-to-world pose
    and the camera intrinsics."""
    out = Path(out_dir)
    images = out / "images"
    images.mkdir(parents=True, exist_ok=True)
    for old in images.iterdir():  # rerunning with a different selection must not leave stale links
        old.unlink()
    gt = {"K": frames[0]["calib"]["camera_intrinsic"], "images": {}}
    gt["width"], gt["height"] = Image.open(frames[0]["path"]).size
    for i, f in enumerate(frames):
        name = f"{i:03d}.jpg"
        (images / name).symlink_to(f["path"])
        gt["images"][name] = {"source": Path(f["path"]).name, "timestamp": f["timestamp"],
                              "c2w": camera_to_world(f).tolist()}
    (out / "gt.json").write_text(json.dumps(gt, indent=1))
    return out


def run_sfm(workspace, overlap=10):
    """Structure-from-motion on workspace/images with COLMAP. Returns the largest reconstruction.

    Three stages, each writing into workspace/database.db (a SQLite file COLMAP uses as its scratchpad):
      1. extract: find distinctive points (SIFT keypoints) in every image, each with a 128-number descriptor
      2. match:   pair up keypoints between images whose descriptors look alike, then keep only the pairs
                  that agree with a single consistent camera motion (geometric verification)
      3. map:     pick a good starting pair, recover their relative pose and triangulate points, then add
                  images one by one, re-optimising everything as it goes (bundle adjustment)"""
    ws = Path(workspace)
    gt = json.loads((ws / "gt.json").read_text())
    db, sparse = ws / "database.db", ws / "sparse"
    db.unlink(missing_ok=True)  # stale features from an earlier run would be mixed in
    sparse.mkdir(exist_ok=True)

    # 1. extract. One shared PINHOLE camera with the calibrated intrinsics, so COLMAP does not guess them.
    K = gt["K"]
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "PINHOLE"  # fx, fy, cx, cy; nuScenes images are already undistorted
    reader.camera_params = f"{K[0][0]},{K[1][1]},{K[0][2]},{K[1][2]}"
    pycolmap.extract_features(db, ws / "images", camera_mode=pycolmap.CameraMode.SINGLE, reader_options=reader)

    # 2. match. Video: each image only overlaps its neighbours, so don't compare every pair. COLMAP's
    # default quadratic_overlap=True pairs each image with those 1, 2, 4, 8, ... positions away (in both
    # directions, up to `overlap` steps of doubling), mixing short baselines with a few long ones.
    pairing = pycolmap.SequentialPairingOptions()
    pairing.overlap = overlap
    pycolmap.match_sequential(db, pairing_options=pairing)

    # 3. map. Intrinsics are known, so bundle adjustment may only move cameras and points, not the lens.
    opts = pycolmap.IncrementalPipelineOptions()
    opts.ba_refine_focal_length = False
    opts.ba_refine_principal_point = False
    opts.ba_refine_extra_params = False
    models = pycolmap.incremental_mapping(db, ws / "images", sparse, options=opts)
    if not models:
        raise RuntimeError("COLMAP could not initialise a reconstruction")
    return max(models.values(), key=lambda r: r.num_reg_images())


def align_similarity(src, dst):
    """Best scale s, rotation R, translation t with  dst ~= s * R @ src + t  (Umeyama, 1991).

    src, dst: (N, 3) matching points. Solves least squares in closed form: centre both sets, take the
    SVD of their cross-covariance for the rotation, then the scale from the variances. 7 unknowns
    (1 scale + 3 rotation + 3 translation), exactly what monocular SfM cannot know on its own."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    X, Y = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(Y.T @ X / len(src))
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U) * np.linalg.det(Vt))])  # forbid a mirror image
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / (X ** 2).sum(1).mean()
    t = mu_d - s * R @ mu_s
    return s, R, t


def align_with_orientations(est_c, est_R, true_c, true_R):
    """Like align_similarity, but the rotation comes from the camera orientations, not the centres.

    Centres along a near-straight road cannot pin down the roll around the road axis; orientations
    can. Each camera gives its own estimate of the frame rotation, R_i = true_R_i @ est_R_i^T (camera
    -> COLMAP world -> back to camera -> true world). Average them by summing and projecting back onto
    a rotation with an SVD (the chordal mean), then fit scale and translation to the centres with that
    rotation held fixed. est_R / true_R: camera->world rotations; est_c / true_c: camera centres."""
    M = sum(Rt @ Re.T for Re, Rt in zip(est_R, true_R))
    U, _, Vt = np.linalg.svd(M)
    R = U @ np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))]) @ Vt  # nearest true rotation
    est_c, true_c = np.asarray(est_c, float), np.asarray(true_c, float)
    mu_e, mu_t = est_c.mean(0), true_c.mean(0)
    X, Y = (est_c - mu_e) @ R.T, true_c - mu_t  # rotated COLMAP centres vs true centres, both centred
    s = (X * Y).sum() / (X * X).sum()  # least-squares scale along the fixed rotation
    t = mu_t - s * R @ mu_e
    return s, R, t


def evaluate(rec, workspace):
    """Compare COLMAP's camera poses with nuScenes ground truth (gt.json) after a similarity alignment.

    Aligns with align_with_orientations (centre-only alignment left a 21 deg roll on a straight road),
    then reports per image:
      position error    distance between aligned COLMAP centre and true centre (metres), split into
                        horizontal and vertical because nuScenes poses are flat (z = 0)
      orientation error angle of the rotation between aligned COLMAP and true camera orientation (degrees);
                        catches a model spun around the road axis, which position error alone can miss
    Returns a dict of summary numbers plus the per-image arrays."""
    gt = json.loads((Path(workspace) / "gt.json").read_text())
    names, est_c, est_R, true_c, true_R = [], [], [], [], []
    for im in sorted(rec.images.values(), key=lambda i: i.name):
        pose = im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world
        m = pose.matrix()  # 3x4 world -> camera [R | t]
        c2w = np.array(gt["images"][im.name]["c2w"])
        names.append(im.name)
        est_R.append(m[:, :3].T)  # camera -> world rotation in COLMAP's frame
        est_c.append(-m[:, :3].T @ m[:, 3])  # camera centre C = -R^T t
        true_R.append(c2w[:3, :3])
        true_c.append(c2w[:3, 3])
    est_c, true_c = np.array(est_c), np.array(true_c)

    s, R, t = align_with_orientations(est_c, est_R, true_c, true_R)
    d = (s * est_c @ R.T + t) - true_c
    pos_err = np.linalg.norm(d, axis=1)
    # orientation: bring COLMAP's camera->world rotation into the true frame (R @ R_est), compare with truth
    rot_err = np.array([np.degrees(np.arccos(np.clip((np.trace(Rt.T @ (R @ Re)) - 1) / 2, -1, 1)))
                        for Re, Rt in zip(est_R, true_R)])
    path = lambda c: np.linalg.norm(np.diff(c, axis=0), axis=1).sum()  # noqa: E731
    return {
        "images": len(names), "scale_m_per_unit": s,
        "pos_err_mean": pos_err.mean(), "pos_err_median": np.median(pos_err), "pos_err_max": pos_err.max(),
        "horiz_err_mean": np.linalg.norm(d[:, :2], axis=1).mean(), "vert_err_mean": np.abs(d[:, 2]).mean(),
        "rot_err_mean": rot_err.mean(), "rot_err_max": rot_err.max(),
        "path_len_true": path(true_c), "path_len_est": s * path(est_c),
        "names": names, "pos_err": pos_err, "rot_err": rot_err, "residual": d,
        "R": R, "t": t,  # with scale_m_per_unit: COLMAP frame -> map frame, x_map = s * R @ x + t
    }


def export_ply(rec, r, workspace):
    """Write WORKSPACE/aligned.ply: the 3D points in nuScenes map coordinates (metres), in their image
    colours, plus every camera centre as a bright green point so the drive path is visible.
    Also writes WORKSPACE/transform.json with the scale, rotation and translation used.

    PLY is a plain format: a text header listing the fields, then one row per point. Point cloud
    viewers (CloudCompare, MeshLab, Blender) all open it."""
    ws = Path(workspace)
    s, R, t = r["scale_m_per_unit"], r["R"], r["t"]
    pts = np.array([p.xyz for p in rec.points3D.values()])
    cols = np.array([p.color for p in rec.points3D.values()], dtype=np.uint8)
    cams = []
    for im in rec.images.values():
        m = (im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world).matrix()
        cams.append(-m[:, :3].T @ m[:, 3])
    xyz = s * np.vstack([pts, cams]) @ R.T + t
    rgb = np.vstack([cols, np.tile([0, 255, 0], (len(cams), 1))]).astype(np.uint8)

    header = (f"ply\nformat ascii 1.0\nelement vertex {len(xyz)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    rows = "\n".join(f"{x:.3f} {y:.3f} {z:.3f} {r_} {g} {b}" for (x, y, z), (r_, g, b) in zip(xyz, rgb))
    (ws / "aligned.ply").write_text(header + rows + "\n")
    (ws / "transform.json").write_text(json.dumps({"scale": float(s), "R": R.tolist(), "t": t.tolist(),
                                                   "note": "x_map = scale * R @ x_colmap + t"}, indent=1))
    return ws / "aligned.ply", len(pts), len(cams)


def main():
    """Run the whole pipeline for one scene: frames -> selection -> workspace -> COLMAP -> evaluation.

    Example: python nuscenes_sfm.py scene-0103 --spacing 1.0
    --reuse skips COLMAP and evaluates the model already in WORKSPACE/sparse/0 (COLMAP is the slow part).
    Writes WORKSPACE/eval.json with the summary numbers."""
    ap = argparse.ArgumentParser(description=main.__doc__.splitlines()[0])
    ap.add_argument("scene", help="nuScenes scene name, e.g. scene-0103")
    ap.add_argument("--channel", default="CAM_FRONT")
    ap.add_argument("--spacing", type=float, default=1.0, help="metres between selected frames")
    ap.add_argument("--workspace", default=None, help="default: ~/scenes/nusc-<scene number>")
    ap.add_argument("--reuse", action="store_true", help="skip COLMAP, evaluate the existing model")
    ap.add_argument("--version", default="v1.0-mini")
    a = ap.parse_args()
    ws = Path(a.workspace or Path.home() / "scenes" / f"nusc-{a.scene.split('-')[-1]}")

    if a.reuse:
        rec = pycolmap.Reconstruction(str(ws / "sparse" / "0"))
        print(f"reusing {ws / 'sparse' / '0'}")
    else:
        nusc = NuScenes(version=a.version, dataroot=DATAROOT, verbose=False)
        frames = camera_frames(nusc, a.scene, a.channel)
        kept = select_frames(frames, a.spacing)
        prepare_workspace(kept, ws)
        print(f"{a.scene} {a.channel}: {len(kept)}/{len(frames)} frames at {a.spacing} m spacing -> {ws}")
        t0 = time.time()
        rec = run_sfm(ws)
        print(f"COLMAP: {time.time() - t0:.0f} s, {rec.num_points3D()} points, "
              f"reprojection error {rec.compute_mean_reprojection_error():.2f} px")

    r = evaluate(rec, ws)
    summary = {k: (float(v) if isinstance(v, (float, np.floating)) else v)
               for k, v in r.items() if not isinstance(v, (list, np.ndarray))}
    (ws / "eval.json").write_text(json.dumps(summary, indent=1))
    ply, n_pts, n_cams = export_ply(rec, r, ws)
    print(f"wrote {ply} ({n_pts} points + {n_cams} camera centres in green)")
    print(f"registered {r['images']} images | scale {r['scale_m_per_unit']:.2f} m/unit | "
          f"path {r['path_len_est']:.1f} m (true {r['path_len_true']:.1f} m)")
    print(f"position error mean {r['pos_err_mean'] * 100:.1f} cm (horizontal {r['horiz_err_mean'] * 100:.1f}, "
          f"vertical {r['vert_err_mean'] * 100:.1f}), max {r['pos_err_max'] * 100:.1f} cm")
    print(f"orientation error mean {r['rot_err_mean']:.2f} deg, max {r['rot_err_max']:.2f} deg")


if __name__ == "__main__":
    main()
