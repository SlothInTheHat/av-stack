"""Make a front-camera mp4 (all ~12 Hz frames: keyframes + sweeps) for each requested nuScenes scene.
Usage: python nuscenes_preview.py OUT_DIR scene-0103 scene-0553 ..."""
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg
from nuscenes.nuscenes import NuScenes

ROOT = "/home/nikhil/data/nuscenes"
nusc = NuScenes(version="v1.0-mini", dataroot=ROOT, verbose=False)
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)

for name in sys.argv[2:]:
    scene = next(s for s in nusc.scene if s["name"] == name)
    sd = nusc.get("sample_data", nusc.get("sample", scene["first_sample_token"])["data"]["CAM_FRONT"])
    files = []
    while True:  # walk the camera's frame chain, keyframes and sweeps alike
        files.append(f"{ROOT}/{sd['filename']}")
        if not sd["next"]:
            break
        sd = nusc.get("sample_data", sd["next"])
    lst = out / f"{name}.txt"
    lst.write_text("".join(f"file '{f}'\nduration 0.0833\n" for f in files))
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
                    "-vf", "scale=960:-2", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out / f"{name}.mp4")], check=True)
    lst.unlink()
    print(name, len(files), "frames")
