# Phase 2 tooling check (2026-10-20)

Goal: confirm COLMAP and 3D Gaussian Splatting run on the RTX 5070 Ti (Blackwell, sm_120) before capturing real data.

## Environment (WSL2 Ubuntu 24.04, `~/venvs/recon`)
- System nvcc was 12.0 (too old for sm_120). Installed the CUDA 12.8.1 toolkit without sudo:
  `sh cuda_12.8.1_570.124.06_linux.run --silent --toolkit --toolkitpath=$HOME/cuda-12.8 --no-man-page --override --tmpdir=$HOME/tmp`
- `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128`
- `pip install pycolmap-cuda12 numpy pillow opencv-python-headless trimesh ninja gsplat packaging setuptools`
- gsplat compiles its CUDA kernels on first use (~4 min): `export CUDA_HOME=$HOME/cuda-12.8 PATH=$CUDA_HOME/bin:$PATH TORCH_CUDA_ARCH_LIST=12.0`

## Check (mapping/gpu_check/)
    python make_scene.py ~/scenes/synth     # 28 rendered views of 3 textured planes, known poses
    python run_colmap.py ~/scenes/synth     # SfM + comparison with ground truth
    python train_gsplat.py ~/scenes/synth   # 800-step 3DGS fit

## Results
- COLMAP (pycolmap 4.2.1, CUDA build): 28/28 images registered, 10,657 points, 0.19 px reprojection error,
  camera positions within 0.13 cm mean / 0.31 cm max of ground truth after similarity alignment. 28 s.
- gsplat: kernels built for sm_120; PSNR 13.4 -> 25.9 dB in 800 steps (5.4 s, 0.19 GB VRAM; no densification, toy scene).

## Notes
- The C: drive filled up during setup (0 GB free), which made WSL throw I/O errors. Freed ~49 GB. Real captures plus COLMAP/3DGS output need tens of GB: keep an eye on it.
- Not yet verified: whether COLMAP's feature extraction actually ran on the GPU (28 s for 28 images suggests it may not have).

## OSM -> drivable CARLA map (2026-10-21)
    python mapping/osm_to_carla.py seaport_test 42.3440 -71.0450 42.3500 -71.0370      # Overpass download + Osm2Odr
    PYTHONPATH=control python sim/run_lap.py --xodr mapping/maps/seaport_test.xodr --camera chase
- Boston Seaport test bbox: 687 OSM ways -> 537 OpenDRIVE roads, 101 junctions. Pure pursuit lap: 0.07 m mean / 0.49 m max cross-track.
- Known issues: the converter warns about junctions it cannot smooth; `geoReference` in the .xodr fails to parse in CARLA
  ("cannot parse georeference", uses defaults) so lat/lon <-> world mapping must be handled by us (needed to align with nuScenes/COLMAP).
- Not yet looked at visually; this bbox was a pipeline test, not chosen to match nuScenes.
