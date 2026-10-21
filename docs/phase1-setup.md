# Phase 1 setup (WSL2 Ubuntu 24.04)

Checklist:
- [ ] Windows NVIDIA driver is recent (617.14 detected; Blackwell needs CUDA 12.8+ support)
- [ ] `nvidia-smi` works inside WSL
- [ ] PyTorch (cu128+) sees the GPU: `python -c "import torch;print(torch.cuda.is_available())"`
- [ ] ROS 2 (Jazzy for Ubuntu 24.04) installed
- [ ] CARLA installed + CARLA ROS bridge
- [ ] Pure pursuit + PID drives a lap of Town01

Known risk: CARLA's prebuilt Linux packages target Ubuntu 20.04/22.04 and older CUDA/Vulkan stacks; WSL2 GUI/Vulkan adds friction. Alternatives: run CARLA natively on Windows and the ROS side in WSL, or dual boot Ubuntu 22.04.

## Log
(record failures here)

## Status (2026-10-20)
- [x] nvidia-smi in WSL; PyTorch 2.11.0+cu128 sees the 5070 Ti (sm_120)
- [x] venv `~/venvs/av` with carla==0.9.16 client, numpy, matplotlib
- [x] control/ (PID, pure pursuit) and sim/run_lap.py written; controller smoke-tested, not yet run against CARLA
- [x] CARLA 0.9.16 Windows in C:\CARLA_0.9.16 (zip had no top folder and no CarlaUE4.exe launcher; run `CarlaUE4\Binaries\Win64\CarlaUE4-Win64-Shipping.exe CarlaUE4 -quality-level=Low`)
- [x] WSL connects to Windows CARLA server on port 2000; Town01 lap: mean cross-track 0.08 m, max 1.01 m (1500 steps)
- [ ] ROS 2 Jazzy in WSL (needs sudo apt) + carla-ros-bridge

## Run
    source ~/venvs/av/bin/activate
    PYTHONPATH=control python sim/run_lap.py --town Town01
