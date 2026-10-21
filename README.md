# Mini AV Stack

A small autonomous driving stack: map a real place, localize in it, perceive, predict, plan, and drive in a simulated twin.

Hardware: RTX 5070 Ti laptop (12 GB VRAM). Simulator: CARLA. Middleware: ROS 2.

## Layout
mapping/ localization/ perception/ prediction/ planning/ control/ sim/ scenarios/ docker/ docs/

## Phases
| # | Phase | Status |
|---|-------|--------|
| 1 | Foundation: CARLA + ROS 2 + pure pursuit/PID | done (demo: docs/media/phase1_demo.mp4) |
| 2 | Mapping and reconstruction | - |
| 3 | Localization (EKF + map matching) | - |
| 4 | Perception | - |
| 5 | Prediction and planning | - |
| 6 | Evaluation (20+ scenarios in CI) | - |

See [docs/phase1-setup.md](docs/phase1-setup.md) for setup.

## Phase 1: Foundation

A Tesla Model 3 follows 600 m of Town01 lane centre using pure pursuit (steering) and a PID (speed), at 8 m/s.
Green dots are the path, the red dot is the pure-pursuit target point. Rendered at Epic quality, `ClearSunset` weather.

    PYTHONPATH=control python sim/run_lap.py --camera chase --weather ClearSunset --realtime   # standalone
    bash sim/launch_demo.sh                                                                    # ROS 2: controller node + CARLA bridge

Result: mean cross-track error 0.08 m, max 1.01 m. Stack: CARLA 0.9.16 on Windows, ROS 2 Jazzy + PyTorch (CUDA 12.8) in WSL2 Ubuntu 24.04, RTX 5070 Ti.
`control/` is a ROS 2 package; `sim/carla_bridge.py` is a small custom bridge (publishes `/odom`, `/path`, applies `/cmd`).

### What broke
- Path wrapped around at its end, so the car steered at the start of the list and left the road (4 m mean error). Fix: stop at end of path.
- `rclpy` vanished because a launch script overwrote `PYTHONPATH`; the bridge silently did nothing because `main()` was never called.
- A killed bridge left CARLA in synchronous mode, so the next `load_world` timed out until the server was restarted.
- No speed planning yet: the car does not slow for corners (Phase 5).
