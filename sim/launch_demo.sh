#!/usr/bin/env bash
# Run in WSL with the CARLA server already running on Windows.
set -e
cd "$(dirname "$0")/.."
source /opt/ros/jazzy/setup.bash
source ~/ws/install/setup.bash
ros2 run avstack_control controller_node &   # system python: started before the venv is active
CTRL=$!
trap 'kill $CTRL 2>/dev/null' EXIT
source ~/venvs/av/bin/activate               # bridge needs the carla client from the venv
PYTHONPATH="sim:$PYTHONPATH" python sim/carla_bridge.py
