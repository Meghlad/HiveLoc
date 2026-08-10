#!/usr/bin/env bash
set -e
source /opt/ros/jazzy/setup.bash
source /workspace/coop-swarm/ros2_ws/install/setup.bash

# colcon's merged setup util omits our custom message package (swarm_msgs) from
# install/setup.bash, so force EVERY built prefix onto the runtime paths. Without
# this the swarm_msgs typesupport .so (Rust nodes) and Python module (rclpy
# bringup) are not found at launch time.
WS=/workspace/coop-swarm/ros2_ws/install
for p in "$WS"/*/; do
  p="${p%/}"
  case ":$AMENT_PREFIX_PATH:" in *":$p:"*) ;; *) export AMENT_PREFIX_PATH="$p:$AMENT_PREFIX_PATH" ;; esac
  [ -d "$p/lib" ] && export LD_LIBRARY_PATH="$p/lib:$LD_LIBRARY_PATH"
  for sp in "$p"/lib/python3.12/site-packages "$p"/local/lib/python3.12/dist-packages; do
    [ -d "$sp" ] && export PYTHONPATH="$sp:$PYTHONPATH"
  done
done

cd /workspace/coop-swarm
exec "$@"
