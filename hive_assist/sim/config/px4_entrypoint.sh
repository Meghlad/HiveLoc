#!/usr/bin/env bash
# D4.1 — one PX4 SITL instance, attaching to the SHARED gz-sim server.
#
# PX4_GZ_STANDALONE=1 is the whole point: without it every PX4 instance spawns
# its own Gazebo, which is the arrangement that puts a 20-vehicle stack ~20 GB
# over a 16 GB budget. With it, they all attach to the single `gz-server`
# service and the physics is hosted once.
#
# The instance index comes from the container hostname, which docker compose
# sets to <project>-px4-<N> for scaled replicas. That way there is exactly one
# service definition regardless of fleet size.
set -euo pipefail

HOST="$(hostname)"
IDX="${HOST##*-}"
[[ "$IDX" =~ ^[0-9]+$ ]] || IDX=1
IDX=$((IDX - 1))                       # compose replicas are 1-based, PX4 is 0-based

# lay the fleet out on a ring around the surveyed anchor, matching the loiter
# mesh the FSM expects at ANCHOR_INIT
N="${HIVE_N_VEHICLES:-20}"
RADIUS="${HIVE_RING_RADIUS:-8.0}"
POSE=$(awk -v i="$IDX" -v n="$N" -v r="$RADIUS" \
  'BEGIN{ a = 2*3.14159265358979*i/n; printf "%.3f,%.3f,0", r*cos(a), r*sin(a) }')

export PX4_SYS_AUTOSTART=4001
export PX4_GZ_MODEL_POSE="$POSE"
export PX4_GZ_STANDALONE=1
export PX4_SIM_MODEL="${PX4_SIM_MODEL:-gz_x500}"
export PX4_GZ_WORLD="${PX4_GZ_WORLD:-anchor_site}"

# cameras are the VRAM killers on an 8 GB card: only the ACTIVE drones carry
# depth/RGB, the loiter mesh is kinematic + range-only
if [[ ",${HIVE_CAMERA_VEHICLES:-}," == *",${IDX},"* ]]; then
  export PX4_SIM_MODEL="gz_x500_depth"
fi

echo "px4 instance $IDX/$N  pose=$POSE  model=$PX4_SIM_MODEL"
exec /px4/build/px4_sitl_default/bin/px4 -i "$IDX" /px4/build/px4_sitl_default/etc
