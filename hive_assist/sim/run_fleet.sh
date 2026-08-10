#!/usr/bin/env bash
# D4.6 — launch N headless ArduCopter SITL instances and close the loop on them.
#
#   ./run_fleet.sh                       S0: 1 vehicle, hold, 40 s
#   ./run_fleet.sh -n 4 -m hold          S1: 4-vehicle formation hold
#   ./run_fleet.sh -n 4 -m translate     S2: coordinated translation
#   ./run_fleet.sh -n 1 --no-fly         S0 de-risk: converge at rest only
#   ./run_fleet.sh -n 4 --sitl-only      bring the fleet up, drive it yourself
#   ./run_fleet.sh -n 1 -m square --gcs  same, plus telemetry to QGroundControl
#   ./run_fleet.sh -n 4 -m standoff --gcs   S3: auction -> 2-agent standoff
#                                        dispatch, target drawn on the QGC map
#
# NO RENDERER, NO GAZEBO, NO PX4, NO ROS 2. That is the restructure: ranges are
# geometry, not pixels, so nothing here needs a GPU and the whole fleet fits on
# an 8 GB M1 with room to spare. Each arducopter instance is ~50-100 MB.
#
# START ORDER IS THE DELIVERABLE, not the process list. The plan's operational
# lessons are unambiguous and every one of them was paid for:
#
#   1. SITL up and streaming SIMSTATE            (truth exists)
#   2. estimator converges AT REST               (the estimate is worth flying on)
#   3. external nav priming, then global origin  (AHRS can set home)
#   4. arm                                       (never stream at an unarmed vehicle)
#   5. command
#
# Steps 2-5 live inside sim/orchestrator.py, which is why this script's job ends
# at "SITL is up and healthy" and it then hands over rather than racing it.
#
# TRANSPORT. Each instance gets --serial2 udpclient:127.0.0.1:PORT, so the
# autopilot pushes to a port the loop binds. SITL's socket is connected to that
# port, so replies from the same bound socket are accepted: one hop,
# bidirectional, no MAVProxy in the critical path. PORT = 14551 + 10*i, which is
# also the Rust supervisor's --base-port/--port-stride default, so a port number
# means the same thing everywhere in this stack.
#
# SERIAL2 AND NOT SERIAL0, AND THAT CHOICE IS THE SECURITY BOUNDARY. ArduPilot
# gates unsigned MAVLink with a compiled-in callback, not a parameter, and
# libraries/GCS_MAVLink/GCS_Signing.cpp:116 reads:
#
#     if (status == mavlink_get_channel_status(MAVLINK_COMM_0)) {
#         // always accept channel 0, assumed to be secure channel. This
#         // is USB on PX4 boards
#         return true;
#     }
#
# Channel 0 accepts every unsigned frame unconditionally, on the assumption it
# is a physically-secure USB console. serial0 IS channel 0. While the closed
# loop ran there, provisioning a signing key hardened our outgoing frames and
# hardened the vehicle's inbound path not at all — anyone able to put a datagram
# on that port could still inject a command or, worse, a forged
# VISION_POSITION_ESTIMATE (COMMS_HARDENING_PLAN.md §1.3). Moving the loop to
# serial2 puts it on a channel where reject-unsigned is real.
#
# GCS::setup_uarts() assigns channels by walking the MAVLink-protocol serial
# ports in order, so serial2 lands on channel 1 or 2 depending on whether the
# --gcs observer occupies serial1. Either is non-zero, which is the only
# property that matters here.
#
# serial0 stays a console, but it has to be passed EXPLICITLY as `tcp:0`.
# SITL's default for serial0 is `tcp:5760:wait`, and the `:wait` blocks in
# accept() until something connects — with the loop moved to serial2, nothing
# ever does, and the whole fleet hangs at "Waiting for connection ...." with no
# heartbeat and no error. Measured, once, the hard way.
#
# `tcp:0` and not `tcp:5760`: UARTDriver.cpp:331 treats a port <= 1000 as an
# offset from this instance's base_port, so each vehicle gets its own console
# instead of N instances fighting over 5760.
#
# That console remains an unauthenticated path by ArduPilot's design — it is
# the channel-0 hole above — and must never be exposed past localhost. On real
# hardware it is the USB port you plug into on the bench.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
REPO="$(cd "$ROOT/.." && pwd)"

ARDUPILOT="${ARDUPILOT_HOME:-$HOME/ardupilot}"
BIN="${ARDUCOPTER_BIN:-$ARDUPILOT/build/sitl/bin/arducopter}"
DEFAULTS="$ARDUPILOT/Tools/autotest/default_params/copter.parm"
GPS_DENIED="$HERE/params/gps_denied.parm"
PY="${PY:-$REPO/.venv/bin/python}"

# The surveyed anchor / TacFrame origin. Must match sim/ground_truth_bridge.py's
# SITL_HOME_*: the estimator scores itself against a frame anchored here, so a
# mismatch shows up as a constant position offset that looks like a bias.
HOME_LAT="${HIVE_HOME_LAT:--35.363261}"
HOME_LON="${HIVE_HOME_LON:-149.165230}"
HOME_ALT="${HIVE_HOME_ALT:-584}"
HOME_YAW="${HIVE_HOME_YAW:-0}"

N=1
MODE="hold"
SECONDS_ARG=40
SPACING=6.0
SPEEDUP="${HIVE_SPEEDUP:-1}"
EXTRA=()
SITL_ONLY=0
GCS_PORT=0

usage() { sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'; exit 0; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--vehicles)  N="$2"; shift 2 ;;
    -m|--mode)      MODE="$2"; shift 2 ;;
    -s|--seconds)   SECONDS_ARG="$2"; shift 2 ;;
    --spacing)      SPACING="$2"; shift 2 ;;
    --speedup)      SPEEDUP="$2"; shift 2 ;;
    --sitl-only)    SITL_ONLY=1; shift ;;
    --gcs)          if [[ "${2:-}" =~ ^[0-9]+$ ]]; then
                      GCS_PORT="$2"; shift 2
                    else
                      GCS_PORT=14550; shift
                    fi ;;
    -h|--help)      usage ;;
    *)              EXTRA+=("$1"); shift ;;
  esac
done

# ---------------------------------------------------------------- preflight
fail=0
note() { printf '  %-34s %s\n' "$1" "$2"; }
check() { if [[ -e "$2" ]]; then note "$1" "ok"; else note "$1" "MISSING: $2"; fail=1; fi; }

echo "preflight"
check "arducopter binary"        "$BIN"
check "ArduPilot default params" "$DEFAULTS"
check "GPS-denied params"        "$GPS_DENIED"
check "python (gtsam + pymavlink)" "$PY"
if [[ -x "$REPO/brain/rust/target/release/swarm-supervisor" ]]; then
  note "rust supervisor" "ok"
else
  note "rust supervisor" "MISSING — cargo build --release --manifest-path $REPO/brain/rust/Cargo.toml"
  fail=1
fi
if (( fail )); then echo "preflight failed"; exit 1; fi

# RAM: ~50-100 MB per instance plus ~0.5 GB for the estimator. The ceiling on
# this machine is graph growth, not memory, but a machine already at its limit
# will show up as SITL scheduling jitter and be misread as estimator latency.
if command -v vm_stat >/dev/null 2>&1; then
  free_gb=$(vm_stat | awk '/Pages free/ {gsub(/\./,"",$3); printf "%.1f", $3*16384/1073741824}')
  note "free RAM" "${free_gb} GB (need ~$(awk -v n="$N" 'BEGIN{printf "%.1f", 0.5 + n*0.1}') GB)"
fi
echo

LOGDIR="$HERE/logs"
mkdir -p "$LOGDIR"
PIDS=()

cleanup() {
  echo
  echo "stopping ${#PIDS[@]} SITL instance(s) ..."
  for pid in "${PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ------------------------------------------------------------------ launch
# Spawn positions on a ring of radius $SPACING about the anchor. Distinct homes
# matter for more than tidiness: N vehicles spawned on one coordinate have zero
# pairwise range, which is a degenerate geometry the estimator would be scored
# on and a collision the supervisor would (correctly) refuse to plan through.
echo "launching $N ArduCopter SITL instance(s), headless, speedup ${SPEEDUP}x"
for ((i=0; i<N; i++)); do
  read -r vlat vlon <<<"$(awk -v lat="$HOME_LAT" -v lon="$HOME_LON" \
                              -v i="$i" -v n="$N" -v r="$SPACING" '
    BEGIN {
      pi = 3.14159265358979;
      if (n <= 1) { print lat, lon; exit }
      a  = 2*pi*i/n;
      de = r*cos(a);                       # east  metres
      dn = r*sin(a);                       # north metres
      printf "%.9f %.9f\n", lat + dn/111320.0, lon + de/(111320.0*cos(lat*pi/180));
    }')"

  port=$((14551 + 10*i))
  log="$LOGDIR/vehicle_${i}.log"
  echo "  vehicle $i: home ${vlat},${vlon}  ->  udp 127.0.0.1:${port}   (log $log)"

  # Each instance runs in its OWN directory. arducopter writes eeprom.bin,
  # logs/ and terrain/ into the CWD with no way to redirect them, so launching
  # from the repo root drops three untracked artifacts into the source tree and
  # makes N instances fight over one eeprom.bin. A subshell keeps the cd local.
  work="$LOGDIR/instance_${i}"
  mkdir -p "$work"

  # serial1 is a read-only observer for a GCS. It is deliberately a SEPARATE
  # port: serial2 carries the closed loop, and a GCS sharing it would drain
  # messages the loop's own reader needs (recv_match discards non-matching
  # traffic).
  #
  # Under `--sign`, "read-only" stops being a convention. ArduPilot activates a
  # provisioned key on EVERY channel (GCS_Signing.cpp:94-100), and QGroundControl
  # holds no key, so it keeps receiving telemetry — signed frames still decode —
  # while its own commands are refused. Measured both ways against SITL: ACKed
  # before provisioning, refused after. Watching a flight still works; steering
  # it from QGC does not, which is the intent.
  gcs_arg=()
  if (( GCS_PORT )); then
    gcs_arg=(--serial1 "udpclient:127.0.0.1:${GCS_PORT}")
  fi

  (
    cd "$work"
    exec "$BIN" \
      --model quad \
      --instance "$i" \
      --wipe \
      --speedup "$SPEEDUP" \
      --sysid "$((i+1))" \
      --home "${vlat},${vlon},${HOME_ALT},${HOME_YAW}" \
      --defaults "${DEFAULTS},${GPS_DENIED}" \
      --serial0 "tcp:0" \
      --serial2 "udpclient:127.0.0.1:${port}" \
      ${gcs_arg[@]+"${gcs_arg[@]}"}
  ) >"$log" 2>&1 &
  PIDS+=("$!")
done

# Give the autopilots time to boot and start streaming. The loop then waits for
# an actual heartbeat, so this is a courtesy rather than a synchronisation.
sleep 6
for ((i=0; i<N; i++)); do
  if ! kill -0 "${PIDS[$i]}" 2>/dev/null; then
    echo "vehicle $i died during boot — last lines of its log:"
    tail -20 "$LOGDIR/vehicle_${i}.log"
    exit 1
  fi
done
echo "  all $N instance(s) alive"
if (( GCS_PORT )); then
  echo "  GCS telemetry -> udp 127.0.0.1:${GCS_PORT}  (open QGroundControl; it auto-connects)"
fi
echo

if (( SITL_ONLY )); then
  echo "--sitl-only: fleet is up. Endpoints:"
  for ((i=0; i<N; i++)); do echo "  vehicle $i  udpin:127.0.0.1:$((14551 + 10*i))"; done
  echo "Ctrl-C to stop."
  wait
  # `exit`, not fall-through. Ctrl-C fires the trap and hides this, but if the
  # SITL processes exit on their own -- OOM, a crash -- `wait` returns normally
  # and the script would continue into the closed loop below, streaming
  # setpoints at a fleet that is already dead and reporting a 60 s heartbeat
  # timeout instead of "your vehicles died".
  exit 0
fi

# -------------------------------------------------------------------- loop
cd "$ROOT"
"$PY" -m sim.orchestrator \
  --vehicles "$N" --mode "$MODE" --seconds "$SECONDS_ARG" \
  --spacing "$SPACING" ${EXTRA[@]+"${EXTRA[@]}"}
