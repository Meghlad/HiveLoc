#!/usr/bin/env bash
# D4.2 — the fault-injection sweep. The highest-value experiment in Domain 4.
#
# Degrade one vehicle's link across 0-30% loss and verify the failure mode is
# ALWAYS "freeze safely":
#
#     delayed estimate  ->  EstimateStale fires
#     dropped plan      ->  zero setpoints emitted, the vehicle holds
#
# WHAT TO LOOK FOR, because the naive reading of this sweep passes while the
# system is unsafe. hive/loss_model.py runs the same experiment in simulation and
# found that the dangerous moment is not the outage — it is the RECOVERY. While
# the vehicle is frozen the planner keeps advancing its stream, so the first plan
# to land afterwards can be several ticks' worth of distance away, and it passes
# every gate the supervisor currently has (it is fresh, in-fence, correctly
# spaced, from a trusted vehicle). Freshness bounds a plan's AGE. Nothing bounds
# its DISTANCE.
#
# So this script reports MAX COMMANDED JUMP alongside the hold statistics, and
# that column is the one that matters. Expect:
#
#     ~0.9 m   the FSM re-planned from the vehicle's actual position: correct
#     >3 m     a stale stream was accepted: the supervisor needs SlewTooLarge
#              (see the recommendation in hive/loss_model.py)
#
#   ./netem_sweep.sh                     sweep the default vehicle
#   ./netem_sweep.sh --iface veth-drone7 --auto
#   ./netem_sweep.sh --clear             remove all qdiscs and exit
set -uo pipefail

IFACE="${IFACE:-veth-drone7}"
DWELL="${DWELL:-45}"                 # seconds per loss level
LOSSES=(0 2 5 8 12 18 24 30)
DELAY_MS=50
JITTER_MS=15
REORDER_PCT=2
OUT="${OUT:-reports/netem_$(date +%Y%m%d_%H%M%S)}"
AUTO=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --iface)  IFACE="$2"; shift 2 ;;
    --dwell)  DWELL="$2"; shift 2 ;;
    --out)    OUT="$2"; shift 2 ;;
    --auto)   AUTO=1; shift ;;
    --clear)  tc qdisc del dev "$IFACE" root 2>/dev/null
              echo "cleared qdisc on $IFACE"; exit 0 ;;
    *) echo "unknown argument: $1"; exit 2 ;;
  esac
done

command -v tc >/dev/null || { echo "tc missing — install iproute2"; exit 1; }
ip link show "$IFACE" >/dev/null 2>&1 || {
  echo "no such interface: $IFACE"
  echo "list the swarm's veth pairs with:  ip -o link | grep veth"
  exit 1
}

mkdir -p "$OUT"
CSV="$OUT/sweep.csv"
echo "loss_pct,accepted,rejected,estimate_stale,plan_stale,max_jump_m,max_hold_s,verdict" > "$CSV"

cleanup() { tc qdisc del dev "$IFACE" root 2>/dev/null; }
trap cleanup EXIT INT TERM

echo
echo "netem sweep on $IFACE  —  ${DWELL}s per level, output -> $OUT"
echo "========================================================================"
printf '%7s %10s %10s %14s %14s %12s\n' \
  "loss" "accepted" "rejected" "EstimateStale" "max jump (m)" "verdict"
echo "------------------------------------------------------------------------"

for LOSS in "${LOSSES[@]}"; do
  cleanup
  if (( LOSS > 0 )); then
    tc qdisc add dev "$IFACE" root netem \
       delay "${DELAY_MS}ms" "${JITTER_MS}ms" distribution normal \
       loss "${LOSS}%" reorder "${REORDER_PCT}%" 2>/dev/null
  else
    tc qdisc add dev "$IFACE" root netem \
       delay "${DELAY_MS}ms" "${JITTER_MS}ms" distribution normal 2>/dev/null
  fi

  LOG="$OUT/decisions_${LOSS}.jsonl"
  : > "$LOG"

  # The supervisor writes one JSON decision per evaluated plan. Collect the
  # window, then read it — never sample a running counter, because a sweep that
  # measures its own scheduler jitter measures nothing.
  timeout "$DWELL" ros2 topic echo --no-daemon /hive/plan_decision \
      --field data > "$LOG" 2>/dev/null || true

  ACC=$(grep -c '"accepted": *true'  "$LOG" 2>/dev/null || echo 0)
  REJ=$(grep -c '"accepted": *false' "$LOG" 2>/dev/null || echo 0)
  EST=$(grep -c 'EstimateStale'      "$LOG" 2>/dev/null || echo 0)
  PLN=$(grep -c 'PlanStale'          "$LOG" 2>/dev/null || echo 0)

  JUMP=$(python3 - "$LOG" <<'PY' 2>/dev/null || echo 0
import json, sys, math
prev, worst = None, 0.0
for line in open(sys.argv[1], errors="ignore"):
    try:
        d = json.loads(line)
    except Exception:
        continue
    if not d.get("accepted"):
        continue
    wps = [a["waypoint_ne"] for a in d.get("assignments", [])]
    if prev and len(prev) == len(wps):
        worst = max(worst, max(math.dist(a, b) for a, b in zip(prev, wps)))
    prev = wps
print(f"{worst:.2f}")
PY
)

  # awk, not `python3 -c`: the obvious f-string version nests single quotes
  # inside an already single-quoted f-string. That is a Python syntax error that
  # `bash -n` cannot see, because to bash the whole thing is just a string.
  HOLD=$(awk -v r="$REJ" -v a="$ACC" -v d="$DWELL" \
             'BEGIN{ t=a+r; if (t < 1) t=1; printf "%.1f", r/t*d }')

  VERDICT="safe-hold"
  awk "BEGIN{exit !($JUMP > 3.0)}" && VERDICT="LUNGE"
  (( ACC == 0 && LOSS < 30 )) && VERDICT="stalled"

  printf '%6s%% %10s %10s %14s %14s %12s\n' \
    "$LOSS" "$ACC" "$REJ" "$EST" "$JUMP" "$VERDICT"
  echo "$LOSS,$ACC,$REJ,$EST,$PLN,$JUMP,$HOLD,$VERDICT" >> "$CSV"

  (( AUTO )) || { read -r -p "  enter for next level (q to stop): " k
                  [[ "$k" == "q" ]] && break; }
done

cleanup
echo "========================================================================"

WORST=$(awk -F, 'NR>1 && $6+0>m {m=$6+0} END{printf "%.2f", m}' "$CSV")
LUNGES=$(awk -F, 'NR>1 && $8=="LUNGE"' "$CSV" | wc -l)

echo "worst commanded jump across the sweep: ${WORST} m"
if (( LUNGES > 0 )); then
  cat <<EOF

  ${LUNGES} level(s) produced a LUNGE.

  This is the failure hive/loss_model.py predicts, and it is not a tuning
  problem. The plan that caused it was fresh, inside the geofence, correctly
  spaced, and from a vehicle the estimator trusts — every gate the supervisor
  has, passed. Two fixes, and the sweep shows you need both:

    1. brain/rust/swarm-supervisor: add a SlewTooLarge violation rejecting any
       waypoint further from the vehicle's estimate than v_max * dt.
    2. the planner must re-plan from the vehicle's ACTUAL position after a hold.
       MissionFSM already does this; anything else feeding the supervisor must
       too, or the gate above will simply stall the mission instead.
EOF
  exit 1
fi

echo "  No lunges. Failure mode was freeze-safely at every level."
echo "  Report: $CSV"
