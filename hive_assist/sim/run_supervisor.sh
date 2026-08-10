#!/usr/bin/env bash
# D4.1 — the supervisor runs NATIVE, pinned, and slightly elevated.
#
# It is deliberately not in docker-compose.yml. It is the only component that can
# emit a setpoint, so it should not be sharing a scheduler with twenty flight
# stacks and a physics server. Two cores are reserved for it and DDS; PX4 gets
# the rest. Containerising it buys isolation we do not want and costs determinism
# we do.
#
#   ./run_supervisor.sh                  one-shot, decide only (default, safe)
#   ./run_supervisor.sh --emit           one-shot, gate AND transmit
#   ./run_supervisor.sh --loop --emit    poll the shared dir, gate every plan
#
# THE BINARY IS ONE-SHOT BY DESIGN: read a plan, validate, emit, exit. That is
# right for a gate — there is no state to carry between plans, and a gate with
# memory is a gate that can be argued into a decision. --loop does not change
# that; it just re-invokes the same one-shot binary each time the FSM writes a
# new plan, so every plan is judged from scratch.
#
# --scale 1.0 IS NOT OPTIONAL. The binary defaults to 5.0 ("metres of NED per
# estimator unit"), which is right for the Brain's normalised Crazyflie units
# and catastrophically wrong here: hive_assist plans in metres, and the scale is
# applied AFTER validate(), so a waypoint the gate just certified inside a
# +/-40 m fence would be transmitted five times further out. Every geofence,
# spacing and clearance guarantee would be void at the last step.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
BIN="$REPO/brain/rust/target/release/swarm-supervisor"

SHARED="${HIVE_SHARED_DIR:-/tmp/hive}"
CPUS="${HIVE_SUPERVISOR_CPUS:-0-1}"
NICE="${HIVE_SUPERVISOR_NICE:--5}"
POLL_S="${HIVE_SUPERVISOR_POLL_S:-0.05}"
SCALE="${HIVE_SUPERVISOR_SCALE:-1.0}"
ALT="${HIVE_SUPERVISOR_ALT:-2.5}"
# PX4 SITL instance i listens for offboard MAVLink on 14580+i, not the
# 14551+10i the binary defaults to (that is the ArduPilot-style pattern the
# Brain used). Wrong here means setpoints land on nothing, silently.
BASE_PORT="${HIVE_SUPERVISOR_BASE_PORT:-14580}"
PORT_STRIDE="${HIVE_SUPERVISOR_PORT_STRIDE:-1}"

LOOP=0
ARGS=()
for a in "$@"; do
  case "$a" in
    --loop) LOOP=1 ;;
    *) ARGS+=("$a") ;;
  esac
done

if [[ ! -x "$BIN" ]]; then
  echo "supervisor not built. From $REPO:"
  echo "  cargo build --release --manifest-path brain/rust/Cargo.toml"
  exit 1
fi

mkdir -p "$SHARED"
COMMON=(--estimate "$SHARED/estimate.json"
        --config "$HERE/config/supervisor.json"
        --scale "$SCALE" --alt "$ALT"
        --base-port "$BASE_PORT" --port-stride "$PORT_STRIDE")

# A negative nice needs CAP_SYS_NICE. Without it `nice` warns on every single
# invocation and, in loop mode, that is once per plan — noise that would bury
# the REJECT lines this loop exists to surface. Priority is a nice-to-have;
# gating is not, so drop the request rather than the run.
# `nice` WARNS and still returns 0 when it cannot lower the value, so the exit
# status is not the signal — the probe has to look at stderr.
PRIO=(nice -n "$NICE" taskset -c "$CPUS")
if [[ -n "$(nice -n "$NICE" true 2>&1 >/dev/null)" ]]; then
  echo "  note: no CAP_SYS_NICE — running at default priority"
  echo "        (grant it with: sudo setcap cap_sys_nice+ep $BIN)"
  PRIO=(taskset -c "$CPUS")
fi

echo "supervisor: cpuset $CPUS, nice $NICE, native"
echo "  binary $BIN"
echo "  shared $SHARED"
echo "  scale  $SCALE (1.0 = plan metres are NED metres)"

if (( LOOP == 0 )); then
  # taskset pins it away from the PX4 herd; nice keeps it ahead of them when the
  # machine saturates, which at this RAM budget it will
  exec "${PRIO[@]}" "$BIN" "${COMMON[@]}" "${ARGS[@]}"
fi

echo "  loop mode: polling $SHARED/plan.json every ${POLL_S}s"
echo "  every REJECT below is a bug report, not an expected event"

# THE LEFTOVER-PLAN PROBLEM. The FSM writes plan.json only while it has
# something to command: _step_toward_goal returns early once the movers are
# inside arrive_tol, so between dispatches the LAST plan of the previous one
# just sits on disk. With last="" the first mtime this loop ever sees always
# differs from it, so the loop's opening act was to submit that artifact — and
# the gate correctly called it PlanStale. That REJECT said nothing about the
# mission; it said the loop had judged a plan nobody was commanding.
#
# The rule below is deliberately narrow. A plan that predates the loop is
# skipped ONLY if it is already past the gate's own window, i.e. only when
# submitting it could not have produced anything but PlanStale. One that
# predates the loop but is still fresh IS judged — it is a live command and the
# loop has no business dropping it. And a plan that goes stale mid-run is still
# rejected loudly, because there the loop WAS watching and a >max_plan_age gap
# between write and verdict is a real pipeline fault.
MAX_PLAN_AGE_MS="$(sed -n 's/.*"max_plan_age_ms"[[:space:]]*:[[:space:]]*\([0-9]\+\).*/\1/p' \
                   "$HERE/config/supervisor.json" | head -n1)"
MAX_PLAN_AGE_MS="${MAX_PLAN_AGE_MS:-5000}"

last=""
if [[ -f "$SHARED/plan.json" ]]; then
  # AGE IS READ FROM issued_unix_ms, NOT FROM mtime. mtime is the right signal
  # for "a new plan appeared" (used in the poll below), but it is only a PROXY
  # for the plan's age, and the gate does not judge the proxy — it computes
  # now - plan.issued_unix_ms. The two normally agree to a few ms because the
  # FSM stamps and writes in one atomic step, but anything that moves a file
  # without rewriting it (a copy, a restore, a touch) splits them, and then a
  # skip decision made on mtime would disagree with the verdict it is trying to
  # predict. Reading the same field the gate reads makes the skip exact.
  pre_issued="$(sed -n 's/.*"issued_unix_ms"[[:space:]]*:[[:space:]]*\([0-9]\+\).*/\1/p' \
                 "$SHARED/plan.json" 2>/dev/null | head -n1)"
  if [[ -n "$pre_issued" ]]; then
    pre_age_ms=$(( $(date +%s%3N) - pre_issued ))
    if (( pre_age_ms > MAX_PLAN_AGE_MS )); then
      last="$(stat -c %.Y "$SHARED/plan.json" 2>/dev/null || echo "")"
      echo "  skipping the plan already on disk: ${pre_age_ms}ms old, past the"
      echo "  ${MAX_PLAN_AGE_MS}ms window — left over from before this loop started,"
      echo "  not a command. Waiting for the FSM to write a new one."
    fi
  fi
fi

accepted=0
rejected=0
trap 'echo; echo "supervisor: $accepted ACCEPT, $rejected REJECT"; exit 0' INT TERM

while true; do
  if [[ -f "$SHARED/plan.json" ]]; then
    # %.Y is FRACTIONAL mtime. Plain %Y is integer seconds, which at a 5 Hz
    # plan rate silently drops four plans in five — the vehicle would receive
    # 1 Hz while the planner believed it was emitting at 5 Hz, and the gap
    # between commanded and actual position would grow accordingly.
    stamp="$(stat -c %.Y "$SHARED/plan.json" 2>/dev/null || echo "")"
    if [[ -n "$stamp" && "$stamp" != "$last" ]]; then
      last="$stamp"
      # stdout is the Decision JSON, stderr is human commentary. Merging them
      # makes decision.json unparseable, so they are kept apart.
      set +e
      out="$("${PRIO[@]}" "$BIN" \
               "${COMMON[@]}" --plan "$SHARED/plan.json" "${ARGS[@]}" \
               2>"$SHARED/supervisor.err")"
      rc=$?
      set -e
      if (( rc == 0 )); then
        accepted=$((accepted + 1))
      else
        rejected=$((rejected + 1))
        echo "REJECT #$rejected:"
        # $out is the Decision JSON (violations included) even on reject —
        # main.rs prints it to stdout before exiting 2. Printing it here,
        # not just the generic stderr line, is the difference between "a
        # plan was rejected" and "THIS is what it violated". Also archived
        # per-rejection so the NEXT accepted plan's decision.json overwrite
        # (50ms later) does not erase the only copy of why.
        echo "$out" | sed 's/^/  /'
        printf '%s' "$out" > "$SHARED/last_reject_$(date +%s%N).json" 2>/dev/null || true
      fi
      if [[ -n "$out" ]]; then
        printf '%s' "$out" > "$SHARED/decision.json.tmp" \
          && mv -f "$SHARED/decision.json.tmp" "$SHARED/decision.json"
      fi
    fi
  fi
  sleep "$POLL_S"
done
