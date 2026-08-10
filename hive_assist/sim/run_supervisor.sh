#!/usr/bin/env bash
# D4.1 — the supervisor runs NATIVE, pinned, and slightly elevated.
#
# It is deliberately not in docker-compose.yml. It is the only component that can
# emit a setpoint, so it should not be sharing a scheduler with twenty flight
# stacks and a physics server. Two cores are reserved for it and DDS; PX4 gets
# the rest. Containerising it buys isolation we do not want and costs determinism
# we do.
#
#   ./run_supervisor.sh --emit           gate AND transmit
#   ./run_supervisor.sh                  decide only (default, safe)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
BIN="$REPO/brain/rust/target/release/swarm-supervisor"

CPUS="${HIVE_SUPERVISOR_CPUS:-0-1}"
NICE="${HIVE_SUPERVISOR_NICE:--5}"

if [[ ! -x "$BIN" ]]; then
  echo "supervisor not built. From $REPO:"
  echo "  cargo build --release --manifest-path brain/rust/Cargo.toml"
  exit 1
fi

echo "supervisor: cpuset $CPUS, nice $NICE, native"
echo "  binary $BIN"

# taskset pins it away from the PX4 herd; nice keeps it ahead of them when the
# machine saturates, which at this RAM budget it will
exec nice -n "$NICE" taskset -c "$CPUS" "$BIN" \
  --estimate /tmp/hive/estimate.json \
  --config "$HERE/config/supervisor.json" \
  "$@"
