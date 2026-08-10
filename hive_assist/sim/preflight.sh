#!/usr/bin/env bash
# D4.2 — host capability check. Run this BEFORE docker compose up.
#
# Everything here is a thing that, if wrong, wastes twenty minutes and a reboot.
# The RAM budget in docker-compose.yml has ~1-3 GB of headroom on a 16 GB
# machine, which is not enough to absorb a leak, so the swapfile check is a hard
# failure rather than a warning.
#
#   ./preflight.sh            check
#   ./preflight.sh --fix      check, and offer to create the swapfile
set -uo pipefail

FIX=0
[[ "${1:-}" == "--fix" ]] && FIX=1

PASS=0 WARN=0 FAIL=0
ok()   { printf '  \033[32m[ ok ]\033[0m  %s\n' "$1"; PASS=$((PASS+1)); }
warn() { printf '  \033[33m[warn]\033[0m  %s\n' "$1"; WARN=$((WARN+1)); }
bad()  { printf '  \033[31m[FAIL]\033[0m  %s\n' "$1"; FAIL=$((FAIL+1)); }

echo
echo "hive_assist sim preflight"
echo "========================================================================"

# ---------------------------------------------------------------- platform
echo "platform"
if [[ "$(uname -s)" != "Linux" ]]; then
  bad "$(uname -s) — this stack needs Linux. gz-sim headless rendering against
         NVIDIA and 'tc netem' on veth pairs both require it. Domain 4 runs on
         the Zephyrus; the rest of hive_assist runs anywhere."
else
  ok "Linux $(uname -r)"
fi

# ---------------------------------------------------------------- memory
echo
echo "memory  (the binding constraint)"
if [[ -r /proc/meminfo ]]; then
  MEM_KB=$(awk '/MemTotal/{print $2}' /proc/meminfo)
  MEM_GB=$((MEM_KB / 1024 / 1024))
  SWAP_KB=$(awk '/SwapTotal/{print $2}' /proc/meminfo)
  SWAP_GB=$((SWAP_KB / 1024 / 1024))

  if   (( MEM_GB >= 15 )); then ok "RAM ${MEM_GB} GB"
  elif (( MEM_GB >= 11 )); then warn "RAM ${MEM_GB} GB — drop to HIVE_N_VEHICLES=12"
  else bad "RAM ${MEM_GB} GB — not enough for a 20-vehicle stack"; fi

  if (( SWAP_GB >= 8 )); then
    ok "swap ${SWAP_GB} GB"
  else
    bad "swap ${SWAP_GB} GB — the budget leaves 1-3 GB of headroom, which one
         leak erases. Want 8 GB."
    if (( FIX )); then
      echo
      read -r -p "  create /swapfile-hive (8 GB, needs sudo)? [y/N] " a
      if [[ "$a" == "y" ]]; then
        sudo fallocate -l 8G /swapfile-hive &&
        sudo chmod 600 /swapfile-hive &&
        sudo mkswap /swapfile-hive &&
        sudo swapon /swapfile-hive &&
        echo "  created. Add to /etc/fstab to persist:" &&
        echo "    /swapfile-hive none swap sw 0 0"
      fi
    else
      echo "         sudo fallocate -l 8G /swapfile-hive && sudo chmod 600 /swapfile-hive"
      echo "         sudo mkswap /swapfile-hive && sudo swapon /swapfile-hive"
      echo "         (or re-run: ./preflight.sh --fix)"
    fi
  fi

  if command -v zramctl >/dev/null 2>&1 && zramctl 2>/dev/null | grep -q zram; then
    ok "zram active"
  else
    warn "no zram — optional, but it buys real headroom at this budget"
  fi
fi

# ---------------------------------------------------------------- cpu
echo
echo "cpu"
CORES=$(nproc 2>/dev/null || echo 0)
if (( CORES >= 12 )); then ok "${CORES} threads"
elif (( CORES >= 8 )); then warn "${CORES} threads — PX4 will be tight at 20 vehicles"
else bad "${CORES} threads"; fi

if [[ -d /sys/fs/cgroup ]]; then
  ok "cgroups present (needed to cpuset the supervisor)"
else
  warn "no cgroups — run_supervisor.sh cannot pin cores"
fi

# ---------------------------------------------------------------- gpu
echo
echo "gpu"
if command -v nvidia-smi >/dev/null 2>&1; then
  VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
  if [[ -n "$VRAM" ]] && (( VRAM >= 7000 )); then
    ok "NVIDIA, ${VRAM} MiB VRAM"
  else
    warn "NVIDIA, ${VRAM:-?} MiB VRAM — give cameras to the active drones only"
  fi
  if docker info 2>/dev/null | grep -qi nvidia; then
    ok "nvidia container runtime registered with docker"
  else
    bad "docker cannot see the NVIDIA runtime — install nvidia-container-toolkit"
  fi
else
  warn "no nvidia-smi — gz-sim will fall back to software rendering (slow, but
         it will run if you cut the cameras entirely)"
fi

# ---------------------------------------------------------------- docker
echo
echo "docker"
if command -v docker >/dev/null 2>&1; then
  if docker info >/dev/null 2>&1; then
    ok "daemon reachable"
  else
    bad "daemon not reachable (start it, or add yourself to the docker group)"
  fi
  docker compose version >/dev/null 2>&1 \
    && ok "compose v2" \
    || bad "docker compose v2 not found"
else
  bad "docker not installed"
fi

# ---------------------------------------------------------------- netem
echo
echo "fault injection"
if command -v tc >/dev/null 2>&1; then
  ok "tc present"
  if modinfo sch_netem >/dev/null 2>&1 || grep -q netem /proc/modules 2>/dev/null; then
    ok "sch_netem available"
  else
    bad "sch_netem missing — install linux-modules-extra-\$(uname -r)"
  fi
else
  bad "tc missing — install iproute2 (netem_sweep.sh needs it)"
fi

# ---------------------------------------------------------------- disk
echo
echo "disk"
AVAIL=$(df -BG --output=avail . 2>/dev/null | tail -1 | tr -dc '0-9')
if [[ -n "$AVAIL" ]] && (( AVAIL >= 60 )); then
  ok "${AVAIL} GB free"
elif [[ -n "$AVAIL" ]]; then
  warn "${AVAIL} GB free — images + rosbags want ~60 GB, and it should be NVMe"
fi

echo
echo "========================================================================"
printf '  %d ok, %d warn, %d fail\n' "$PASS" "$WARN" "$FAIL"
if (( FAIL > 0 )); then
  echo "  Fix the failures before 'docker compose up' — every one of them is a"
  echo "  twenty-minute detour discovered halfway through a build."
  exit 1
fi
echo "  Ready. Start small:  HIVE_N_VEHICLES=8 docker compose up --build"
echo "  Then climb to 20 while watching:  watch -n2 free -g"
