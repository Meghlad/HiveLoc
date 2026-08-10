# RUN — hive_assist

> All commands run **from `hive_assist/`** with the repo venv active.

## Setup

```bash
cd hive_assist
source ../.venv/bin/activate          # or: python3 -m venv ../.venv && pip install -r ../brain/requirements.txt
pip install pytest scipy
```

`gtsam`, `numpy`, `scipy` and `matplotlib` come from the Brain's
`requirements.txt`. `networkx` is deliberately **not** used — the graph code is
numpy-only so this has one fewer dependency than it looks like it should.

## Everything at once

```bash
make            # tests, then every figure
make test       # 218 tests, ~4 s
make figures    # regenerate figures/*.png
```

## Domain 1 — anchored estimation

```bash
python -m hive.frames              # frame self-check, round-trip error
python -m hive.nullspace           # the rank ladder + covariance tables + figure
python -m hive.anchored_isam2      # anchored vs pinned over 120 keyframes + figure
```

`nullspace.py` prints the whole ladder, names the surviving free direction
(rotation about the anchor, alignment 1.000000000), and reports the radial vs
tangential centroid uncertainty that makes the rank result physical.

## Domain 2 — auction and the guarded FSM

```bash
python -m hive.mission_fsm         # the M->N handover, 80/80 ACCEPT
```

The parity test needs the Rust supervisor built. Without it those two tests skip
and everything else still runs:

```bash
cargo build --release --manifest-path ../brain/rust/Cargo.toml
python -m pytest tests/test_gate_parity.py -v
```

## Domain 3 — event-triggered dispatch

```bash
python -m hive.standoff            # all four task geometries + figure
```

## Domain 4 — safe hold

Here, host-independently:

```bash
python -m hive.loss_model          # the 0-30% loss sweep + figure
```

On the Zephyrus (Linux + NVIDIA + Docker) — see
[sim/README.md](sim/README.md) for what still has to be built there:

```bash
cd sim
./preflight.sh                     # --fix to offer creating the swapfile
HIVE_N_VEHICLES=8 docker compose up --build       # start small, then climb to 20
../sim/run_supervisor.sh --emit                   # native, cpuset-pinned
./netem_sweep.sh --iface veth-drone7 --auto
```

Watch RAM while scaling up — the budget leaves 1–3 GB of headroom:

```bash
watch -n2 free -g
```

## Tests

```bash
python -m pytest tests/                       # all 218
python -m pytest tests/test_nullspace.py -v   # the D1 rank ladder
python -m pytest tests/test_safe_hold.py -v   # the D4 lunge finding
```
