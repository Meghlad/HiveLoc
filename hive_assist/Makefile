# hive_assist — run from this directory.
PY ?= ../.venv/bin/python

.PHONY: all test figures d1 d2 d3 d4 latency s0 s1 s2 supervisor clean

all: test figures

test:
	$(PY) -m pytest tests/

# Each module prints its own result table and writes its figure, so `figures`
# doubles as the "show me the findings" target.
figures: d1 d2 d3 d4

d1:
	$(PY) -m hive.frames
	$(PY) -m hive.nullspace
	$(PY) -m hive.anchored_isam2

d2:
	$(PY) -m hive.mission_fsm

d3:
	$(PY) -m hive.standoff

# Domain 4, offline: the safe-hold sweep, the radio, and the estimator running
# over it. None of these need SITL.
d4:
	$(PY) -m hive.loss_model
	$(PY) -m sim.range_world
	$(PY) -m sim.live_estimator

# D4.7 — the metric that replaced RTF. Exits non-zero if p99 misses 25 ms.
latency: supervisor
	$(PY) -m sim.loop_latency -n 4 -t 400

# The Rust gate. Required by the parity tests, the latency harness's setpoint
# path, and every live stage.
supervisor:
	@test -x ../brain/rust/target/release/swarm-supervisor \
	  || cargo build --release --manifest-path ../brain/rust/Cargo.toml

# --- live stages (need ArduPilot SITL built at $ARDUPILOT_HOME) -------------
# S0  one vehicle, range-only closed loop: converge at rest, then fly a square.
s0: supervisor
	./sim/run_fleet.sh -n 1 -m square -s 60

# S1  N-vehicle formation hold, ground-anchored.
s1: supervisor
	./sim/run_fleet.sh -n 4 -m hold -s 60

# S2  coordinated translation, staying inside the anchor footprint.
s2: supervisor
	./sim/run_fleet.sh -n 4 -m translate -s 90

# S3  D3's standoff dispatch, live: the CBBA auction elects a coalition, each
#     elected agent flies its own curvature-bounded approach to its own slot on
#     the standoff perimeter, and the rest hold the mesh. --gcs puts the target
#     and both stations on the QGroundControl map.
s3: supervisor
	./sim/run_fleet.sh -n 4 -m standoff -s 120 --gcs

clean:
	rm -rf figures/*.png .pytest_cache sim/logs
	find . -name __pycache__ -type d -exec rm -rf {} +
