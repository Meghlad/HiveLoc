# RUN

> All commands run **from `brain/`** (this directory) with the venv active.
> Every script below resolves `data/` and `figures/` relative to the working
> directory, so `cd brain` first or nothing will be found.

## 1. Setup

```bash
cd brain
python3 -m venv ../.venv && source ../.venv/bin/activate
pip install -r requirements.txt
python src/tools/check.py
```

## 2. Part A — estimator

```bash
python src/estimation/day1_snl.py
python src/estimation/day2_rigidity.py
python src/estimation/day3_distributed.py
python src/estimation/day4_robust_certify.py
python src/estimation/day5_dynamic.py
python src/estimation/day6_gtsam_smoother.py
python src/estimation/day7_realistic_robust.py
python src/estimation/day8_isam2.py
python src/estimation/day8_isam2_traj.py
python src/viz/animate_swarm.py
```

## 3. Part C — Rust workspace

```bash
cargo build --release --manifest-path rust/Cargo.toml
cargo test  --release --manifest-path rust/Cargo.toml
```

## 4. Part C — L2 vision pipeline

```bash
python src/vision/layer2_bearing_phase_diagram.py
python src/vision/layer2_isam2_bearing.py
python src/vision/layer2_make_dataset.py
ORT_DYLIB_PATH=$PWD/.venv/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.1.27.0.dylib \
  ./rust/target/release/swarm-perception --frames frames --model data/detector.onnx --out bearings.jsonl
python src/vision/layer2_perception_closeloop.py
```

## 5. Part C — L1 latency benchmark

```bash
python src/transport/layer1_bench.py
```

## 6. Part C — L3 planner + interactive scenario

```bash
python src/planning/layer3_vlm_planner.py "form a tight line along the north edge"
python src/planning/swarm_scenario.py
```

## 7. Part C — ROS 2 demo

```bash
docker compose up --build
```

## 8. Part B — fly it (SITL finale)

```bash
cd ~/ardupilot
Tools/autotest/sim_vehicle.py -v ArduCopter --console --map --out 127.0.0.1:14551
```

```bash
param set VISO_TYPE 1
param set GPS_TYPE 0
param set EK3_SRC1_POSXY 6
param set EK3_SRC1_VELXY 0
param set EK3_SRC1_POSZ  1
param set EK3_SRC1_VELZ  0
param set EK3_SRC1_YAW   6
```

```bash
# restart SITL, then from the repo root in the estimator venv:
python src/flight/close_the_loop.py
# MAVProxy: mode guided → arm throttle → takeoff 2
```
