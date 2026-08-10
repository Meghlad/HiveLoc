# 🛰️ The Four Layers — Overview & Build Order

The estimator (Days 1–8) localizes a GPS-denied swarm from radio ranges and closes the loop into ArduCopter. These four layers take it from an estimator demo to a system: they make the estimate **trustworthy** (Layer 2), the transport **real-time** (Layer 1), the interface **operable** (Layer 3), and the whole thing **ROS-native**. Each layer has a tutorial in this folder.

Built in the order they deepen the work — not the order they're numbered.

| Order | Layer | What it adds | Deliverable | Tutorial |
|---|---|---|---|---|
| 1 | **Layer 2 — Vision bearings** | a camera whose bearing factors constrain exactly the DOF ranges leave ambiguous; the sensor that makes marginal covariance honest | how many vision detections/frame make a marginal range graph well-conditioned — a measured plot | [LAYER2_TUTORIAL.md](LAYER2_TUTORIAL.md) |
| 2 | **Layer 1 — Rust transport** | `swarm-link`: estimator/transport split, one 20 Hz sender per vehicle, borrow-checked ownership | estimate-to-wire p50/p99, Python vs Rust, N=1 & N=12 | [LAYER1_TUTORIAL.md](LAYER1_TUTORIAL.md) |
| 3 | **ROS 2 (rclrs)** | the Rust layers as ROS 2 nodes; bearings, estimates, plan verdicts become topics | `ros2 launch` demo — **built & verified**: accept + spacing-reject + live covariance-reject on topics | [ROS2_TUTORIAL.md](ROS2_TUTORIAL.md) |
| 4 | **Layer 3 — Mission planner** | language-conditioned planner (`claude-opus-4-8`, constrained decoding) + a Rust safety supervisor that assumes the model is wrong | a plan that never reaches MAVLink unless geofence + spacing + covariance + freshness all pass | [LAYER3_TUTORIAL.md](LAYER3_TUTORIAL.md) |

## The through-line

**Estimation quality and safety are the same problem.** Layer 2's degraded range-only estimator reported σ = 4 cm while actually erring 28 cm — a floppy graph makes the covariance a liar. Bearing factors made it honest. Layer 3's supervisor gates every plan on that same covariance. So the safety guarantee at the top of the stack is only meaningful because of the sensor fusion at the bottom.

## Headline numbers (all measured)

- **Layer 2:** at the marginal radius R=0.37, ~16 vision detections/frame turn a 33%-rigid range graph into 100%-rigid, cutting median RMSE 0.26 m → 0.05 m. Full ONNX pipeline (pixels → Rust `ort` → track-based association → iSAM2): **RMSE 0.20 m → 0.076 m** in the degraded-radio condition, within 2× of a perfect-identity oracle. Detector: recall 94%, precision 96%, bearing RMS 0.15°.
- **Layer 1:** at N=12, Rust estimate-to-wire **p99 336 µs vs Python 1,988 µs (~6×)**; Python's median grows 4.8× from 1→12 vehicles (the GIL), Rust's stays flat.
- **Layer 3:** supervisor is 11 passing unit tests incl. the hallucinated-hillside rejection; a bad waypoint produces **zero** MAVLink packets.

## Build everything

```bash
# --- estimator venv (Python) ---
python src/estimation/day8_isam2_traj.py                 # the estimator the layers build on

# --- Layer 2 science + perception pipeline ---
python src/vision/layer2_bearing_phase_diagram.py    # the deliverable plot
python src/vision/layer2_isam2_bearing.py            # bearings in the live estimator
python src/vision/layer2_make_dataset.py             # 1440 camera frames + detector.onnx
cargo build --release --manifest-path rust/Cargo.toml
ORT_DYLIB_PATH=$PWD/.venv/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.1.27.0.dylib \
  ./rust/target/release/swarm-perception --frames frames --model data/detector.onnx --out bearings.jsonl
python src/vision/layer2_perception_closeloop.py

# --- Layer 1 benchmark ---
python src/transport/layer1_bench.py                    # writes layer1_latency.png

# --- Layer 3 planner + supervisor ---
cargo test -p swarm-supervisor --release --manifest-path rust/Cargo.toml   # 11 tests
python src/planning/layer3_vlm_planner.py "form a tight line along the north edge"

# --- ROS 2 (needs Docker or a ROS 2 Jazzy host) ---
docker build -f ros2_ws/Dockerfile -t coop-swarm-ros .
docker run --rm -it coop-swarm-ros ros2 launch swarm_bringup supervisor_demo.launch.py
```

## Resume framing

- Layer 2 is the strongest technical content — a quantified sensor-fusion result that deepens the validated estimator.
- Layer 1's "why Rust" answer is a measured latency histogram, not an assertion. Rust's real footprint is embedded/safety-critical (Ferrocene is a qualified toolchain for ISO 26262 / IEC 61508), not blockchain.
- Layer 3 is a **language-conditioned mission planner with a safety supervisor** — never a VLA. The action space is waypoints; the low-level policy is a flight controller better than anything we'd train; the open problem in a GPS-denied swarm is whether the estimate is trustworthy enough to act on.
