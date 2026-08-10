# 🤖 ROS 2 Tutorial — Wrapping the Rust Layers as ROS 2 Nodes (rclrs)

**The gap this closes:** the job listing marks "ROS Framework" mandatory, and the project had none. The cheapest fix with the best story is to wrap the Rust layers as ROS 2 nodes using **rclrs** (the Rust client library) — so bearing observations, pose estimates, and plan-accept/reject events become topics. This satisfies a hard requirement **and reinforces the Rust thread** instead of forking effort into a separate Python ROS demo.

---

## 1. The graph

```
              sensor_msgs/Image                swarm_msgs/BearingObservation
  camera ──────────────────────▶ [perception_node] ──────────────────────▶ (to estimator)
  (per vehicle)                    (Rust / rclrs)                            /bearings

  estimator ──▶ /swarm_estimate ──┐   swarm_msgs/SwarmEstimate
                (pose + cov)       │
                                   ▼
  planner ────▶ /mission_plan ──▶ [supervisor_node] ──▶ /plan_decision ──▶ (whole system)
                (Layer 3)           (Rust / rclrs)         accept/reject
                                   swarm_supervisor crate
```

Three ROS 2 packages, plus a Python bringup package:

| Package | Kind | Wraps | Topics |
|---|---|---|---|
| `swarm_msgs` | `ament_cmake` (rosidl) | — | defines `BearingObservation`, `SwarmEstimate`, `MissionPlan`, `Assignment`, `PlanDecision` |
| `swarm_supervisor_node` | `ament_cargo` (rclrs) | `swarm-supervisor` crate | sub `/mission_plan`, `/swarm_estimate` → pub `/plan_decision` |
| `swarm_perception_node` | `ament_cargo` (rclrs) | `swarm-perception` crate + `ort` | sub `/camera/image_raw` → pub `/bearings` |
| `swarm_bringup` | `ament_python` (rclpy) | Layer 2 & 3 replay | pub `/swarm_estimate`, `/mission_plan`; echo `/plan_decision` |

**Files:** `ros2_ws/src/…`, `ros2_ws/Dockerfile`.

---

## 2. Why this reinforces the Rust thread

The supervisor node is **not a re-implementation** — it depends on the exact `swarm-supervisor` crate the standalone binary and the 11 unit tests use:

```toml
# ros2_ws/src/swarm_supervisor_node/Cargo.toml
swarm-supervisor = { path = "../../../rust/swarm-supervisor" }
```

The node body is a thin rclrs shim: cache the latest `SwarmEstimate`, and on each `MissionPlan` call the library's `validate(plan, estimate, config, now)` and publish the `PlanDecision`. Same for perception — `perception_node` reuses `swarm_perception::{find_peaks, centroid, column_to_world_bearing}`, the same functions the file-driven CLI calls (I refactored those into `swarm-perception/src/lib.rs` precisely so both consumers share one source). One safety-critical code path, three front-ends (CLI, ROS node, tests).

The accept/reject decision being a **topic** is the point: any node — a logger, a fleet monitor, a ground station — can subscribe to `/plan_decision` and see exactly why a plan did or didn't reach the aircraft.

---

## 3. Build & run — verified end-to-end

The image **builds and the demo runs** (ros:jazzy + Rust + ros2-rust, ~5 GB). From the repo root (build context = repo root; the image needs `rust/` and the Python layers):

```bash
docker build -f ros2_ws/Dockerfile -t coop-swarm-ros .

docker run --rm -it coop-swarm-ros \
  ros2 launch swarm_bringup supervisor_demo.launch.py \
       instruction:="form a tight circle in the center"
```

**Actual verified trace** — the three nodes come up, the planner publishes a 12-assignment plan, the estimator replays real frames, and the supervisor validates each against the *current* frame:

```
[swarm_supervisor]   swarm_supervisor up: /mission_plan + /swarm_estimate -> /plan_decision
[plan_publisher]     planner source: offline-geometric; publishing plan 'offline-…' with 12 assignments
[estimate_publisher] replaying 120 frames of r055 estimate -> /swarm_estimate
[swarm_supervisor]   plan 'offline-…' ACCEPTED
[plan_publisher]     /plan_decision: plan 'offline-…' ACCEPTED
[swarm_supervisor]   plan 'offline-…' REJECTED (2 violations) — no setpoints emitted
[plan_publisher]     /plan_decision: … REJECTED  violations=['CovarianceTooHigh { vehicle: 1, trace: 0.00405, max: 0.004 }', 'CovarianceTooHigh { vehicle: 2, trace: 0.00439, max: 0.004 }']
```

**The covariance gate fires live**, mid-run: as the replayed estimate advances frame-to-frame, some frames carry a marginal covariance just above the 0.004 threshold on a couple of vehicles, and the supervisor refuses to command exactly those — the Layer 2 → Layer 3 through-line, on a topic.

Point it at a colliding formation to watch the spacing gate fire too:
```bash
docker run --rm coop-swarm-ros \
  ros2 launch swarm_bringup supervisor_demo.launch.py instruction:="stack everyone on one tiny point"
# → REJECTED  violations=['SpacingTooClose { a: 0, b: 1, dist: 0.0196, min: 0.08 }', ...]
```

The Dockerfile clones `ros2-rust` into the workspace, imports the jazzy `.repos`, drops the onnxruntime `.so` where the perception node's `load-dynamic` ort backend finds it (`ORT_DYLIB_PATH`) — the same pattern that would point at the JetPack onnxruntime on a Jetson — and `colcon build`s everything.

### Two build/run gotchas worth knowing (the debugging journey)

Getting a custom-message ROS package to link into a Rust node surfaced two real ros2-rust/colcon issues — both fixed in the Dockerfile and entrypoint:

| 🧱 Symptom | 🔍 Root cause | ✅ Fix |
|-----------|--------------|--------|
| `cargo: no matching package named 'swarm_msgs' found` when building the Rust nodes | colcon-ros-cargo discovers a message's generated Rust crate by scanning `AMENT_PREFIX_PATH` for `rust_packages` markers — but colcon's merged `install/setup.bash` **silently omits our custom `swarm_msgs`**, so its prefix is never on the path. (`std_msgs`, `rclrs`, etc. resolve fine.) | **Two-pass build**: build `swarm_msgs` + `rclrs` first, then `export AMENT_PREFIX_PATH="$(ls -d install/*/):$AMENT_PREFIX_PATH"` to force every built prefix on, then build the nodes. |
| Nodes build but launch can't find the `swarm_msgs` typesupport `.so` / Python module | Same omission bites at **runtime** — `ros2 launch` sources the same incomplete `setup.bash`. | `entrypoint.sh` forces every `install/*` prefix onto `AMENT_PREFIX_PATH`, `LD_LIBRARY_PATH`, and `PYTHONPATH`. |
| `error[E0061]: this method takes 1 argument but 2 were supplied` (×7) in the perception node | rclrs's parameter API is a builder: `declare_parameter("name").default(v).mandatory()?`, not `declare_parameter("name", v)`. | Use the builder; string params are `Arc<str>`. |

### Native half (buildable without ROS)

```bash
cargo build  --release --manifest-path rust/Cargo.toml   # 3 binaries + 2 libs
cargo test   --release --manifest-path rust/Cargo.toml   # 13 tests: 11 supervisor + 2 perception
```

---

## 4. Message design notes

- **`BearingObservation` carries no target identity** — just observer, world bearing, pixel, confidence. Data association is the estimator's job (it has the predicted state); keeping identity off the wire is the same seam the CLI enforces, and keeps the node swappable for a real camera.
- **`SwarmEstimate` carries `cov_trace`** — the per-vehicle marginal covariance is the trust signal the supervisor gates on. It rides the same message as the poses because they come from the same iSAM2 solve.
- **`PlanDecision` is published on every plan**, accept or reject, with human-readable `violations` — the audit trail is a topic, not a log line.

## 5. What to say in the room

- *"ROS was a hard requirement I didn't have, so I wrapped the Rust layers as rclrs nodes rather than writing a throwaway Python demo — the supervisor node depends on the same crate as the standalone binary and the unit tests, so there's one validation code path, not two."*
- *"The plan-accept/reject decision is a topic. Anything in the system can see why a plan was refused."*
- *"In the launch demo the covariance gate fires live — as the estimate replays, a couple of frames drift just over the trust threshold and the supervisor refuses to command exactly those vehicles. That's the Layer 2 → Layer 3 link happening on a ROS topic, not a slide."*
- *"The perception node and the CLI share `swarm-perception`'s lib functions — I refactored the detector math into a library so the ROS wrapper couldn't drift from the tested version."*
