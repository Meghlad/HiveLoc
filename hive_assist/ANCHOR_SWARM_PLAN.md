# Anchor-Referenced Cooperative Swarm — Architecture & Implementation Plan

> **Status:** live plan. Each domain lands as a working artifact under `hive_assist/`.
> **Builds on:** the Brain (`../brain/`) — iSAM2 estimator, `PLAN_SCHEMA`,
> Rust `swarm-supervisor`, MAVLink pipeline. Reuse first, extend second.

---

## RESTRUCTURE — 2026-08-08

This revision pulls the project back to its original goal: **a few drones fly to a
location, GPS-denied, localized by inter-drone ranges plus an anchor (ground or
flying), closing a real loop through a real autopilot.**

The pivot is one decision: **delete the vision front end.** The estimator core was
always range + anchor + a constant-velocity motion prior (`brain` `day5`→`day6`→
`close_the_loop.py`); the VIO/IMU/RGB-D path was a second stack layered on top, and
it is the sole reason anything needed a GPU. Removing it deletes, not defers, every
red row in `done_till_now.md` — the 14.9 m in-flight lie, the 7 m-on-a-stationary-
airframe, the IMU-starved subscribers, the Ogre-Next EGL segfault, the RTF ceiling —
because those were all vision/render faults. What remains fits the 8 GB M1 with room
to spare, because ranges are geometry, not pixels.

**Domain-level effect of the restructure:**

| Domain | Before | After |
|---|---|---|
| 1 Anchor estimation | 15-DOF `NavState` + IMU preintegration + RGB-D `Between` | **range + anchor + motion prior only**; IMU/attitude/gravity fusion moves to the autopilot's EKF3 |
| 2 Topology / CBBA / FSM | unchanged | unchanged |
| 3 Dispatch / standoff | unchanged | unchanged |
| 4 Simulation | 20-veh PX4 + Gazebo + cameras + VIO on Zephyrus | **N-veh headless ArduCopter SITL, MAVLink-native, on the M1** |
| 5 ROS 2 | (implicit, load-bearing) | **explicit, optional wrapper, post-core** |

**Metric change:** RTF is retired as the headline number. With no renderer, RTF ≈ 1
is automatic; the number that proves flight-readiness — and that industry evaluates —
is **loop latency and jitter** (target < 25 ms end to end). See §4.7.

---

## 0. Scope & Handoff Boundary

**What is NOT our domain:** target *detection*. Some external system finds the
target and produces a geodetic coordinate `(lat, lon[, alt])`.

**What IS our domain:** everything from *coordinate-in-hand* onward — establishing a
metric frame, holding an anchored loiter mesh, electing a sub-team, and dispatching
an agent to a standoff station at the coordinate.

**Input contract:** a single message `{lat, lon, alt?, task_id}`.
**Output contract:** per-tick, per-vehicle `SET_POSITION_TARGET_LOCAL_NED` setpoints,
already gated by the supervisor.

The anchor is **GPS-surveyed**, so `TacFrame` has a rigid, known transform to
geodetic coordinates. The external target lat/lon is projected into `TacFrame` once
on ingest; all downstream control stays in the local metric frame.

---

## 1. Frames & the Geodetic → TacFrame Transform

- **WGS84 geodetic** `(lat, lon, alt)` — the external system's output.
- **ECEF** — intermediate.
- **TacFrame (local ENU)** — origin = surveyed anchor `A`, axes East-North-Up, yaw
  fixed by the survey. The frame the estimator, supervisor, and MAVLink `LOCAL_NED`
  setpoints live in.

Because `A` is surveyed, `T_geo→tac ∈ SE(3)` is known and constant. Target ingest:

```
X_ecef = geodetic_to_ecef(lat, lon, alt)
X_enu  = R_ecef→enu(A_lla) · (X_ecef − A_ecef)
X_tac  = R_z(−yaw_offset) · X_enu
```

**Deliverable D1.0 — DONE:** `hive/frames.py`, `tests/test_frames.py`.

---

## 2. Mission / CONOPS State Flow

```
ANCHOR_INIT → LOITER_MESH → TASK_INGEST → AUCTION → RECONFIG
                                                       │
                                             (external go-signal)
                                                       ▼
                                                   DISPATCH → APPROACH
                                                       │
                                                  STANDOFF_TASK → REJOIN
```

Every transition is a pre-validated, forward-simulated setpoint stream over horizon
`H`. The supervisor is never routed around; the compiler only emits streams that
satisfy the invariants by construction, so `ACCEPT` is the steady state and any
`REJECT` is a caught bug.

---

## Domain 1 — Anchor-Assisted Estimation & Full-Rank Hessian

**Goal:** with a surveyed anchor supplying range (and optional bearing) factors, show
the estimator's information matrix `H` becomes full-rank — the gauge freedom (global
translation + yaw) collapses from dimension 3 (2D) to **0**, with no artificial
pinning.

### 1.1 State — post-restructure

Per agent `i`, per keyframe `t`: position `p_i^t ∈ R²` in `TacFrame`. Anchor pose
`T_A` is a known constant, not estimated.

**The live estimator now matches the 2D reference.** IMU preintegration and RGB-D
`BetweenFactorPose3` are removed from *our* factor graph. IMU, attitude, and
gravity are fused by the **autopilot's EKF3**, which already does this job well;
our estimator's sole output is a **position** in `TacFrame`, handed to EKF3 as
`VISION_POSITION_ESTIMATE`. This is a separation of concerns, not a capability
loss: `hive/anchored_isam2.py` (218 tests, the pinned 2D reference) *is* this
estimator. GPS stays OFF (`GPS_TYPE=0`, EKF3 external-nav source set); the anchor's
observability argument in §1.3 is only non-vacuous with GPS off.

> Reverses the "GOLDEN IMPLEMENTATION locked 2026-08-05" 15-DOF decision — see the
> restructure banner. The 15-DOF NavState existed to fuse gz IMU + RGB-D in the
> Gazebo world; with the autopilot's EKF3 doing IMU fusion, it is redundant.

### 1.2 MAP objective

```
J(x) = Σ ‖e_motion‖²_Σm      # constant-velocity prior  (p_{t-2} − 2p_{t-1} + p_t) — the odometry factor
     + Σ ‖e_range(i,j)‖²/σ²   # inter-agent range (Huber ρ)
     + Σ ‖e_bearing(i,j)‖²/σ² # inter-agent bearing (optional, off by default)
     + Σ ‖e_anchor(i,A)‖²_ΣA  # anchor→agent range (+ optional anchor bearing)  ← the pin
```

`e_motion` is the "no teleporting" smoothness term from `day5/day6` — a soft
constant-velocity constraint that connects keyframes and keeps under-constrained
frames non-singular. It replaces the IMU-preintegration `e_odom`; on real hardware
it is swapped for real UWB-rate odometry behind the same factor slot.

### 1.3 Observability argument (unchanged — the core theory)

- Without the anchor factor, `J` is invariant under global translation `t` and yaw
  `φ` → `H·n = 0` for gauge generators `{n_tx, n_ty, n_φ}`.
- An anchor **range** factor makes global translation observable; an anchor
  **bearing** (or a second surveyed anchor) makes global **yaw** observable.
- ⇒ null-space generators gain gradient support ⇒ `H = JᵀJ ≻ 0`.

Measured rank ladder (`hive/nullspace.py`, singular-value gap 1.5e13 — not
threshold-sensitive):

| configuration | `dim ker(H)` | why |
|---|---|---|
| no anchor | **3** | full SE(2) gauge |
| 1 anchor, 1 agent, 1 keyframe | **2** | one scalar range on a 3-DoF body |
| 1 anchor, 1 agent, all keyframes | **1** | temporal baseline recovers translation |
| 1 anchor, all agents, 1 keyframe | **1** | spatial baseline does the same, instantly |
| + ANCHOR-frame bearing (surveyed) | **0** | external heading → full rank |
| 2 surveyed anchors, range only | **0** | second known point → full rank |

Range-only pins the centroid **radially to 2.7 cm**, leaves it **tangentially free**
(prior-limited) — the yaw null-space as a position error scaling with anchor-to-swarm
distance. A surveyed anchor removes *drift* (real external information); a reference
*drone* removes only the *singularity* (the cloud drifts with it).

**Deliverables D1.x**
- D1.1 `hive/anchor_factor.py` — analytic residuals + Jacobians for anchor range (and
  optional anchor bearing) + gauge generators. **DONE.**
- D1.2 `hive/nullspace.py` + `tests/test_nullspace.py` — the rank ladder. **DONE.**
- D1.3 `hive/anchored_isam2.py` — iSAM2 with anchor factors; bounded covariance along
  former gauge directions. **DONE (218 tests).**

---

## Domain 2 — Topology Partitioning & Supervisor-Certified FSM

**Goal:** split the swarm into an active sub-team and a residual loiter set, elect the
sub-team by auction, hand over between modes without tripping a supervisor gate.

### 2.1 Adjacency cut
`A_ij(t) = 1[‖p_i − p_j‖ ≤ R_comm]`. After electing set `S`, permute `S` first:
```
A_c(t⁺) = Pᵀ (A_sub1 ⊕ A_sub2) P
L_sub  = D − A_sub
```
Physical links persist; only the **control** graph is block-diagonalized.

### 2.2 CBBA / auction (single task, N-slot coalition)
```
c_i(X) = w_d/(1+dist(p_i,X)) + w_b·batt_i + w_s·sensor_i − w_r·κ_loiter_i
```
`κ_loiter_i` penalizes pulling an agent whose removal breaks loiter coverage.
Max-consensus adopts neighbor `(y_k, z_k)` if `y_k > y_i` (tie-break by id);
converges in ≤ `diam(G)` rounds to a conflict-free set.

### 2.3 FSM with invariant guards
| Transition | Guard (over horizon `H`) |
|---|---|
| `LOITER_MESH→AUCTION` | task ingested; zero setpoint change → no gate can fire |
| `AUCTION→RECONFIG` | consensus conflict-free ∧ predicted min-clearance ≥ `d_clear` |
| `RECONFIG` | rate-limited Lloyd re-solve (over `M−N`); per-tick slew ≤ `v_max`, spacing ≥ `d_clear` |
| `RECONFIG→DISPATCH` | full approach stream predicted in-geofence, `v ≤ v_max`, clearance ≥ `d_clear` before motion |

**Deliverables D2.x**
- D2.1 `hive/cbba.py` + `tests/test_cbba.py` — convergence on random graphs. **DONE.**
- D2.2 `hive/supervisor_gate.py` — Python mirror of the Rust supervisor, cross-checked
  in `tests/test_gate_parity.py`. **DONE.**
- D2.3 `hive/mission_fsm.py` + sim run: `M→N` handover, zero supervisor rejections.
  **DONE (logic); to be re-exercised against the new Domain 4 loop — see S1.**

---

## Domain 3 — Event-Triggered Dispatch & Guided Standoff Approach

Dispatch is **reactive**. On the external go-signal, the first agent to receive it
breaks off and runs an individual guided approach to a **standoff station** near
`X_tac`. No impact-time consensus, no simultaneity coupling, no homing onto the point.

> **Out of scope:** terminal-homing-to-point laws, PN + `t_go`-bias salvo timing, any
> `r_i → 0` multi-vehicle convergence onto one coordinate.

### 3.1 Standoff station
```
s_i = X_tac + d_s · [cos θ, sin θ]   # distinct viewing/approach point, not X itself
```

### 3.2 Guided approach (vector-field path following)
```
ψ_des = ψ_tan(γ) − arctan( k_e · e_i / v_i )
ė      = v_i · sin(ψ − ψ_tan(γ))       # cross-track → 0
γ̇      = v_i / L_i(γ)                  # single agent, no coupling
```
Saturated to `[v_min, v_max]`, `[ω_min, ω_max]`. Every setpoint passes the supervisor.

**Deliverables D3.x**
- D3.1 `hive/standoff.py` — station placement + vector-field follower. **DONE.**
- D3.2 `tests/test_approach.py` — `e→0`, arrival at `s_i`, every setpoint valid.
  **DONE.**

---

## Domain 4 — Range-Only MAVLink-Native Closed Loop (RESTRUCTURED)

> **Replaces** the 20-vehicle PX4 + Gazebo + camera Domain 4. Runs on the **M1**.
> No renderer, no camera, no Gazebo, no PX4, no ROS 2 in the critical path.

### 4.1 Architecture

```
 N headless ArduCopter SITL          ground anchor(s)          (S3) flying anchor
 (GPS OFF, EKF3 external nav)        surveyed, fixed           GPS leader, known pos
        │ true position                     │                        │
        │ (SITL ground-truth channel,        │                        │
        │  NOT LOCAL_POSITION_NED)            ▼                        ▼
        └────────────►  RANGE SIMULATOR  ◄────────────────────────────┘
                        noisy pairwise + anchor ranges (reuse measure())
                                 │
                                 ▼
                        iSAM2 ESTIMATOR (Domain 1, unchanged factor types)
                        per-vehicle position + marginal covariance trace
                                 │
                 ┌───────────────┴─────────────────┐
                 ▼                                   ▼
        VISION_POSITION_ESTIMATE              ORCHESTRATOR (Domains 2–3)
        per vehicle → each EKF3 fuses         formation / translate / dispatch → Plan
                 ▲                                   │
                 │                                   ▼
                 │                          RUST SUPERVISOR (existing gate)
                 │                          geofence · spacing · cov · staleness
                 │                                   │ accepted
                 │                                   ▼
                 └──── true pos moves ◄──── SET_POSITION_TARGET_LOCAL_NED per vehicle
```

The loop is real: estimator error → EKF3 → controller → motors → true position →
ranges → estimate. Only the UWB radio is simulated (ranges from geometry), exactly as
SITL already fakes GPS. Swapping to real UWB is a hardware step, not an architecture
change.

### 4.2 Range simulator

Reuse `measure()` from `brain` `day5/day6`: inter-agent ranges within `R_comm`, anchor
ranges within `R_anchor` (anchors are mains-powered infrastructure — their range does
**not** degrade with the inter-drone links; verified). Noise model: Gaussian + NLOS +
occasional outlier + dropout, Huber-gated downstream.

**Correctness trap — read this.** Take positions from SITL's **ground-truth** channel
(the simulator state, e.g. `SIM_STATE`), *not* `LOCAL_POSITION_NED`. The latter is the
EKF's own fused output; feeding it back to generate its own measurements closes a fake
loop on itself. (Verify the exact ground-truth field per instance as D4.1.)

### 4.3 N-vehicle SITL + per-vehicle external nav

- Launch `sim_vehicle.py --count N` (ArduCopter, Copter-4.5.7), headless, fixed
  instance offsets, one MAVLink endpoint per vehicle.
- Per vehicle: GPS off; EKF3 external-nav source set; `SET_GPS_GLOBAL_ORIGIN` once
  before arming; `POSZ = Baro (1)`, `VELZ` off (feeding position only — don't fight
  the climb command).
- Keep the external-nav stream **continuous** — a stream that dies mid-loop throws
  "AHRS waiting for home."

### 4.4 Orchestrator

Emits the existing `Plan` (`plan_id`, `issued_unix_ms`, `min_spacing_m`,
`assignments:[{vehicle, waypoint_ne}]`). Modes:
- **hold** — fixed formation (diamond/square).
- **translate** — move the formation centroid to a target within anchor reach.
- **dispatch** — Domain-3 standoff approach for the elected agent.

Every `Plan` passes the Rust supervisor before any setpoint leaves.

### 4.5 Resourcing (M1, 8 GB)

| Component | Est. RAM |
|---|---|
| ArduCopter SITL ×N (headless, ~50–100 MB ea) | 0.3–0.6 GB (N=6) |
| Python iSAM2 estimator + range sim (GTSAM) | ~0.5 GB |
| Rust supervisor (native binary) | negligible |
| MAVProxy / MAVLink routing | ~0.1 GB |
| **Total (N=6)** | **~1.5 GB** |

N=4–6 is a comfortable demo; the ceiling is graph growth, not RAM — iSAM2's
incremental marginalization is exactly the fix for that.

### 4.6 Stage ladder — the build order

Each stage runs on the M1. Acceptance is the **error/sigma ratio ≈ 1** (an honest
sigma, not a small one) plus arrival at commanded waypoints.

- **S0 — one live vehicle, range-only closed loop (de-risk).** 1 SITL + 3–4 ground
  anchors. Estimator converges *at rest first* (calibrated: 0.065 m / 0.083 σ), then
  arm, then fly a small commanded square. Proves the in-flight break was VIO, not the
  range estimator. Smallest change to `close_the_loop.py`.
- **S1 — N-vehicle formation hold, ground-anchored.** N=4, per-vehicle estimate,
  supervisor accepts continuously. The "fleet flight" `done_till_now` never reached —
  unblocked because the VIO garbage and camera-vehicle IMU starvation are gone.
- **S2 — coordinated translation to a location (ground anchor).** Move the formation
  to a target **within anchor reach** (bulk translation is observable only through
  changing anchor ranges — stay inside the footprint). Optionally elect movers by CBBA.
  *This is the original goal.*
- **S3 — flying / GPS-leader anchor, migration beyond the footprint.** One GPS-equipped
  leader (position known) + GPS-denied followers ranging to it; the moving anchor
  carries the absolute frame, so the swarm migrates past any static anchor's reach.
  Honest caveat: a flying anchor resolves gauge freedom only while its own position is
  pinned (GPS gives that). The GPS-free mobile-anchor / leapfrog case is separate,
  research-grade, and scoped as such.

### 4.7 Transport & metrics

- **Transport is MAVLink end to end.** SITL over UDP; ranges + estimate in-process
  Python; supervisor a callable Rust gate; setpoints back over MAVLink. No DDS in the
  critical path.
- **RTF retired.** With no renderer, RTF ≈ 1 is automatic (ArduCopter SITL defaults to
  real-time and `SIM_SPEEDUP` can exceed 1 for faster iteration). RTF is a
  physics-lockstep number and there is no lockstep here.
- **New headline metric: loop latency & jitter.** Measure range-in →
  `VISION_POSITION_ESTIMATE`-out (and → setpoint-out). Target **< 25 ms** with
  sub-frame jitter — the number that proves the estimator would fly on real hardware
  and the one industry evaluates.

**Deliverables D4.x**
- D4.1 `sim/ground_truth_bridge.py` — per-vehicle true-position reader (verify
  `SIM_STATE` field); the range simulator's input.
- D4.2 `sim/range_world.py` — live `measure()` over N SITL truths + anchors.
- D4.3 `sim/live_estimator.py` — Domain-1 iSAM2 over N live vehicles; publishes
  `EstimateSnapshot{pos, cov_trace}`.
- D4.4 `sim/external_nav_fanout.py` — per-vehicle `VISION_POSITION_ESTIMATE` +
  `SET_GPS_GLOBAL_ORIGIN`, continuous stream.
- D4.5 `sim/orchestrator.py` — hold / translate / dispatch → `Plan` → supervisor →
  `SET_POSITION_TARGET_LOCAL_NED`.
- D4.6 `sim/run_fleet.sh` — launch N SITL + the loop, right start order (SITL →
  estimator-at-rest → arm).
- D4.7 `sim/loop_latency.py` — latency/jitter harness; the < 25 ms acceptance.
- D4.8 `tests/test_stationary_estimate.py` — a static vehicle must produce near-zero
  relative pose (the regression `done_till_now` §7.3 identified as missing).

---

## Domain 5 — ROS 2 Wrapper (OPTIONAL, post-core)

> **Not on the critical path.** The loop closes over MAVLink without ROS 2. This
> domain wraps the working loop as ROS 2 nodes so the ROS 2 competency is demonstrated
> — valuable for autonomy roles and portfolio. Build it **after S2**, never before.

- Thin `rclrs` (ros2-rust, Jazzy) nodes wrapping the existing Rust layers, mirroring
  the `supervisor_demo` graph already proven in `ros2_ws/`.
- Estimator publishes `/swarm_estimate` (`EstimateSnapshot`); orchestrator publishes
  `/mission_plan`; supervisor publishes `/plan_decision`. Any monitor can subscribe.
- Host natively via **Pixi + RoboStack** (osx-arm64), removing the Docker
  fix-loss failure mode (`done_till_now` §6). Budget yak-shaving on Jazzy osx-arm64
  package coverage and the `rclrs` + `swarm_msgs` build (get `swarm_msgs` onto
  `AMENT_PREFIX_PATH` before colcon-ros-cargo resolves the generated crate).
- Adds transport overhead but renders nothing → stays comfortably real-time; does
  **not** affect the S0–S3 latency numbers, which are measured on the MAVLink loop.

**Deliverables D5.x**
- D5.1 `ros2_ws/` Pixi manifest (Jazzy + rclrs env), replacing the Docker build.
- D5.2 `estimate_publisher` / `plan_publisher` rclrs nodes over the live loop.
- D5.3 launch + a verified trace: estimate → plan → decision on live topics.

---

## Execution status & binding lessons

**Done and valid:** offline layers (Brain Parts A–C); **224 unit tests pass**;
Domain 1 null-space ladder; Domain 2 CBBA/gate/FSM; Domain 3 standoff; estimator
**calibrated at rest** (0.065 m err / 0.083 σ = 0.78); **one vehicle completes a
GPS-denied, compass-free flight** (held 3.87–4.22 m for 25 s, landed on command).

**Designed out by the restructure:** the in-flight 14.9 m-at-10 cm-σ lie, the VIO
front end reporting 7 m of motion on a stationary airframe, camera-vehicle IMU
starvation, the Gazebo EGL/render failure, the RTF ceiling. All were vision/render
faults; none survive the deletion of the vision path. (Full 45-entry forensic log
retained in git history / prior `ANCHOR_SWARM_PLAN` for provenance.)

**Operational lessons that constrain the new loop:**
- Ground truth in, estimate out — never generate ranges from the EKF's own output.
- Ground truth is `SIMSTATE` (int32 degE7, exact to 1.1 cm). `SIM_STATE.lat` is a
  **float32 holding a degE7 value** — quantised to 32 units, ~18 cm worst case,
  twelve times the UWB sigma. Its `lat_int`/`lon_int` extensions are fine. This
  is the D4.1 verification, and it has an answer now.
- One socket, one reader. pymavlink's `recv_match` discards what it does not
  match, so a second consumer eats the first's traffic — including the
  STATUSTEXT that explains an arming refusal.
- Start the external-nav sender BEFORE setting the global origin, not after; the
  origin sleeps are otherwise a hole in the stream at the worst moment.
- Start order: SITL → estimator converges **at rest** → arm → command. Never stream
  setpoints at vehicles that never armed (it prints `done` and silently invalidates
  any "after-flight" measurement).
- External-nav stream must be **continuous**; `SET_GPS_GLOBAL_ORIGIN` before arm.
- `POSZ = Baro`, `VELZ` off when feeding position only.
- Acceptance is **error/sigma ≈ 1**, not a small sigma — a confidently wrong estimate
  is the exact failure the supervisor would certify, and the whole project's thesis.

---

## Deliverables index

| ID | Artifact | Status |
|---|---|---|
| D1.0–D1.3 | frames, anchor factor, null-space, anchored iSAM2 | ✅ done |
| D2.1–D2.3 | cbba, supervisor gate, mission FSM | ✅ done (D2.3 re-run in S1) |
| D3.1–D3.2 | standoff placement + follower, approach tests | ✅ done |
| D4.1 | `sim/ground_truth_bridge.py` — SIMSTATE, source verified at runtime | ✅ done |
| D4.2 | `sim/range_world.py` — live `measure()` + day-7 noise model | ✅ done |
| D4.3 | `sim/live_estimator.py` — iSAM2 over N live vehicles, fixed-lag | ✅ done |
| D4.4 | `sim/external_nav_fanout.py` — continuous extnav + origin | ✅ done |
| D4.5 | `sim/orchestrator.py` — modes → Plan → gate → setpoints + the loop | ✅ done |
| D4.6 | `sim/run_fleet.sh` + `sim/params/gps_denied.parm` | ✅ done |
| D4.7 | `sim/loop_latency.py` — p99 8.4 ms live / 16.0 ms offline vs 25 ms | ✅ done |
| D4.8 | `tests/test_stationary_estimate.py` (+ `test_domain4_loop.py`) | ✅ done |
| D5.1–D5.3 | ROS 2 (Pixi/RoboStack) wrapper | ▢ optional, post-S2 |

**261 tests pass.** The old `ros2_ws/` (rclpy nodes bound to PX4 topics and gz)
was deleted with the vision stack; Domain 5 is rebuilt fresh as `rclrs` under
Pixi after S2, per §5. `supervisor_io.py` was kept and moved to `hive/` — it is
pure JSON file hand-off with no ROS import, and the orchestrator needs exactly it.

### Stage status

- **S0 — DONE, live.** One headless ArduCopter SITL, `GPS_TYPE=0`, no camera, no
  renderer. Estimator converged at rest, then armed, flew all four legs of a 5 m
  commanded square at 1.86–2.34 m AGL, landed, disarmed. **Error 0.026 m against
  a reported sigma of 0.017 m — ratio 1.44, in motion.** 130 plans submitted,
  130 ACCEPT, 0 REJECT, 0 horizon-guard failures. Range-in → external-nav on the
  wire: p50 1.8 ms, p99 8.4 ms. External-nav stream continuous throughout.
  That is the thesis: the old stack was calibrated at rest (0.78) and 147×
  overconfident the moment anything moved. Same acceptance criterion, in flight,
  with no vision anywhere.
- **S1–S3 — wired, unrun.** `run_fleet.sh -n 4 -m hold|translate` launches them;
  nothing in the code path is single-vehicle-specific, but they have not been
  exercised, so no numbers are claimed.
