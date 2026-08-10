# Anchor-Referenced Cooperative Swarm — Architecture & Implementation Plan

> **Status:** live plan. Each domain lands as a working artifact under
> `hive_assist/`.
> **Builds on:** the Brain (`../brain/`) — iSAM2 estimator, `PLAN_SCHEMA`,
> Rust `swarm-supervisor`, MAVLink pipeline. Reuse first, extend second.

---

## 0. Scope & Handoff Boundary

**What is NOT our domain:** target *detection*. Some external system finds the
target of interest and produces a geodetic coordinate `(lat, lon[, alt])`.

**What IS our domain:** everything from *coordinate-in-hand* onward —
establishing a metric frame, holding an anchored loiter mesh, electing a
sub-team, and dispatching an agent to a standoff station at the coordinate for
the task (inspection / sampling / delivery / probe placement).

**Our input contract:** a single message `{lat, lon, alt?, task_id}`.
**Our output contract:** per-tick, per-vehicle `SET_POSITION_TARGET_LOCAL_NED`
setpoints, already gated by the supervisor.

The anchor is **GPS-surveyed**, so `TacFrame` has a *rigid, known* transform to
geodetic coordinates. The external target lat/lon is projected into `TacFrame`
**once** on ingest; all downstream control stays in the local metric frame the
supervisor and MAVLink already speak.

---

## 1. Frames & the Geodetic → TacFrame Transform

- **WGS84 geodetic** `(lat, lon, alt)` — the external system's output.
- **ECEF** — intermediate.
- **TacFrame (local ENU)** — origin = surveyed anchor `A`, axes East-North-Up,
  yaw fixed by the survey. This is the frame the estimator, supervisor, and
  MAVLink `LOCAL_NED` setpoints live in.

Because `A` is surveyed, the transform `T_geo→tac ∈ SE(3)` is **known and
constant** (no drift, no estimation). Target ingest is a one-shot:

```
X_ecef = geodetic_to_ecef(lat, lon, alt)
X_enu  = R_ecef→enu(A_lla) · (X_ecef − A_ecef)   # ENU about the anchor
X_tac  = R_z(−yaw_offset) · X_enu                # known ENU→TacFrame yaw
```

**Deliverable D1.0 — DONE:** [`hive/frames.py`](hive/frames.py),
[`tests/test_frames.py`](tests/test_frames.py).

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

Every transition is a **pre-validated, forward-simulated setpoint stream** over
horizon `H`. The supervisor is never *routed around*; the compiler only ever
emits streams that satisfy the invariants by construction, so `ACCEPT` is the
steady state and any `REJECT` is a caught bug, not an expected event.

---

## Domain 1 — Anchor-Assisted Estimation & Full-Rank Hessian

**Goal:** with a single surveyed anchor supplying range **and** bearing factors,
show the estimator's information matrix `H` becomes full-rank — the gauge
freedom (global translation + yaw) collapses from dimension 3 (2D) to **0**,
*without* any artificial pinning or null-space projection.

### 1.1 State
Per agent `i ∈ {1..M}`, per keyframe `t`: position `p_i^t ∈ R^2` in the planar
study; the full-stack state adds velocity and IMU bias, which do not change the
gauge argument (they are observable through the IMU/VIO factors already).
Anchor pose `T_A` is a **known constant** (not estimated).

### 1.2 MAP objective
```
J(x) = Σ ‖e_odom‖²_Σodom      # relative VIO / IMU-preintegrated odometry
     + Σ ‖e_range(i,j)‖²/σ²   # inter-agent range (Huber ρ)
     + Σ ‖e_bearing(i,j)‖²/σ² # inter-agent bearing (optional)
     + Σ ‖e_anchor(i,A)‖²_ΣA  # anchor→agent range + bearing   ← the pin
```

### 1.3 Observability argument
- Without the anchor factor, `J` is invariant under global translation `t` and
  global yaw `φ` → `H·n = 0` for gauge generators `{n_tx, n_ty, n_φ}`.
- The anchor sits at a **known, externally-true, non-drifting** position.
  An anchor **range** factor makes global translation observable; an anchor
  **bearing** factor (or ranges from a second known point / over a motion
  baseline) makes global **yaw** observable.
- ⇒ the null-space generators all get non-trivial gradient support ⇒
  `H = JᵀJ ≻ 0` (strictly positive-definite, invertible).

**Both yaw channels are implemented and selectable** (see D1.2), so the study
reports the full ladder rather than a single configuration. The measured
result — `hive/nullspace.py`, singular-value gap 1.5e13, so the rank claims are
not threshold-sensitive:

| configuration | `dim ker(H)` | why |
|---|---|---|
| no anchor | **3** | full SE(2) gauge |
| 1 anchor, 1 agent, 1 keyframe | **2** | one scalar range on a 3-DoF body |
| 1 anchor, 1 agent, all keyframes | **1** | *temporal* baseline recovers translation |
| 1 anchor, all agents, 1 keyframe | **1** | *spatial* baseline does the same, instantly |
| 1 anchor, all agents, all keyframes | **1** | more range data, same kernel |
| + BODY-frame bearing (vehicle AoA) | **1** | invariant to rotation about the anchor |
| + ANCHOR-frame bearing (surveyed) | **0** | external heading → full rank |
| 2 surveyed anchors, range only | **0** | second known point → full rank |

**Two corrections the numbers forced on this plan.**

1. **A motion baseline does not rescue yaw.** Rotate the entire solution — every
   agent, every keyframe, every heading — about the anchor, and every anchor
   range, every inter-agent range and every body-frame odometry residual is
   preserved. The symmetry is exact, so the null vector is exact, and no amount
   of flying changes it. What a baseline *does* buy is translation: it takes the
   single-keyframe single-agent case from 2 free directions to 1. A **mesh**
   supplies that baseline spatially, at one instant, without moving at all.

2. **A bearing only counts in an externally-known frame.** The surveyed ground
   station is bolted down and its heading is surveyed with its position, so a
   bearing measured *at the anchor* is real yaw information. An AoA antenna on
   the *vehicle* measures the anchor in the vehicle's own drifting body frame —
   rotate the world about the anchor and that measurement is unchanged. It buys
   nothing, and the ladder shows it rather than asserting it.

**What the surviving freedom costs, in metres.** With a range-only anchor the
swarm centroid is pinned **radially to 2.7 cm** and left **tangentially free
(2006 m, prior-limited)**. Ranges tell you how far away you are; only a bearing
or a second anchor tells you which way round. That is the yaw null-space showing
up as a position error that scales with anchor-to-swarm distance.

**Contrast to reference-drone pinning:** a reference *drone* removes the
*singularity* (convention) but the cloud drifts *with* it. A surveyed anchor
removes the *drift* (real external information). That is the whole reason the
anchor comes back here.

### 1.4 iSAM2 integration
- Add anchor factors to the same incremental graph; they touch only the ranged
  agent's current pose, so Bayes-tree re-elimination stays local and cheap.
- Numerical note: anchor factor `ΣA` should reflect true survey + ranging noise;
  don't make it artificially tight or it dominates and hides VIO/range
  inconsistencies.

**Deliverables D1.x**
- D1.1 `hive/anchor_factor.py` — analytic residuals + Jacobians for anchor range
  and anchor bearing, plus the gauge generators.
- D1.2 `hive/nullspace.py` + `tests/test_nullspace.py` — the rank ladder above.
- D1.3 `hive/anchored_isam2.py` — GTSAM/iSAM2 with anchor factors; covariance
  along the former gauge directions, bounded vs unbounded.

---

## Domain 2 — Topology Partitioning & Supervisor-Certified FSM

**Goal:** split the swarm into an active sub-team and a residual loiter set,
elect the sub-team by auction, and hand over between modes without ever tripping
a supervisor gate.

### 2.1 Adjacency cut
Comms graph `A_ij(t) = 1[‖p_i − p_j‖ ≤ R_comm]`. After electing set `S`, permute
`S` first via `P`:
```
A_c(t⁺) = Pᵀ (A_sub1 ⊕ A_sub2) P    # block-diagonal control graph
L_sub  = D − A_sub                   # per-set Laplacian
```
Physical links persist across the cut; only the **control** graph is
block-diagonalized so each set coordinates independently.

### 2.2 CBBA / auction (single task, N-slot coalition)
Bid score for agent `i` on target `X_tac`:
```
c_i(X) = w_d/(1+dist(p_i,X)) + w_b·batt_i + w_s·sensor_i − w_r·κ_loiter_i
```
`κ_loiter_i` penalizes pulling an agent whose removal breaks loiter coverage.
Max-consensus over the graph: adopt neighbor's `(y_k, z_k)` if `y_k > y_i`
(tie-break by id); converges in ≤ `diam(G)` rounds to a conflict-free set.

### 2.3 FSM with invariant guards
| Transition | Guard (must hold over horizon `H`) |
|---|---|
| `LOITER_MESH→AUCTION` | task ingested; **zero setpoint change** → no gate can fire |
| `AUCTION→RECONFIG` | consensus conflict-free ∧ predicted min-clearance ≥ `d_clear` |
| `RECONFIG` | rate-limited loiter re-solve (Lloyd over `M−N`); per-tick slew ≤ `v_max`, spacing ≥ `d_clear` every tick |
| `RECONFIG→DISPATCH` | full approach stream predicted in-geofence, `v ≤ v_max`, clearance ≥ `d_clear` **before** motion |

**Deliverables D2.x**
- D2.1 `hive/cbba.py` + `tests/test_cbba.py` — convergence on random graphs.
- D2.2 `hive/supervisor_gate.py` — Python mirror of the Rust supervisor's
  invariants, cross-checked against it in `tests/test_gate_parity.py`.
- D2.3 `hive/mission_fsm.py` + a sim run: `M→N` handover with **zero**
  supervisor rejections logged.

---

## Domain 3 — Event-Triggered Dispatch & Guided Standoff Approach

**Reframed and locked:** dispatch is **reactive**, not scheduled. When the
external **go-signal** arrives, the agent that receives it first breaks off and
runs an **individual** guided approach to a **standoff station** near `X_tac`.
There is **no impact-time consensus, no simultaneity coupling, and no homing
onto the target point** — the agent converges to a standoff station on a
perimeter around `X`, at a distinct approach geometry suited to the task
(inspection view / sample point / delivery drop). This is the simplest, safest
variant: single-agent path-following triggered by an event.

> **Explicitly out of scope:** terminal-homing-to-point laws, PN + `t_go`-bias
> salvo timing, and any construction whose objective is `r_i → 0` convergence of
> multiple vehicles onto one coordinate. Not derived here.

### 3.1 Standoff station placement
Given `X_tac` and standoff radius `d_s`, place the station at the task-appropriate
bearing `θ`:
```
s_i = X_tac + d_s · [cos θ, sin θ]    # distinct viewing/approach point, not X itself
```

### 3.2 Guided approach (vector-field path following)
Fixed curvature-bounded path `p_i(γ)`, `γ ∈ [0,1]`, ending at `s_i`.
Cross-track error `e_i` to the path; heading command:
```
ψ_des = ψ_tan(γ) − arctan( k_e · e_i / v_i )
ė:  ė = v_i · sin(ψ − ψ_tan(γ))       # cross-track dynamics → 0
γ̇ = v_i / L_i(γ)                      # progress; single agent, no coupling term
```
Speed / turn-rate saturated to `[v_min, v_max]`, `[ω_min, ω_max]`. Every emitted
setpoint still passes the supervisor's geofence / clearance / velocity /
freshness gates.

### 3.3 Trigger semantics
- Dispatch is edge-triggered on go-signal receipt at the agent.
- Loiter set continues its coverage pattern unchanged.
- (Future option: if a second signal arrives, a second agent dispatches
  independently — same law, own station. Still no inter-agent timing coupling.)

**Deliverables D3.x**
- D3.1 `hive/standoff.py` — station placement + vector-field follower.
- D3.2 `tests/test_approach.py` — convergence of `e→0`, arrival at `s_i`, every
  setpoint supervisor-valid.

---

## Domain 4 — Simulation Orchestration on ASUS Zephyrus

> **Executed on the other laptop.** Everything under `sim/` is authored,
> self-checked, and committed here; the 20-vehicle SITL run itself happens on the
> Zephyrus (Linux + NVIDIA + Docker). macOS cannot run `gz-sim` headless with
> NVIDIA rendering or `tc netem` on veth pairs.

**Binding constraint: 16 GB RAM.** Architect around it; GPU/CPU have slack.

### 4.1 Physics topology
**One** headless Gazebo Harmonic (`gz-sim`) server hosting all `M` models via
PX4 multi-vehicle SITL. **Do not** launch N Gazebo instances — this is the single
biggest RAM win.

### 4.2 RAM budget (M=20, headless)
| Component | Est. |
|---|---|
| PX4 SITL ×20 (~250–400 MB ea) | 6–8 GB |
| gz-sim server (headless, sensors limited) | 2–4 GB |
| ROS 2 + DDS + our nodes | 1–2 GB |
| OS | ~2 GB |
| **Total** | **~13–15 GB** |
Add an 8 GB NVMe swapfile (`fallocate`) as margin; consider `zram`.

### 4.3 CPU (8c/16t 7900HS)
Dedicate 2 cores (cpuset) to DDS + the **native** Rust supervisor
(`nice -n -5`, not containerized — cheap and deterministic). Spread PX4 across
the rest.

### 4.4 VRAM (8 GB RTX 4060)
Cameras are the VRAM killers. Give depth/RGB sensors to the **active** drone(s)
only; loiter drones run kinematic/range-only. Render headless
(`gz sim -s -r --headless-rendering`). Isaac Sim only for small perception sweeps
(4–8 envs, reduced res) — never simultaneously with 20-vehicle Gazebo.

### 4.5 Containers & ROS 2
- `docker compose`: `gz-server`, `px4-sitl` (scaled replica), `ros-gz-bridge`,
  `hive-fsm`, `supervisor` (native), `netem`.
- All vehicles share **one** `ROS_DOMAIN_ID`, separate namespaces
  `/drone_1…/drone_20` (they must see each other for the mesh). Fence off
  monitoring/background stacks on a **different** domain ID so discovery traffic
  doesn't inflate participant tables at scale.
- Container layers, logs, rosbags on NVMe.

### 4.6 Fault-injection stress test (highest-value)
```
tc qdisc add dev veth-drone7 root netem \
   delay 50ms 15ms distribution normal loss 8% reorder 2%
```
Sweep loss 0–30%; verify the failure mode is **always** "freeze safely":
- delayed estimate → `EstimateStale` fires
- dropped plan → **zero** setpoints emitted, vehicle holds (never lunges)

**Deliverables D4.x**
- D4.1 `sim/docker-compose.yml`, `sim/launch/*.py`, `sim/config/*`.
- D4.2 `sim/netem_sweep.sh` + `sim/preflight.sh` (host capability check).
- D4.3 `hive/loss_model.py` + `tests/test_safe_hold.py` — the safe-hold property
  proven **here**, host-independently, so the Zephyrus run is a confirmation and
  not the first evidence.

---

## Milestones

| # | Milestone | Artifact | Status |
|---|---|---|---|
| M1 | anchored estimator + null-space ladder | rank ladder + covariance figures | **done** |
| M2 | auction + FSM handover in sim | zero-rejection `M→N` handover, 80/80 ACCEPT | **done** |
| M3 | event-triggered dispatch + guided approach | arrives on station, peak \|e\| 17 cm | **done** |
| M4 | 20-vehicle SITL + fault injection | safe-hold **proven here**; SITL run on the Zephyrus | artifacts authored |

Sequencing rationale: **M1 first** — it directly closes the loop with the
anchored-vs-anchorless analysis. Prove the single surveyed anchor collapses the
null-space to 0 before building anything on top of the estimate.

---

## Repo Layout
```
hive_assist/
  hive/
    frames.py            # D1.0  geodetic → TacFrame
    anchor_factor.py     # D1.1  residuals + Jacobians + gauge generators
    nullspace.py         # D1.2  the rank ladder
    anchored_isam2.py    # D1.3  GTSAM / iSAM2 integration
    cbba.py              # D2.1  auction
    supervisor_gate.py   # D2.2  Python mirror of the Rust invariants
    mission_fsm.py       # D2.3  guarded FSM
    standoff.py          # D3.1  station + vector-field follower
    loss_model.py        # D4.3  comms loss / staleness model
  tests/                 # one test module per deliverable
  sim/                   # D4: compose, launch, netem  (runs on Zephyrus)
  figures/               # generated
  docs/
```

## Open Questions / Risks

- ~~Anchor bearing channel vs. yaw from ranges over a motion baseline~~ →
  **resolved, and not the way the plan assumed.** The motion-baseline route to
  global yaw *does not exist* for a single anchor. Either survey the anchor's
  heading and get a real bearing channel, or survey a second anchor. Both are
  cheap; neither is optional. See the D1.2 ladder above.
- Standoff radius `d_s` and approach geometry per task type — still open. It is
  a per-task config table in `hive/standoff.py`, not a derivation, because the
  right numbers come from sensor working distance, rotor wash and local rules of
  engagement rather than from geometry.
- Loiter coverage re-solve latency during `RECONFIG` at large `M`.

## Findings that changed the design

Recorded here because each one was a correction to something this plan asserted,
and each is pinned by a test.

| # | finding | where |
|---|---|---|
| 1 | A single range-only anchor leaves `dim ker(H) = 1` — rotation about the anchor, exactly. Motion does not fix it. | `test_nullspace.py` |
| 2 | A vehicle-side AoA bearing to the anchor adds no yaw information. Only the anchor's own surveyed frame does. | `test_nullspace.py` |
| 3 | Reference-drone pinning does not merely drift — it becomes **overconfident**, ending 1.16 m from truth while reporting σ = 0.26 m. The supervisor's covariance gate reads that number. | `test_anchored_isam2.py` |
| 4 | Block-diagonalising the **control** graph does not block-diagonalise the **collision** constraint. Guards must check the moving set against the whole swarm. | `test_mission_fsm.py` |
| 5 | An untrusted vehicle must be excluded from the plan (the supervisor refuses to command it) but still treated as an obstacle. Those are different lists. | `test_mission_fsm.py` |
| 6 | A vector field regulates cross-track error and nothing else: without a terminal capture phase the vehicle flies past the station reporting `e = 0.000` forever. "e → 0" and "arrives" are separate claims. | `test_approach.py` |
| 7 | A Dubins path tighter than `v_cruise/ω_max` is a reference the aircraft cannot fly. Sized from the vehicle now, not chosen. | `test_approach.py` |
| 8 | **The post-blackout lunge.** Freshness bounds a plan's *age*; nothing bounds its *distance*. At 30% loss a stale stream commands a 7.2 m jump — 8× a legal tick — and passes every existing gate. Fixing it needs a `SlewTooLarge` gate **and** re-planning from the vehicle's actual position; either alone gives a lunge or a stall. | `test_safe_hold.py` |

Finding 8 is a concrete recommendation for
[`../brain/rust/swarm-supervisor/src/lib.rs`](../brain/rust/swarm-supervisor/src/lib.rs).
It is ~10 lines and it is the only gate that makes "never lunges" true rather
than merely intended.
