# Anchor-Referenced GPS-Denied Drone Swarm

**A swarm that knows where it is without GPS, decides who goes, and flies there — with every
command it issues checked by something that assumes the planner is wrong.**

Noisy inter-drone radio ranges go in. A trusted position comes out. That position becomes the
autopilot's sense of place. The aircraft flies on it. No satellite anywhere in the loop.

Two workspaces, one lineage: `brain/` is the estimator and flight stack; `hive_assist/` is the
swarm that stands on it.

---

## The system, end to end

```
                        ┌──────────────────────────────────┐
   NOT OUR DOMAIN ─────▶│   external detector / ISR feed    │
                        └────────────────┬─────────────────┘
                                         │  {lat, lon, alt?, task_id}   ← the ONLY input
                                         ▼
╔═════════════════════════════════════════════════════════════════════════════════════╗
║  hive_assist/ — ANCHOR-REFERENCED COOPERATIVE SWARM                                 ║
╠═════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                     ║
║   D1  FRAME          WGS84 → ECEF → TacFrame (local ENU)                            ║
║       frames.py      origin = GPS-SURVEYED ANCHOR, yaw fixed by survey               ║
║                      the one geodesy boundary in the whole system                   ║
║                                 │                                                   ║
║                                 ▼                                                   ║
║   D1  ESTIMATOR      ┌────────────────────────────────────────────────┐             ║
║       anchored_      │  iSAM2 factor graph, incremental, fixed-lag     │◀── ranges   ║
║       isam2.py       │   ├ inter-agent range   (Huber ρ)               │             ║
║                      │   ├ ANCHOR range/bearing  ← kills gauge freedom │             ║
║                      │   └ constant-velocity motion prior              │             ║
║                      │  out: position + marginal covariance PER VEHICLE│             ║
║                      └────────────────────────┬───────────────────────┘             ║
║                                               │  EstimateSnapshot{pos, cov_trace}   ║
║                    ┌──────────────────────────┴────────────────────────┐            ║
║                    ▼                                                   ▼            ║
║   D2/D3  ORCHESTRATOR                                        EXTERNAL-NAV FAN-OUT   ║
║   ┌──────────────────────────────────────────┐               one continuous stream  ║
║   │ TASK_INGEST   target → TacFrame          │               per vehicle            ║
║   │ AUCTION       CBBA: bids on distance,    │                      │               ║
║   │               battery, sensor, coverage  │                      │               ║
║   │               → coalition of N, ≤diam(G) │                      │               ║
║   │ RECONFIG      remainder re-solves loiter │                      │               ║
║   │ DISPATCH      Dubins + vector-field      │                      │               ║
║   │               approach to a STANDOFF     │                      │               ║
║   │               STATION, never the target  │                      │               ║
║   └────────────────────┬─────────────────────┘                      │               ║
║                        │  Plan{plan_id, issued_ms, min_spacing,      │               ║
║                        │       assignments[{vehicle, waypoint}]}     │               ║
║                        ▼                                            │               ║
║   ╔════════════════════════════════════════╗                        │               ║
║   ║  RUST SUPERVISOR — the only way out    ║  every transition      │               ║
║   ║  geofence · spacing · covariance ·     ║  forward-simulates its │               ║
║   ║  freshness · slew rate                 ║  WHOLE stream over     │               ║
║   ║  REJECT ⇒ zero MAVLink packets emitted ║  horizon H before      │               ║
║   ╚════════════════════┬═══════════════════╝  committing            │               ║
║                        │ ACCEPT                                     │               ║
╚════════════════════════╪════════════════════════════════════════════╪═══════════════╝
                         │ SET_POSITION_TARGET_LOCAL_NED              │ VISION_POSITION_
                         │ per vehicle, per tick                      │ ESTIMATE, 20 Hz
      ┌──────────────────┴────────────────────────────────────────────┴───────────────┐
      │  MAVLink 2  ·  per-vehicle signing keys  ·  unsigned frames REFUSED           │
      └──────────────────┬────────────────────────────────────────────┬───────────────┘
                         ▼                                            ▼
   ┌────────────────────────────────────────────────────────────────────────────────┐
   │   N × ArduCopter  (headless SITL today, real airframes tomorrow)               │
   │   GPS_TYPE = 0  ·  EKF3 source set → external nav  ·  no compass               │
   │   EKF3 fuses OUR estimate as the vehicle's sole horizontal position            │
   │        │                                                                       │
   │        ▼   attitude controller → motors → the drone actually moves             │
   └────────┬───────────────────────────────────────────────────────────────────────┘
            │ true position
            ▼
   ┌─────────────────────────────┐       ┌───────────────────────────────┐
   │  UWB RANGE MESH             │◀──────│  4 × SURVEYED ANCHOR           │
   │  drone↔drone within 30 m    │       │  masts at the corners of a    │
   │  σ 1.5 cm · 15% NLOS (one-  │       │  50 m square, ranging to 120 m│
   │  sided) · 3% multipath      │       │  FOUR, not one — and the      │
   │  spike · 10% dropout/tick   │       │  reason why is finding #3     │
   └────────┬────────────────────┘       └───────────────────────────────┘
            │
            └──────────────▶ back into the ESTIMATOR.  THE LOOP IS CLOSED.
                             estimate → EKF3 → motors → true position → ranges → estimate
                             Only the radio is simulated, exactly as SITL already fakes GPS.
```

### The complete sensor list — this is the flex

```
  WHAT IT FLIES ON                         WHAT IT DOES NOT HAVE
  ────────────────────────────────         ─────────────────────────────
  UWB radio ranging     σ 1.5 cm           ✗  GPS          (GPS_TYPE = 0)
  4 × surveyed anchor   known points       ✗  magnetometer / compass
  the autopilot's IMU   via EKF3, not us   ✗  camera        (deleted, not deferred)
                                            ✗  LiDAR, radar, optical flow
                                            ✗  GPU          (runs on an 8 GB M1)
```

**One radio and four surveyed points on the ground.** That is the entire perception system. No
camera in the live loop at all — the vision/VIO front end was **deleted** in the 2026-08-08
restructure, not shelved, because the estimator core was always range + anchor + a motion prior
and the camera was a second stack layered on top that dragged in a GPU, a renderer, and every red
row of the old forensic log. Attitude, gravity and IMU fusion are delegated to the autopilot's
own EKF3, which already does that job well. What is left is geometry, and geometry is cheap.

The UWB noise model is not a Gaussian toy: NLOS bias is **one-sided** (an obstructed path is
always *longer*, never shorter, so the error mean is positive by construction), multipath adds
0.2–0.5 m spikes, and dropout is per-link per-tick — so the graph's connectivity changes every
tick. That is the condition the Huber kernels and the motion prior exist to survive.

### What `hive_assist/` inherited from `brain/`

| Brain asset | How the swarm uses it |
|---|---|
| iSAM2 estimator (`day8_isam2.py`) | the graph the anchor factors attach to; same `CustomFactor` idiom |
| `measure()` + day-7 UWB noise model | becomes `sim/range_world.py` — the live radio, unchanged noise |
| Rust `swarm-supervisor` | still authoritative and **unmodified**; the Python mirror is diffed against the real binary on 120 randomised cases |
| `PLAN_SCHEMA` | the wire shape the mission FSM and orchestrator emit |
| MAVLink `close_the_loop.py` | successor is `sim/orchestrator.py`: live ranges instead of a replayed trajectory |
| Layer-2 ONNX perception (Rust) | the bearing front end; range+bearing rigidity result feeds the anchor observability proof |

Reuse first, extend second — nothing in `brain/` was edited to make the swarm work.

---

## Architecture — C4 view

### Level 1 — System context

Who talks to this system, and what is deliberately *outside* it.

```
   ┌────────────────┐                              ┌──────────────────────────┐
   │  OPERATOR       │                              │  TARGET DETECTOR / ISR   │
   │  [person]       │                              │  [external system]       │
   │  watches, issues│                              │  camera, acoustic array, │
   │  the go-signal  │                              │  radar, or a human       │
   └───────┬─────────┘                              └────────────┬─────────────┘
           │ read-only telemetry                                 │
           │ (serial1, holds no key)                {lat, lon, alt?, task_id}
           │                                                     │
           ▼                                                     ▼
   ╔═══════════════════════════════════════════════════════════════════════════╗
   ║           ANCHORED COOPERATIVE SWARM CONTROL SYSTEM                       ║
   ║           [software system — this repo]                                   ║
   ║                                                                           ║
   ║   Localizes N drones with no GPS, elects a coalition, and flies them      ║
   ║   to a standoff perimeter — emitting only setpoints a safety gate         ║
   ║   has already certified.                                                  ║
   ╚═══╤═══════════════════════════════════════════════════════╤═══════════════╝
       │ signed MAVLink 2                                      │ ranges in
       │ VISION_POSITION_ESTIMATE ↓                            │
       │ SET_POSITION_TARGET_LOCAL_NED ↓                       │
       ▼                                                       ▼
   ┌────────────────────────────────┐              ┌──────────────────────────┐
   │  ARDUPILOT / ArduCopter 4.5.7  │              │  UWB RANGE MESH +        │
   │  [external system]             │              │  4 SURVEYED ANCHORS      │
   │  EKF3, attitude, motors        │              │  [external hardware]     │
   │  WE DO NOT WRITE THIS          │              │  σ 1.5 cm, one-sided     │
   │  — integration via its own     │              │  NLOS, 10% dropout       │
   │    documented external-nav API │              └──────────────────────────┘
   └────────────────────────────────┘

   OUT OF SCOPE, STATED: target detection · attitude estimation · motor control
```

### Level 2 — Containers

Each box is a separately deployable process. Note that the supervisor is a
**native Rust binary** and the config file is read by *both* sides.

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║  ANCHORED COOPERATIVE SWARM CONTROL SYSTEM                                    ║
║                                                                               ║
║  ┌─────────────────────────┐         ┌──────────────────────────────────┐    ║
║  │ ground_truth_bridge     │         │ SECURITY LAYER                    │    ║
║  │ [Python · pymavlink]    │────────▶│ [Python · stdlib crypto only]     │    ║
║  │ one socket, one reader, │  binds  │ keystore (scrypt + EtM)           │    ║
║  │ routes to mailboxes     │  keys to│ enable_signing (SETUP_SIGNING,    │    ║
║  │ true position per veh   │  a link │   reject-unsigned, replay guard)  │    ║
║  └───────────┬─────────────┘         └──────────────────────────────────┘    ║
║              │ truth[i]                                                      ║
║              ▼                                                               ║
║  ┌─────────────────────────┐                                                 ║
║  │ range_world             │  ← the ONLY simulated component.                ║
║  │ [Python · numpy]        │    Swap for a real UWB driver = hardware        ║
║  │ pairwise + anchor ranges│    step, not an architecture change.            ║
║  └───────────┬─────────────┘                                                 ║
║              │ RangeFrame                                                    ║
║              ▼                                                               ║
║  ┌─────────────────────────┐         ┌──────────────────────────────────┐    ║
║  │ live_estimator          │────────▶│ external_nav_fanout               │    ║
║  │ [Python · GTSAM/iSAM2]  │ Estimate│ [Python · dedicated thread]       │    ║
║  │ fixed-lag factor graph  │ Snapshot│ 20 Hz, MUST NEVER GAP             │    ║
║  │ pos + cov_trace per veh │         │ + SET_GPS_GLOBAL_ORIGIN once      │    ║
║  └───────────┬─────────────┘         └──────────────┬───────────────────┘    ║
║              │                                       │ VISION_POSITION_ESTIMATE
║              ▼                                       │                       ║
║  ┌─────────────────────────┐                        │                       ║
║  │ orchestrator            │                        │                       ║
║  │ [Python]                │                        │                       ║
║  │ mission FSM · CBBA ·    │                        │                       ║
║  │ standoff/Dubins → Plan  │                        │                       ║
║  └───────────┬─────────────┘                        │                       ║
║              │ Plan (JSON, PLAN_SCHEMA)              │                       ║
║              ▼                                       │                       ║
║  ┌─────────────────────────┐   ┌──────────────────┐ │                       ║
║  │ swarm-supervisor        │◀──│ supervisor.json  │ │                       ║
║  │ [Rust · native binary]  │   │ [config]         │ │                       ║
║  │ THE ONLY WAY OUT        │   │ READ BY BOTH     │ │                       ║
║  │ geofence·spacing·cov·   │   │ SIDES — two      │ │                       ║
║  │ freshness·slew          │   │ copies that      │ │                       ║
║  │ ACCEPT ⇒ emit           │   │ drift produce an │ │                       ║
║  │ REJECT ⇒ zero packets   │   │ inexplicable     │ │                       ║
║  └───────────┬─────────────┘   │ REJECT           │ │                       ║
║              │                  └──────────────────┘ │                       ║
╚══════════════╪═══════════════════════════════════════╪═══════════════════════╝
               │ SET_POSITION_TARGET_LOCAL_NED         │
               ▼                                       ▼
        ═══════════════ signed MAVLink 2, serial2 ═══════════════
                              │
                              ▼
                    N × ArduCopter (EKF3)
```

### Level 3 — Components inside two containers

**`live_estimator` — the factor graph**

```
  ┌──────────────────────────────────────────────────────────────────┐
  │  live_estimator                                                  │
  │                                                                  │
  │   frames.py ──────────▶ TacFrame      (geodetic→ENU, ONE place)  │
  │                            │                                     │
  │   anchor_factor.py ────────┼──▶ residuals + ANALYTIC Jacobians   │
  │                            │    + gauge generators               │
  │                            ▼                                     │
  │   ┌────────────────────────────────────────────────┐             │
  │   │  iSAM2  (anchored_isam2.py)                    │             │
  │   │   ├ inter-agent range      Huber ρ             │             │
  │   │   ├ anchor range × 4       ← removes DRIFT     │             │
  │   │   └ constant-velocity prior ← keeps under-     │             │
  │   │       constrained frames non-singular          │             │
  │   └────────────────────┬───────────────────────────┘             │
  │                        │                                         │
  │   nullspace.py ────────┴──▶ dim ker(H) — the rank ladder;        │
  │                             why there are 4 anchors, not 1       │
  └──────────────────────────────────────────────────────────────────┘
```

**`security` — the MAVLink authentication layer**

```
  ┌──────────────────────────────────────────────────────────────────┐
  │  security/                                                       │
  │                                                                  │
  │   keystore.py                                                    │
  │    ├ secrets.token_bytes(32)      one key PER VEHICLE            │
  │    ├ scrypt(N=2^15) → 64 B → (k_enc, k_mac)   never the same key │
  │    ├ HMAC-CTR keystream XOR, encrypt-then-MAC OVER THE HEADER    │
  │    │    ↳ so KDF params can't be downgraded to N=2 and reopened  │
  │    └ NO default passphrase (open() raises)                       │
  │                          │ key(i)                                │
  │                          ▼                                       │
  │   enable_signing.py                                              │
  │    ├ provision_vehicle()   SETUP_SIGNING ×3 (no ACK + UDP),      │
  │    │                       key OUT first, our side switches      │
  │    │                       after; refuses while ARMED            │
  │    ├ allow_unsigned_cb()   reject-unsigned; allowlist is exactly │
  │    │                       {RADIO_STATUS 109, RADIO 166} —       │
  │    │                       SiK radios inject those, hold no key  │
  │    ├ _strict_check_sig()   FINDING #2: verify HMAC *before*      │
  │    │                       touching stream state                 │
  │    ├ probe_unsigned_...()  ACTIVE PROBE: send an unsigned cmd,   │
  │    │                       an ACK means we are NOT hardened      │
  │    └ RejectionLog          robust_parsing hides rejects as       │
  │                            BAD_DATA — so count them explicitly   │
  └──────────────────────────────────────────────────────────────────┘
```

---

## What it does — six points

**1. It replaces GPS with geometry, and a real autopilot flies on the result.**
Every rung of the estimator ladder is built from first principles rather than imported: convex
**Biswas-Ye SDP** relaxation → **distributed ADMM** with neighbour-only comms → robust outlier
certification → **GTSAM** batch smoothing → **iSAM2** incremental, with bounded per-frame cost
regardless of mission length. Days 1–6 run on clean Gaussian noise to build the intuition; days
7–8 swap in the real UWB model and real Crazyflie flight data and the estimator code barely
changes — only the world around it gets harder. The one that survives that swap is the one handed
to the autopilot: streamed into ArduCopter as `VISION_POSITION_ESTIMATE` with `GPS_TYPE=0`, EKF3
fusing it as the vehicle's only horizontal position source. Armed, flew all four legs of a
commanded 5 m square, landed, disarmed — GPS-denied and compass-free throughout.

**2. Its acceptance metric is honesty, not accuracy — and it holds in motion.**
`error / σ ≈ 1` is the bar, not a small error and not a small σ. Live, in flight: **0.026 m
error against a reported 0.017 m sigma — ratio 1.44.** The reason that number is the headline is
the project's own thesis: a confidently wrong estimate is worse than no estimate, because the
safety gate reads that σ and certifies it. An earlier build was calibrated at rest and **147×
overconfident** the moment it moved; that whole failure class was root-caused and designed out.

**3. THE headline finding: one anchor is not enough, and the measurement said so before the
flight did.**
The design claimed a single surveyed anchor collapses the SE(2) gauge freedom from 3 DOF to 0.
Measured against the actual information matrix, with ranges alone it does **not**. Rotate the
entire solution about the anchor and every anchor range, every inter-agent range and every
odometry residual is *exactly* preserved — the symmetry is perfect, so `dim ker(H) = 1` survives
however long the swarm flies. In metres that free direction costs: range-only pins the swarm
centroid **radially to 2.7 cm and leaves it tangentially free at 2006 m.** You know precisely how
far from the anchor you are, and nothing at all about which way round.

The rank ladder, measured (singular-value gap 1.5e13 — not a thresholding artefact):

| configuration | `dim ker(H)` |
|---|---|
| no anchor | **3** — full SE(2) gauge |
| 1 anchor, range only, all agents, all keyframes | **1** — rotation about the anchor, forever |
| + bearing in the **vehicle's** frame | **1** — buys *nothing* |
| + bearing in the **anchor's surveyed** frame | **0** — full rank |
| **2+ surveyed anchors, range only** | **0** — full rank |

Two things make this a finding rather than a plot. First, it **contradicted the project's own
plan** and produced a definite answer to what the plan had left open: survey the anchor's
*heading*, or survey a *second* anchor — the motion-baseline route people reach for does not
exist. Second, and this is the part worth saying out loud in an interview: **the result changed
the hardware layout.** The deployed site is four anchor masts at the corners of a 50 m square,
and `sim/range_world.py` says why in a comment — four independent radial constraints in different
directions, which is the "2 surveyed anchors → dim ker = 0" row with margin to spare, and it is
also what a real site actually looks like. An observability proof walked directly into where the
masts get planted.

**4. N drones take a coordinate and go — auction, approach, standoff perimeter.**
One message `{lat, lon}` in; gated per-vehicle setpoints out. A CBBA auction elects the coalition
live (bids weigh distance, battery, sensor fit, and a penalty for breaking loiter coverage),
converging conflict-free in ≤ `diam(G)` rounds. Each elected agent flies its own
curvature-bounded Dubins path to its own slot on a **standoff perimeter around the target, never
onto it**. Flown in SITL with 4 vehicles: closest any airframe came to the target was **14.65 m
against a 12 m standoff** — the perimeter is respected by every member, verified against
simulator ground truth, not against the estimate that could be wrong.

**5. Nothing reaches the motors except through a gate that assumes the planner is wrong.**
Every mode transition **forward-simulates its entire setpoint stream over horizon H** and commits
only if the stream already satisfies geofence, spacing, covariance and freshness. So `ACCEPT` is
the steady state by construction and a `REJECT` is a caught bug, not an event: **130 plans, 130
ACCEPT, 0 guard failures.** A rejected plan produces **zero** MAVLink packets. The same gate is
what sits under a language-conditioned mission planner (constrained decoding, no free-text
channel to the aircraft) — the LLM proposes, ~400 lines of deterministic Rust with no model in it
disposes.

**6. It is engineered like flight software, not a notebook.**
**261 tests.** Latency measured against a hard budget end to end: range-in → external-nav on the
wire **p50 1.8 ms / p99 8.4 ms**, plan → gate → setpoint **p50 11.2 / p99 17.6 ms**, against a
25 ms loop. A Rust transport layer (`tokio`, bounded `watch` channels) holds **p99 336 µs at 12
vehicles vs 1,988 µs for the identical Python architecture — ~6×**, because the GIL makes
Python's median grow 4.8× with fleet size while Rust stays flat. A Rust ONNX perception path
(recall 94%, precision 96%, bearing RMS 0.15°) takes pixels to factor-graph bearings, and those
bearings constrain exactly the DOF ranges leave ambiguous: at the marginal connectivity radius,
~16 detections/frame take a **33%-rigid graph to 100% rigid, RMSE 0.26 m → 0.05 m**. Whole thing
runs on an 8 GB M1 — no GPU in the loop.

---

## Why the swarm is hard to disturb — three points on comms

Written against a stated threat model (assets ranked A1–A6, adversaries tiered 0–3) that names
what is **out of scope** — a resourced adversary can always deny an RF channel; that is physics,
not an engineering gap. So the deliverable is cost, detection, and safe failure, not a promise.

**1. There is no GPS to spoof — the cheapest attack on a conventional drone simply has no
purchase.** GPS spoofing is a commodity Tier-1/2 attack, and against this design it does nothing:
there is no GPS fix in the control loop to poison. That is a security property fallen out of the
architecture for free. The honest corollary is stated rather than hidden — the trust GPS would
have carried now sits on the anchor and the range mesh, which is exactly why those are
Huber-gated, covariance-monitored, and flagged for 802.15.4z secure ranging on real hardware.

**2. The links are authenticated, and the proof is a refusal, not a passing test.** MAVLink 2
signing with **one key per vehicle, never a fleet secret** (recovering one crashed airframe must
not hand over the other three), generated from the OS CSPRNG and held in an encrypt-then-MAC,
scrypt-derived keystore that has no default passphrase. The evidence is the same probe against
the same vehicle before and after: an unsigned command **ACKed and executed** on the old
transport, **refused** on the new one. Getting there meant reading ArduPilot's source and finding
that `GCS_Signing.cpp` accepts everything on `MAVLINK_COMM_0` via a compiled-in callback — the
loop had been sitting on the one channel where signing can do nothing, so enabling it there would
have produced healthy counters and zero security. Cost against the flight budget, measured
signed-vs-unsigned back to back: **p50 4.2 ms unchanged, p99 16.4 → 18.4 ms** against 25 ms.
Signing is not the constraint.

**3. Authentication is not enough, and the gate that follows it is the interesting one.**
Signing proves *who sent* a frame. It says nothing about whether the payload can be true —
and this project's own history is the proof that the estimate can be confidently wrong with
no attacker present at all. So two inbound gates sit on the data path, and the sharpest one
is a direct executable statement of the project's thesis. It takes the vehicle's own last two
accepted positions, predicts the current one under constant velocity, and compares the
innovation against the uncertainty the estimator *claims*: a vehicle reporting **σ = 1.7 cm
while jumping half a metre from where its own motion says it should be** is making two claims
that cannot both hold. Which one is false does not matter — an attacker moved it, or the
estimator broke — because the safe response is identical. It needs no ground truth, so it
survives on real hardware. Upstream of that, the range mesh is filtered **asymmetrically**,
which is the domain insight that makes it work: the physical noise model is one-sided (an
obstructed path is *longer*; radio does not take shortcuts), so a range reading 40 cm short
has no benign explanation, while 40 cm long is an ordinary multipath spike. Distance
*reduction* is also the dangerous direction, because it drags the estimate toward the
attacker. A symmetric filter cannot express that and gets it exactly backwards.

**4. When the link degrades, the swarm freezes safely — and the dangerous half is the
*recovery*.** Under simulated loss, the outage itself is easy: nothing is emitted, the vehicle
holds. The failure nobody expects is what lands when the link returns — the planner kept
advancing while the vehicle was frozen, so the first plan through is **7.2 m away, 8× a legal
tick, and it passes every gate**, because freshness bounds a plan's *age* and nothing bounds its
*distance*. That produced a concrete new invariant (`SlewTooLarge`) and the finding that neither
half fixes it alone: the gate alone is safe but stalls the mission to 9% progress at 30% loss;
re-planning alone still lunges. Together: **0.90 m worst jump and 100% mission progress at every
loss level.**

Every refusal above is written to an **HMAC-chained, append-only audit log**, because a
rejection is the only evidence an attack was attempted — everything downstream carries on
working exactly as if nothing had happened. The chain is *keyed* rather than a plain hash,
and that distinction is the whole point: an attacker who can rewrite the file can recompute
every SHA-256 in an unkeyed chain and it verifies perfectly. Forging a keyed one requires the
key. The limit is stated rather than glossed — **truncation of the tail is undetectable from
the file alone**, which is why `verify()` accepts an independently-held expected length and
why the real answer is off-box replication.

---

## Mapping to the role (NewSpace — ML/CV intern, Bangalore)

The posting's mandatory list is *SfM, object detection, object tracking, state estimation, ROS*,
with edge deployment and multimodal transformers preferred. Direct hits:

| JD requirement | Where it lives in this project |
|---|---|
| **State estimation** | the entire spine — SDP → ADMM → GTSAM → iSAM2, with covariance calibration as the acceptance test |
| **SfM** | range+bearing rigidity, gauge/null-space analysis, factor-graph bundle adjustment — the same mathematics, applied in flight |
| **Object detection & tracking** | ONNX detector at recall 94% / precision 96%; detections associated across frames into bearing factors |
| **ONNX / edge inference** | `swarm-perception`: Rust + `ort` runtime, no GPU, runs on 8 GB of unified memory |
| **ROS 2** | `rclrs` nodes wrapping the same Rust crates the CLIs and tests use — not a forked Python demo |
| **Sensors (IMU, stereo, radio)** | IMU/attitude fusion deliberately delegated to EKF3; UWB modelled with NLOS, multipath and dropout |
| **Python / C++ / systems** | Python estimator, Rust transport + supervisor, ArduPilot C++ read as ground truth when docs disagreed |
| **Multimodal / VLM** | language-conditioned mission planner under constrained decoding, behind a deterministic safety gate |

---

## What this is not

Stated plainly, because the failure transparency is the point.

- **Simulation only.** Nothing has flown on a real airframe. SITL is a real autopilot with real
  EKF3 and real MAVLink, but it is not hardware.
- **Detection is not our domain.** The input contract begins at a coordinate someone else found.
- **Multi-vehicle formation stages (S1–S3) are wired but unrun** — `run_fleet.sh -n 4` launches
  them and no code path is single-vehicle-specific, but no numbers are claimed for what has not
  been measured.
- **Comms hardening: H0, H1, H2.2, H3.2 and H5.1 are built and tested; the rest is
  hardware or deployment.** Signing, the estimate plausibility gate, range-mesh integrity,
  verified key rotation and the tamper-evident audit chain are implemented behind `--sign`
  and `--guard`. Deliberately *not* built, with reasons: the confidentiality tunnel (WireGuard
  between two processes on one host would be theatre), FHSS / band diversity / UWB secure
  ranging (radio hardware — no software adds spread-spectrum to a fixed-frequency link), a
  secure element for keys (hardware), the fail-safe ladder wired to ArduPilot failsafe
  parameters, and the regulatory sign-off, which must be read from the current WPC/TEC
  allocations rather than from memory.
- **No RF, no radio, no jamming test.** Everything above is measured on loopback. The loss
  study runs offline in `hive/loss_model.py`; it has never been run against the live loop.
- **16 bugs are documented in full** in `hive_assist/docs/BUGS.md`, of which only two were
  ordinary broken code — the rest were wrong beliefs, lying measurements, or gaps between two
  components that each assumed the other was handling it.

---

## Figures worth showing

| File | What it shows |
|---|---|
| `brain/figures/swarm_real_flight.gif` | the swarm actually moving — lead image |
| `hive_assist/figures/d4_live_standoff.png` | 4 SITL airframes, CBBA-elected, converging on a 12 m standoff perimeter around a target |
| `hive_assist/figures/d1_nullspace_study.png` | the gauge-freedom ladder and what the free direction costs in metres |
| `brain/figures/layer1_latency.png` | Rust vs Python estimate-to-wire latency CDF at N=12 |
| `hive_assist/figures/d4_safe_hold.png` | the recovery lunge, and the gate that bounds it |
