# RUN — hive_assist

> All commands run **from `hive_assist/`** with the repo venv active.

The offline half (Domains 1–3, and Domain 4's estimator and radio) needs only
Python. The live half (S0–S3) additionally needs ArduPilot SITL. There is no
Docker, no Gazebo, no PX4 and no GPU anywhere in this stack — that is what the
2026-08-08 restructure bought.

## Setup — macOS / Linux (native)

```bash
cd hive_assist
source ../.venv/bin/activate          # or: python3 -m venv ../.venv && pip install -r ../brain/requirements.txt
pip install pytest scipy pymavlink
```

`gtsam`, `numpy`, `scipy` and `matplotlib` come from the Brain's
`requirements.txt`. `pymavlink` is the only addition Domain 4 needs.
`networkx` is deliberately **not** used — the graph code is numpy-only.

### The Rust supervisor

Required by the gate-parity tests, by the latency harness's setpoint path, and
by every live stage. Without it the parity tests skip and the rest still runs.

```bash
cargo build --release --manifest-path ../brain/rust/Cargo.toml
```

Build it on the machine you are running on. A binary built for another platform
fails with `OSError: [Errno 8] Exec format error`, which reads like a corrupt
file rather than a wrong architecture.

### ArduPilot SITL — for S0 and up

```bash
git clone --recursive https://github.com/ArduPilot/ardupilot ~/ardupilot
cd ~/ardupilot
./waf configure --board sitl && ./waf copter
```

Verified against **Copter-4.5.7**. Point `ARDUPILOT_HOME` elsewhere if it is not
in `~/ardupilot`. Nothing else is needed — `run_fleet.sh` launches the
`arducopter` binary directly, with no `sim_vehicle.py` and no MAVProxy in the
critical path.

## Everything at once

```bash
make            # tests, then every figure
make test       # 261 tests, ~40 s
make figures    # regenerate figures/*.png and print every result table
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
python -m pytest tests/test_gate_parity.py -v
```

## Domain 3 — event-triggered dispatch

```bash
python -m hive.standoff            # all four task geometries + figure
```

## Domain 4 — offline

Everything here runs without SITL, because the estimator and the radio never
open a socket.

```bash
python -m hive.loss_model          # the 0-30% loss sweep + figure
python -m sim.range_world          # the UWB model: yield, one-sided NLOS, outliers
python -m sim.live_estimator       # 4 static vehicles, cold start, err/sigma
make latency                       # D4.7: p99 vs the 25 ms budget, exits non-zero on FAIL
```

`make latency` measures the real path — real iSAM2, real MAVLink encode, real
UDP send, real Rust gate subprocess. Only the SITL scheduling noise is absent.

## Domain 4 — live

### S0 — one vehicle, range-only closed loop

The de-risk first. Converge the estimator at rest and stop **before** arming:

```bash
./sim/run_fleet.sh -n 1 --no-fly
```

Read the `at rest` line. Error and sigma should agree to within a factor of a
few; a *small* sigma on its own is not the pass condition and never was. Then
fly the commanded square:

```bash
make s0                            # or: ./sim/run_fleet.sh -n 1 -m square -s 60
```

Measured on an 8 GB M1, Copter-4.5.7:

```
estimator error 0.026 m / sigma 0.017 m -> ERROR/SIGMA 1.44
altitude 1.86-2.34 m AGL, 4/4 legs flown, landed and disarmed
range-in -> extnav on the wire: p50 1.8 ms, p99 8.4 ms   (budget 25 ms)
130 plans, 130 ACCEPT, 0 REJECT, 0 guard failures
```

### S1 / S2 — fleet

```bash
make s1                            # 4-vehicle formation hold
make s2                            # coordinated translation, inside the footprint
```

`translate` refuses a goal outside the anchor footprint rather than flying it.
Bulk translation is observable only through *changing anchor ranges*; past
`r_anchor` the swarm's absolute position is pinned by nothing while the estimate
stays confident, which is the exact failure this project exists to refuse. That
case is S3's flying anchor, and it is scoped separately.

### S3 — the standoff dispatch, live

Domain 3's figure, flown. The CBBA auction elects a coalition, each elected
agent flies its own curvature-bounded approach to its own slot on the standoff
perimeter and holds it, and the rest keep the loiter mesh.

```bash
make s3                            # or: ./sim/run_fleet.sh -n 4 -m standoff -s 120 --gcs
```

Measured on an 8 GB M1, Copter-4.5.7:

```
AUCTION: coalition [0, 3] elected in 1 round, converged
v0 -> station (13.01, 8.52)   12.0 m from target, Dubins LSR 12.0 m
v3 -> station (16.15, -0.11)  12.0 m from target, Dubins RSL 18.5 m

closest either airframe came to the target   11.98 m   (standoff 12.0 m)
final distance to station                    0.20 m    against SIMSTATE truth
separation between the two                   8.04 m    (d_clear 2.0)
81 plans, 81 ACCEPT, 0 REJECT, 0 guard failures
```

The number that matters is **11.98 m against a 12 m standoff**: the coalition
converges to the perimeter and never inside it. It writes
`sim/logs/standoff.json` and `figures/d4_live_standoff.png` from the flown
track, so the claim is re-checkable rather than asserted.

`--gcs` additionally uploads a three-item marker mission — the target and both
stations — so the geometry is visible on the QGroundControl map while it flies.
Nothing switches to AUTO; those waypoints exist to be looked at.

**Coalitions of three or more, and the two stalls they exposed.** Both showed up
only above two movers, and both presented identically: the drones stop part way
to the perimeter and the run still reports `done`.

- *The coalition crossed its own paths.* Stations were paired to agents by
  index, so whenever the coalition's angular order differed from the
  perimeter's, two approaches crossed — and because `walk_waypoints` gives every
  path the same rung count, both movers reached the crossing on the SAME rung.
  Measured 0.54 m apart against a 2.0 m floor. Stations are now assigned by
  minimum total transit (`assign_stations`), which cannot contain a crossing
  pair, and the finished ladder is checked before launch.
- *A holder's station-keeping froze the movers.* `step_toward` advanced only
  when EVERY vehicle was within `arrive_tol_m`, holders included — but a holder
  is never exactly on its slot, it is station-keeping. One drifting holder
  stopped the whole coalition. Arrival is now judged on the movers; every
  vehicle is still planned, guarded and commanded each tick.

If a ladder ever stalls again it now says so on the console within 10 s, and
says whether the guard refused or the vehicles simply are not keeping up.

**The auction needed its own weights at this scale.** Above four vehicles the
elected coalition was close to random, and a coalition containing a far-side
agent has to cross the loiter mesh to reach its station — which the check above
then (correctly) refuses. The cause was in the bid, not the geometry: across
targets 20.6 to 32.0 m away, `w_d / (1 + dist)` spans only 0.046 to 0.030 while
battery and sensor add a constant 0.65, so total bid spread was **2% against a
55% spread in distance** and the position-sensitive coverage term cast the
deciding vote. Measured, 5 cm of estimator noise elected **54 distinct
coalitions out of 56 possible** at eight vehicles.

`sim/orchestrator.py` now passes `LIVE_BID_WEIGHTS` (`w_d=25, w_r=0.05`), which
restores what `BidWeights` says it means — "distance dominates by design" — at
this scenario's scale. `hive/cbba.py` is unchanged: its defaults back the
published Domain 2 result, and weights are a parameter precisely so a different
scenario can carry different ones. Refusals over 200 runs at 0.3 m noise:

| | before | after |
|---|---|---|
| n=8, 3 slots | 167/200 | **12/200** |
| n=12, 4 slots | 193/200 | **5/200** |
| n=6, 3 slots | 128/200 | 34/200 |

**Six vehicles remains the weak case** — worse than eight or twelve. It is the
size where the ring is dense enough to obstruct a crossing mover but not sparse
enough for the auction to have an obviously-nearest pair. Closing that last gap
needs transit deconfliction (lanes or staggered departure), not weights, and is
not built.

**Six vehicles saturate an 8 GB M1.** At `-n 6` the extnav stream goes `GAPPED`
(175 ms worst) and `range-in -> extnav-out` p99 reaches ~65 ms against a 25 ms
budget. The dispatch completes and the geometry holds, but those two numbers are
failing and the cause is host contention, not the loop — `make latency` still
passes offline at 4 vehicles. Treat `-n 6` as a demonstration, not a measurement.

**A target too close to the mesh is refused, not flown.** With the D3 figure's
own `(16, 3)`, vehicle 0 starts 10.4 m from the target — already inside the 12 m
perimeter it is supposed to stop at, and no approach out of a circle you begin
within can avoid entering it. `dispatch_coalition` raises rather than flying it.
The default target is `(25, 8)`, far enough out that the whole mesh starts
outside. Override with `--target X Y`, `--task`, `--slots`, `--spread`.

### Driving it yourself

```bash
./sim/run_fleet.sh -n 4 --sitl-only          # fleet up, endpoints printed
python -m sim.ground_truth_bridge -n 4 -s 10 # verify the truth source per vehicle
python -m sim.orchestrator -n 4 -m hold -s 60
```

`ground_truth_bridge` prints which message each vehicle's truth came from.
It must say **SIMSTATE**. `SIM_STATE` is a valid fallback (its `lat_int`
extension is equally exact); anything else means the D4.1 contract broke.

### Signed links — COMMS_HARDENING_PLAN.md stage H0

Off by default. `--sign` gives every vehicle its own MAVLink 2 signing key and
makes the autopilot refuse unsigned commands.

```bash
export HIVE_KEYSTORE_PASSPHRASE='...'          # no default; it will refuse without one
./sim/run_fleet.sh -n 4 -m hold -s 60 --sign   # forwarded straight through
python -m sim.orchestrator -n 4 -m hold -s 60 --sign   # against a --sitl-only fleet
```

Keys live encrypted in `~/.hive/keys/keystore.json` (`$HIVE_KEYSTORE` or
`--keystore` to move it). They are generated once and reused; the vehicles keep
copies in FRAM, so **use the same passphrase every time** — a lost keystore
means re-launching the fleet (`run_fleet.sh` passes `--wipe`) to clear the key
the airframes are still holding.

Step 1 then does three things: pushes the key, checks the vehicle actually
refuses an unsigned command, and stops the run if it does not. That check is a
hard stop rather than a warning, because signing the far end does not enforce
looks identical to signing that works. `--no-probe-unsigned` downgrades it.

The result block gains a signing section. `verified` is the count of frames the
vehicle signed and we checked — zero after a healthy flight means the key never
landed. `bad-sig` above zero is a security event, not noise.

Measured on 2 vehicles, signed against unsigned, same fleet back to back:
p50 4.2 ms both; p99 16.4 ms unsigned, 18.4 ms signed, against a 25 ms budget.

**`--gcs` still works, and becomes honest.** QGroundControl holds no key, so
with `--sign` on it keeps receiving telemetry but its commands are refused —
measured, not assumed. serial1 was always documented as a read-only observer;
signing is what turns that from a convention into something enforced.

### The native gate as a separate process

The loop calls the Rust gate per plan. For the file-handoff workflow — an FSM
writing `plan.json` and a gate polling it — the original runner is still here:

```bash
./sim/run_supervisor.sh --loop       # Linux only: uses taskset and stat -c
```

## Tests

```bash
python -m pytest tests/                            # all 305
python -m pytest tests/test_nullspace.py -v        # the D1 rank ladder
python -m pytest tests/test_stationary_estimate.py # D4.8, the named regression
python -m pytest tests/test_domain4_loop.py        # the loop's silent-failure modes
python -m pytest tests/test_reject_unsigned.py     # H0: what the link must refuse
```

## Troubleshooting the live loop

| symptom | cause |
|---|---|
| `no heartbeat on udpin:127.0.0.1:14551` | SITL not up, or launched without `--serial2 udpclient:127.0.0.1:<port>`. If the SITL log's last line is `Waiting for connection ....`, it was also launched without `--serial0 tcp:0` and is blocked in `accept()` |
| `no ground truth arrived` | SIMSTATE not streaming; check `SR0_EXTRA1` and that nothing else is draining the socket |
| ground truth source reads `SIM_STATE` | SIMSTATE stopped. Harmless (`lat_int` is exact) but worth knowing |
| `only vehicles [] armed` | read the STATUSTEXT lines printed just above — ArduPilot always says why |
| `extnav stream ... GAPPED` | the sender thread was starved; the estimator tick is overrunning |
| any `supervisor REJECT` | a bug report, not an event. The horizon guard should have caught it first |
| `--sign`: `no keystore passphrase` | `export HIVE_KEYSTORE_PASSPHRASE=...`. There is deliberately no default |
| `--sign`: `signing is provisioned but the vehicle still acts on unsigned commands` | the fleet is on serial0 (channel 0 accepts unsigned unconditionally). Relaunch with `run_fleet.sh`, which uses `--serial2` |
| `--sign`: no ground truth, but it worked unsigned | the vehicle holds a different key from a previous session. Relaunch the fleet (`run_fleet.sh` passes `--wipe`) |
| QGroundControl connects but its commands do nothing | expected with `--sign` — QGC holds no key, so serial1 is enforced read-only |
