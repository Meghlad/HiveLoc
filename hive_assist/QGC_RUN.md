# QGC_RUN — flying the standoff dispatch yourself, with QGroundControl open

A cheat sheet for messing around with S3 (the live standoff dispatch) and the
QGC marker upload, without needing me in the loop. Everything here is already
built and tested — this is just how to drive it.

---

## 0. One-time setup check

```bash
ls /Applications/QGroundControl.app        # installed via: brew install --cask qgroundcontrol
cd hive_assist
source ../.venv/bin/activate
```

QGC's data folder is `~/Documents/QGroundControl/` (logs, missions, telemetry
it records). You don't need to touch it.

---

## 1. Open QGC before you launch SITL

```bash
open -a QGroundControl
```

Give it a few seconds. Confirm it's actually listening:

```bash
lsof -nP -iUDP:14550 | grep -i qground
```

If that prints nothing, QGC isn't up yet — wait and re-check before launching
the fleet. SITL will still fly fine without it; you just won't see anything.

---

## 2. Fly it

```bash
make s3
```

is shorthand for:

```bash
./sim/run_fleet.sh -n 4 -m standoff -s 120 --gcs
```

What happens, in order: 4 SITL instances boot → estimator converges at rest →
all 4 arm and climb → the CBBA auction elects a 2-vehicle coalition → each
elected vehicle flies its own curvature-bounded Dubins approach to its own slot
on the 12 m standoff perimeter around the target → a 3-item marker mission
(target + 2 stations) uploads to all 4 vehicles → everyone lands.

**What to watch for in QGC:** the vehicle icons (pick whichever `system N` you
want in the vehicle selector, top-left). Two of them will curve out to the
perimeter and hold; the other two stay on the loiter mesh. The uploaded
mission markers (a waypoint icon at the target, one at each station) should
appear on the map — if they don't show up automatically, hit **Sync → Download
from vehicle** once in the Plan view.

Results print at the end: coalition elected, station coordinates, closest
approach to the target (should be ≥ 12 m — that's the whole point), final
distance to station against ground truth, and the gate's ACCEPT/REJECT tally.
It also writes:

```
sim/logs/standoff.json          # raw flown track, target, stations
figures/d4_live_standoff.png    # the figure, redrawn from that flight
```

---

## 3. Flags worth playing with

`run_fleet.sh` flags:

| flag | default | effect |
|---|---|---|
| `-n` | 4 | vehicle count |
| `-m` | hold | mode — see table below |
| `-s` | 40 | flight duration, seconds |
| `--gcs [port]` | off, 14550 | stream telemetry + mission to a GCS |
| `--speedup N` | 1 | SITL time multiplier (careful — the estimator has a real-time budget) |

Anything after these gets passed straight to `sim/orchestrator.py`. So:

```bash
./sim/run_fleet.sh -n 4 -m standoff -s 120 --gcs --target 30 10 --task sampling --slots 3
```

Orchestrator-level flags (only apply in `-m standoff`, except `--target`/`--task`
which `dispatch` also uses):

| flag | default | effect |
|---|---|---|
| `--target X Y` | `25.0 8.0` | the detector's coordinate, TacFrame ENU metres |
| `--task` | `inspection` | which standoff geometry — see table below |
| `--slots N` | 2 | coalition size the auction elects |
| `--spread DEG` | 45.0 | angular spread between coalition stations on the perimeter |
| `--no-markers` | off | skip the QGC mission upload (useful if QGC isn't open) |

## Task geometries (`--task`)

| task | standoff_m | bearing_deg | notes |
|---|---|---|---|
| `inspection` | 12.0 | 200.0 | faces the target; the default, matches the D3 figure |
| `sampling` | 4.0 | 170.0 | close run-in |
| `delivery` | 6.0 | 140.0 | fixed approach heading (crosswind-style, doesn't face target) |
| `probe` | 5.0 | 225.0 | faces the target |

---

## 4. The one constraint you'll hit if you pick your own `--target`

**The target must be far enough from the loiter mesh that every vehicle starts
outside the standoff perimeter.** With a 6 m mesh and the `inspection` task
(12 m standoff), the target needs to be roughly ≥ 18 m from the anchor in most
directions — closer than that and a vehicle can start already inside the
circle it's supposed to stay outside of. If you pick a target too close, you'll
get a clean refusal instead of a flight:

```
ValueError: vehicles [...] start inside the 12 m standoff perimeter of [...]
```

That's deliberate — flying it anyway would silently break the one claim this
domain exists to make. Move the target out, or pick a smaller-standoff task
(`sampling`, `delivery`, `probe`) if you want to test closer geometries.

---

## 5. Quick experiments to try

```bash
# smaller coalition, tighter task
./sim/run_fleet.sh -n 4 -m standoff -s 90 --gcs --task probe --slots 1

# bigger swarm, 3-vehicle coalition, spread wider apart on the perimeter
./sim/run_fleet.sh -n 6 -m standoff -s 150 --gcs --slots 3 --spread 90

# no GCS, just read the numbers (faster iteration while tuning)
./sim/run_fleet.sh -n 4 -m standoff -s 60 --no-markers

# re-plot the last flight's figure without re-flying
python -c "
import json
from hive.plots import plot_live_standoff
d = json.load(open('sim/logs/standoff.json'))
print(plot_live_standoff(d, task='inspection'))
"
```

---

## 6. Troubleshooting

| symptom | likely cause |
|---|---|
| markers don't appear on QGC's map | QGC cached its mission before upload finished — Sync → Download from vehicle |
| `vehicles [...] start inside the standoff perimeter` | target too close to the mesh — see §4 |
| `the coalition's own streams conflict: vX and vY come within ...` | an elected agent's route to its station passes through the loiter mesh. Refused on the ground rather than stalling in mid-air. Most likely at **n=6**, which is the weak spot (~18% of runs at `--slots 2`, ~30% at `--slots 3`); n=8 and n=12 refuse far less. Re-run, widen `--spread`, drop `--slots`, or push `--target` further out |
| drones stop part way to the perimeter | the loop now prints `STALLED at rung N/M` within 10 s and says why. If it names a guard failure the geometry conflicts (widen `--spread`, fewer `--slots`, or push `--target` out); if it says vehicles aren't reaching the rung, the host is too loaded — drop `-n` |
| `extnav stream ... GAPPED` at `-n 6` | host saturation on an 8 GB M1, not a loop bug. The flight still completes; the latency numbers from that run are not trustworthy |
| `at rest` ratio far from ~1 before it even arms | machine under load; give it a clean run |
| `p99` latency creeping toward/past 25 ms with 4+ vehicles | known — SITL scheduling contention at this vehicle count, not a code bug (see `done_till_now.md`) |
| everything else | see the general troubleshooting table in `RUN.md` |

`RUN.md` has the full picture (S0/S1/S2, the offline domains, test commands).
This file is just the S3 + QGC slice of it.
