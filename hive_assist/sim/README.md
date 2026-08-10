# Domain 4 — simulation orchestration

**This directory runs on the Zephyrus, not on the machine it was written on.**
macOS cannot run `gz-sim` headless against NVIDIA, and `tc netem` needs Linux
veth pairs. Everything here is authored, syntax-checked, and committed; the
20-vehicle run itself happens on the target host.

Start with:

```bash
./preflight.sh            # or --fix to offer creating the swapfile
```

It fails hard on the things that cost twenty minutes and a reboot to discover
halfway through a build — no swap, no `sch_netem`, docker can't see the NVIDIA
runtime.

## What is here

| file | what it is |
|---|---|
| `preflight.sh` | host capability check (D4.2) |
| `docker-compose.yml` | the stack: one gz-server, N PX4 replicas, bridge, our nodes (D4.1) |
| `run_supervisor.sh` | the supervisor, **native**, cpuset-pinned, `nice -5` |
| `netem_sweep.sh` | the 0–30% loss sweep (D4.2) |
| `launch/mission.launch.py` | FSM + estimator + task ingest + go-signal |
| `config/supervisor.json` | geofence and gates, in metres in TacFrame |
| `config/cyclonedds.xml` | DDS tuning for a 20-participant domain |
| `config/px4_entrypoint.sh` | one SITL instance attaching to the **shared** server |

## What still has to be built on the target host

These three depend on the exact PX4 / Gazebo / ROS 2 versions installed there,
so writing them blind would produce files that look complete and do not work:

- **`Dockerfile`** — ROS 2 Jazzy + `ros_gz` + our nodes. Start from
  [`../../brain/ros2_ws/Dockerfile`](../../brain/ros2_ws/Dockerfile), which
  already builds the Rust/`rclrs` half of this and is known to work.
- **`config/worlds/anchor_site.sdf`** — the world, with the surveyed anchor as a
  static model at the origin and spherical coordinates set to
  `anchor_lat/lon/alt` so PX4's GPS origin and `TacFrame` agree.
- **`launch/bridge.launch.py`** — `ros_gz_bridge` topic mappings, which are
  version-specific.

## Two things to know before reading the sweep

**One Gazebo server, not N.** The single biggest RAM decision in the whole
domain is in `docker-compose.yml`: all M models are hosted by one `gz-sim`
server. Launching one Gazebo per vehicle is the obvious-looking arrangement and
it is what puts this ~20 GB over a 16 GB budget.

**The sweep's interesting column is `max_jump`, not the hold count.** The half of
"freeze safely" that already works is the outage: a rejected plan emits nothing
and the vehicle sits still. The dangerous moment is the *recovery*. While the
vehicle is frozen the planner keeps advancing its stream, so the first plan to
land afterwards can be several ticks' worth of distance away — and it passes
every gate the supervisor currently has, because freshness bounds a plan's *age*
and nothing bounds its *distance*.

[`../hive/loss_model.py`](../hive/loss_model.py) runs that experiment in
simulation and measures a **7.2 m** commanded jump at 30% loss, against a legal
tick of 0.9 m. `netem_sweep.sh` reports `LUNGE` when it sees the same thing on
real veth pairs. The fix needs both halves:

1. a `SlewTooLarge` violation in
   [`../../brain/rust/swarm-supervisor/src/lib.rs`](../../brain/rust/swarm-supervisor/src/lib.rs)
   — ~10 lines, rejecting any waypoint further from the vehicle's estimate than
   `v_max · dt`;
2. a planner that re-plans from the vehicle's **actual** position after a hold.
   `MissionFSM` already does this. The gate alone, without re-planning, converts
   the lunge into a permanent safe stall — mission progress collapses to 9% at
   30% loss.

Both are proven host-independently in
[`../tests/test_safe_hold.py`](../tests/test_safe_hold.py), so the Zephyrus run
is a confirmation rather than the first evidence.
