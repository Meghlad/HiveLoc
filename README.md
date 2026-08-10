# coop-swarm

Two workspaces, one lineage.

```
brain/         the estimator + flight stack this all rests on  (Parts A, B, C)
hive_assist/   anchor-referenced cooperative swarm             (Domains 1-4)
```

---

## `brain/` — the Brain

The finished body of work: a GPS-denied cooperative localizer that a real
autopilot flies on. SDP relaxation → distributed ADMM → robust certification →
GTSAM batch smoothing → **iSAM2 incremental**, then closed into ArduCopter SITL
as the vehicle's sole position source with GPS switched off. Plus the four
production layers: vision-bearing fusion, a Rust transport/perception path, a
ROS 2 workspace, and a language planner behind a deterministic Rust supervisor.

Start at [brain/README.md](brain/README.md); run instructions in
[brain/RUN.md](brain/RUN.md). **Run its commands from inside `brain/`** — the
scripts resolve `data/` and `figures/` relative to the working directory.

Nothing in `brain/` changed in the split except its path. It is the substrate
`hive_assist/` builds on, not a dependency to be edited.

## `hive_assist/` — anchor-referenced cooperative swarm

New work. Takes a target coordinate from an external detector and carries it
through to gated setpoints: a GPS-surveyed anchor makes the estimator's
information matrix full-rank (no gauge pinning), the swarm holds an anchored
loiter mesh, an auction elects a sub-team, and an event-triggered agent flies a
guided approach to a standoff station near the target.

- Plan: [hive_assist/ANCHOR_SWARM_PLAN.md](hive_assist/ANCHOR_SWARM_PLAN.md)
- Docs and run instructions: [hive_assist/README.md](hive_assist/README.md)

### What it reuses from the Brain

| Brain asset | How `hive_assist/` uses it |
|---|---|
| iSAM2 estimator (`src/estimation/day8_isam2*.py`) | the graph the anchor factors attach to |
| Rust `swarm-supervisor` invariants | guard predicates share one definition of "valid" |
| `PLAN_SCHEMA` (`src/planning/layer3_vlm_planner.py`) | the wire shape the FSM emits plans in |
| MAVLink `close_the_loop.py` path | where gated setpoints eventually go |

Reuse first, extend second.
