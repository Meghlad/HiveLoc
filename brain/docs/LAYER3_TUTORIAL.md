# 🧭 Layer 3 Tutorial — A Language-Conditioned Mission Planner, With a Safety Supervisor

**The framing, exactly:** this is **not a VLA**. The action space is waypoints; the low-level policy is a flight controller better than anything we'd train; and the open problem in a GPS-denied swarm isn't motor control — it's whether the estimate is trustworthy enough to act on. So Layer 3 is a **language-conditioned mission planner** whose plans pass through a **deterministic safety supervisor** that assumes the model is wrong.

> Say the sentence in the interview verbatim: *"The plan never reaches MAVLink, because the supervisor is the only thing that can emit a setpoint, and it's ~400 lines of Rust with no model in it."*

---

## 1. The two tiers

```
operator instruction ─┐
downlinked mosaic ─────┼──▶ Claude (claude-opus-4-8) ──▶ structured JSON plan ──▶ ┌───────────────────┐
swarm estimate + cov ──┘        constrained decoding      (waypoints, spacing,    │  swarm-supervisor  │
                                (output_config.format)     assignments)           │  (Rust, no model)  │
                                                                                   └─────────┬─────────┘
                                                              reject ◀── any rule fails ──────┤
                                                                                              │ accept
                                                                                              ▼
                                                                          SET_POSITION_TARGET_LOCAL_NED
```

**Files:** `layer3_vlm_planner.py` (the model tier), `rust/swarm-supervisor/` (the gate).

---

## 2. Guardrail 1 — constrained decoding (no free text reaches the aircraft)

The planner calls `claude-opus-4-8` with `output_config.format` set to a strict `json_schema` — the **Plan** schema. The response is constrained to that shape *at decode time*, so there is **no channel for prose** to the aircraft. It's not "we prompt it to return JSON and hope"; the model literally cannot emit anything but a schema-valid plan.

```python
resp = client.messages.create(
    model="claude-opus-4-8",
    thinking={"type": "adaptive"},
    system=SYSTEM,                     # states the hard rules up front
    messages=[{"role": "user", "content": [image?, state_json, instruction]}],
    output_config={"format": {"type": "json_schema", "schema": PLAN_SCHEMA}},
)
```

The world state handed to the model *is* the Layer-2 output: each vehicle's estimated position **and its marginal covariance trace**, with a `trusted` flag. The planner is told to leave high-covariance vehicles where they are. So the planner and the supervisor gate on the *same* trust signal — the one Layer 2 made honest.

**Runs offline too.** With no `ANTHROPIC_API_KEY` or profile, the planner falls back to a deterministic geometric planner (line / circle / grid / hold keyword interpretation) that emits the *identical* Plan schema. The offline path is clearly labeled — it exists so the pipeline is always demonstrable, and because **the interesting engineering is the supervisor underneath, not the model on top.**

---

## 3. Guardrail 2 — the supervisor, where the guardrail is the point

`rust/swarm-supervisor` is a pure Rust crate. `validate(plan, estimate, config, now) -> Decision` is a **pure function** — same inputs, same verdict, testable without an aircraft, an estimator, or a model. Every plan is rejected unless **all** hold:

| # | Rule | Why |
|---|---|---|
| 1 | every waypoint strictly **inside the geofence** | the hillside case — a hallucinated point over the fence |
| 2 | commanded **spacing ≥ minimum** (plan may ask wider, never narrower) | formations that would collide |
| 3 | each assigned vehicle's **marginal covariance ≤ threshold** | moving a drone you can't localize is how formations collide — and Layer 2 is what makes this number trustworthy |
| 4 | plan **and** estimate are **fresh** | a stale plan is a dead plan |

Plus structural sanity: no unknown vehicle IDs, no duplicates, non-empty. **Rejection is total** — one violation and *nothing* moves; there is no partial acceptance, no best-effort, no model-in-the-loop retry. On accept, and only with `--emit`, the supervisor encodes `SET_POSITION_TARGET_LOCAL_NED` per vehicle and sends it to that vehicle's MAVLink port. **A rejected plan produces exactly zero packets** (`process::exit(2)` before any socket write).

### The hillside test, and 10 others

The crate ships 11 unit tests, all passing, that each earn their keep:

```
test hillside_waypoint_rejected ... ok      # the canonical failure
test fence_boundary_is_outside ... ok       # on-the-fence = over-the-fence
test collision_spacing_rejected ... ok
test plan_cannot_shrink_the_spacing_floor ... ok   # the model asks for 0.001; the floor holds
test uncertain_vehicle_not_commanded ... ok        # cov=0.5 → refused
test stale_plan_rejected / stale_estimate_rejected ... ok
test unknown_and_duplicate_vehicles_rejected ... ok
test rejection_is_total ... ok              # one bad assignment poisons the whole plan
```

### End-to-end, on real estimator data

```
$ python src/planning/layer3_vlm_planner.py "form a tight circle in the center"
--- PLAN (source: offline-geometric) ---   # or claude-opus-4-8 with a credential
{ "assignments": [ {"vehicle":0,"waypoint_ne":[0.8,0.5]}, ... ] }
--- SUPERVISOR VERDICT (exit 0) ---
{ "plan_id": "...", "accepted": true, "violations": [] }
```

A hallucinated waypoint at `[1.6, 0.4]` (outside the unit square) is caught:
```
--- SUPERVISOR VERDICT (exit 2) ---
{ "accepted": false, "violations": [{ "kind": "WaypointOutsideGeofence", "vehicle": 0, ... }] }
supervisor: plan 'hallucinated-2' REJECTED - no setpoints emitted
```

And the covariance gate, on a degraded-radio frame where the estimator distrusts one vehicle (cov up to 0.009): the planner assigns **11 of 12** vehicles — the untrusted one is left where it is, automatically.

---

## 4. Why this is the honest design

A VLM *will* eventually hallucinate a waypoint into a hillside. The answer isn't a better prompt or a bigger model — it's that the plan never reaches MAVLink, because the only component that can emit a setpoint has no model in it and rejects anything that violates geofence, spacing, trust, or freshness. Most candidates who mention LLMs have bolted one onto something; this is the thing built to **assume the LLM is wrong**.

The covariance gate is the through-line of the whole project: it's only meaningful because Layer 2's bearing factors made the marginal covariance an honest signal (Layer 2's degraded range-only estimator reported σ = 4 cm while erring 28 cm — a supervisor gating on *that* would be gating on a lie). Estimation quality and safety are the same problem.

---

## 5. Reproduce

```bash
cargo build --release --manifest-path rust/Cargo.toml
cargo test -p swarm-supervisor --release        # 11 tests, incl. hillside

python src/planning/layer3_vlm_planner.py "form a line along the north edge"
python src/planning/layer3_vlm_planner.py "spread into a grid"
# with a credential (ANTHROPIC_API_KEY or `ant auth login`), the same command
# routes through claude-opus-4-8 with constrained decoding instead of the offline planner
```

## 6. Resume framing (the caution)

Call it a **language-conditioned mission planner with a safety supervisor** — never a VLA. If asked why not a VLA: *the action space is waypoints, the low-level policy is a flight controller better than anything I'd train, and the open problem in a GPS-denied swarm isn't motor control, it's whether the estimate is trustworthy enough to act on.*
