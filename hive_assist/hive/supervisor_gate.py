"""D2.2 — the invariants, in Python, byte-compatible with the Rust supervisor.

WHY A SECOND COPY EXISTS AT ALL. The Rust `swarm-supervisor` is the only thing
that may emit a setpoint, and that stays true — nothing here transmits anything.
But the plan's Domain 2 requires that `ACCEPT` be the *steady state*: the FSM
must only ever hand over streams that already satisfy the invariants, so a
`REJECT` is a caught bug rather than an expected event. To promise that, the
planner has to be able to ask "would the supervisor take this?" thousands of
times per second while it searches — which it cannot do by shelling out to a
binary.

So this module is a mirror, and the mirror is *tested against the original*
(`tests/test_gate_parity.py` runs both on the same randomised cases and diffs
the decisions). If they ever disagree, the test fails and the disagreement is
the bug. A mirror nobody checks would be worse than no mirror at all — it would
let the planner believe in a gate that does not exist.

Mirrored exactly from brain/rust/swarm-supervisor/src/lib.rs:
  * freshness first (stale inputs make every other check meaningless)
  * boundary counts as OUTSIDE (a waypoint on the fence is one the wind puts over)
  * a plan may request WIDER spacing than the floor, never narrower
  * rejection is total: one violation and nothing moves
  * violations accumulate in the same order, so the two decisions compare 1:1

On top of the mirror, this module adds the thing the Rust gate has no reason to
know about: `check_stream`, which runs the same invariants over every tick of a
*forward-simulated* setpoint stream, plus the kinematic limits (slew rate) that
only make sense across consecutive ticks. That is what "pre-validated over
horizon H" means in the plan's FSM table.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# --------------------------------------------------------------------------
# Wire types — field names match the Rust structs so the JSON is interchangeable
# --------------------------------------------------------------------------
@dataclass
class Assignment:
    vehicle: int
    waypoint_ne: tuple[float, float]

    def to_json(self) -> dict:
        return {"vehicle": int(self.vehicle),
                "waypoint_ne": [float(self.waypoint_ne[0]),
                                float(self.waypoint_ne[1])]}


@dataclass
class Plan:
    plan_id: str
    issued_unix_ms: int
    assignments: list[Assignment]
    min_spacing_m: float | None = None

    def to_json(self) -> dict:
        return {
            "plan_id": self.plan_id,
            "issued_unix_ms": int(self.issued_unix_ms),
            "assignments": [a.to_json() for a in self.assignments],
            "min_spacing_m": self.min_spacing_m,
        }


@dataclass
class EstimateSnapshot:
    frame_index: int
    stamp_unix_ms: int
    pos: list[tuple[float, float]]
    cov_trace: list[float]

    def to_json(self) -> dict:
        return {
            "frame_index": int(self.frame_index),
            "stamp_unix_ms": int(self.stamp_unix_ms),
            "pos": [[float(p[0]), float(p[1])] for p in self.pos],
            "cov_trace": [float(c) for c in self.cov_trace],
        }


@dataclass
class SupervisorConfig:
    geofence: list[tuple[float, float]] = field(
        default_factory=lambda: [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
    min_spacing_m: float = 0.08
    max_cov_trace: float = 0.004
    max_plan_age_ms: int = 5_000
    max_estimate_age_ms: int = 1_000

    def to_json(self) -> dict:
        return {
            "geofence": [[float(p[0]), float(p[1])] for p in self.geofence],
            "min_spacing_m": self.min_spacing_m,
            "max_cov_trace": self.max_cov_trace,
            "max_plan_age_ms": int(self.max_plan_age_ms),
            "max_estimate_age_ms": int(self.max_estimate_age_ms),
        }


@dataclass
class Decision:
    plan_id: str
    accepted: bool
    violations: list[dict]

    def kinds(self) -> list[str]:
        return [v["kind"] for v in self.violations]


# --------------------------------------------------------------------------
# Geometry — ray casting, boundary excluded, matching the Rust exactly
# --------------------------------------------------------------------------
def inside_polygon(p, poly) -> bool:
    """Boundary counts as OUTSIDE: a waypoint ON the fence is a waypoint the
    wind puts over it. (Rust: `inside_polygon`.)"""
    if len(poly) < 3:
        return False                      # a degenerate fence admits nothing
    px, py = float(p[0]), float(p[1])
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = float(poly[i][0]), float(poly[i][1])
        xj, yj = float(poly[j][0]), float(poly[j][1])

        cross = (xj - xi) * (py - yi) - (yj - yi) * (px - xi)
        if (abs(cross) < 1e-12
                and min(xi, xj) - 1e-12 <= px <= max(xi, xj) + 1e-12
                and min(yi, yj) - 1e-12 <= py <= max(yi, yj) + 1e-12):
            return False                  # exactly on an edge
        if (yi > py) != (yj > py):
            x_at = (xj - xi) * (py - yi) / (yj - yi) + xi
            if px < x_at:
                inside = not inside
        j = i
    return inside


def clearance(points) -> float:
    """Smallest pairwise distance in a set of waypoints. inf for < 2 points."""
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        return float("inf")
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    return float(d.min())


# --------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------
def validate(plan: Plan, est: EstimateSnapshot, cfg: SupervisorConfig,
             now_unix_ms: int) -> Decision:
    """Pure function, same verdict as the Rust `validate` for the same inputs.

    Violation *order* is mirrored too, not just the set: freshness, then
    structure, then geofence, then spacing, then covariance. The parity test
    compares the full ordered list, which is a much sharper check than
    comparing the accept/reject bit alone.
    """
    v: list[dict] = []

    # 4. freshness first: stale inputs make every other check meaningless
    plan_age = max(0, now_unix_ms - plan.issued_unix_ms)   # saturating_sub
    if plan_age > cfg.max_plan_age_ms:
        v.append({"kind": "PlanStale", "age_ms": plan_age})
    est_age = max(0, now_unix_ms - est.stamp_unix_ms)
    if est_age > cfg.max_estimate_age_ms:
        v.append({"kind": "EstimateStale", "age_ms": est_age})

    if not plan.assignments:
        v.append({"kind": "NoAssignments"})

    # structural sanity: every assigned vehicle must exist, exactly once
    n = len(est.pos)
    seen: set[int] = set()
    for a in plan.assignments:
        if a.vehicle >= n:
            v.append({"kind": "UnknownVehicle", "vehicle": a.vehicle})
        if a.vehicle in seen:
            v.append({"kind": "DuplicateAssignment", "vehicle": a.vehicle})
        seen.add(a.vehicle)

    # 1. geofence: every waypoint strictly inside
    for a in plan.assignments:
        if not inside_polygon(a.waypoint_ne, cfg.geofence):
            v.append({"kind": "WaypointOutsideGeofence", "vehicle": a.vehicle,
                      "waypoint_ne": [float(a.waypoint_ne[0]),
                                      float(a.waypoint_ne[1])]})

    # 2. spacing: the plan may ask for WIDER spacing than the floor, never narrower
    min_spacing = max(
        cfg.min_spacing_m if plan.min_spacing_m is None else plan.min_spacing_m,
        cfg.min_spacing_m,
    )
    for i, a in enumerate(plan.assignments):
        for b in plan.assignments[i + 1:]:
            d = float(np.hypot(a.waypoint_ne[0] - b.waypoint_ne[0],
                               a.waypoint_ne[1] - b.waypoint_ne[1]))
            if d < min_spacing:
                v.append({"kind": "SpacingTooClose", "a": a.vehicle,
                          "b": b.vehicle, "dist": d, "min": min_spacing})

    # 3. trust: never command a vehicle the estimator cannot localise
    for a in plan.assignments:
        if 0 <= a.vehicle < len(est.cov_trace):
            trace = est.cov_trace[a.vehicle]
            if trace > cfg.max_cov_trace:
                v.append({"kind": "CovarianceTooHigh", "vehicle": a.vehicle,
                          "trace": float(trace), "max": cfg.max_cov_trace})

    return Decision(plan_id=plan.plan_id, accepted=not v, violations=v)


# --------------------------------------------------------------------------
# Horizon guards — what the Rust gate has no reason to know about
# --------------------------------------------------------------------------
@dataclass
class StreamLimits:
    """Kinematic limits that only exist *between* ticks."""

    v_max: float = 0.9              # m per tick
    d_clear: float = 0.35           # m, hard floor on inter-vehicle spacing
    dt: float = 1.0


@dataclass
class StreamVerdict:
    ok: bool
    violations: list[dict]
    min_clearance: float
    max_step: float

    def kinds(self) -> list[str]:
        return sorted({v["kind"] for v in self.violations})


def check_stream(stream, cfg: SupervisorConfig, limits: StreamLimits,
                 vehicles=None, static=None) -> StreamVerdict:
    """Validate a forward-simulated setpoint stream over the whole horizon.

    `stream` is (H, K, 2): H ticks of K vehicle positions. This is the guard the
    plan's FSM table calls "must hold over horizon H" — every tick in-geofence,
    every tick above the clearance floor, every tick-to-tick step within v_max.

    Checked BEFORE any motion is commanded, which is the entire design: the
    supervisor is never routed around, it is never even given the chance to say
    no, because the compiler does not emit streams that would earn a no.

    `static` is the rest of the swarm: agents not in this stream that are
    holding position. **Pass them.** The first version of this function did not
    take them, and the M->N handover then reported zero violations while a
    re-configuring loiter agent closed to 1.16 m of a stationary coalition
    member — each set had been checked only against itself, because the control
    graph was block-diagonal. Block-diagonalising the *control* graph does not
    block-diagonalise the *collision* constraint; aluminium does not care which
    Laplacian it belongs to.
    """
    s = np.asarray(stream, dtype=float)
    if s.ndim != 3 or s.shape[2] != 2:
        raise ValueError(f"stream must be (H, K, 2), got {s.shape}")
    h, k, _ = s.shape
    ids = list(range(k)) if vehicles is None else list(vehicles)
    held = (np.zeros((0, 2)) if static is None
            else np.asarray(static, dtype=float).reshape(-1, 2))

    viol: list[dict] = []
    worst_clear = float("inf")
    worst_step = 0.0

    for t in range(h):
        for idx, veh in enumerate(ids):
            if not inside_polygon(s[t, idx], cfg.geofence):
                viol.append({"kind": "WaypointOutsideGeofence", "tick": t,
                             "vehicle": veh,
                             "waypoint_ne": s[t, idx].tolist()})

        c = clearance(np.vstack([s[t], held]))
        worst_clear = min(worst_clear, c)
        if c < limits.d_clear:
            viol.append({"kind": "SpacingTooClose", "tick": t, "dist": c,
                         "min": limits.d_clear})

        if t > 0:
            steps = np.linalg.norm(s[t] - s[t - 1], axis=1)
            worst_step = max(worst_step, float(steps.max()))
            for idx, veh in enumerate(ids):
                if steps[idx] > limits.v_max * limits.dt + 1e-9:
                    viol.append({"kind": "VelocityTooHigh", "tick": t,
                                 "vehicle": veh, "step": float(steps[idx]),
                                 "max": limits.v_max * limits.dt})

    return StreamVerdict(ok=not viol, violations=viol,
                         min_clearance=worst_clear, max_step=worst_step)
