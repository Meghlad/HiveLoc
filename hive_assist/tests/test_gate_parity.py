"""D2.2 acceptance: the Python mirror and the Rust supervisor decide identically.

A mirror nobody checks is worse than no mirror — it lets the planner believe in
a gate that does not exist. So this runs the actual Rust binary on randomised
cases and diffs the full ordered violation list, not just the accept bit.

The binary lives under brain/rust/target/, which is gitignored, so a fresh clone
will not have it. The parity tests SKIP rather than fail in that case; the pure
Python behaviour tests below always run. Build it with:

    cargo build --release --manifest-path brain/rust/Cargo.toml
"""

import json
import pathlib
import subprocess

import numpy as np
import pytest

from hive.supervisor_gate import (
    Assignment,
    EstimateSnapshot,
    Plan,
    SupervisorConfig,
    clearance,
    inside_polygon,
    validate,
)

REPO = pathlib.Path(__file__).resolve().parents[2]
BINARY = REPO / "brain/rust/target/release/swarm-supervisor"
NOW = 1_000_100


def rust_decision(plan, est, cfg, now_ms, tmp_path) -> dict:
    p = tmp_path / "plan.json"
    e = tmp_path / "est.json"
    c = tmp_path / "cfg.json"
    p.write_text(json.dumps(plan.to_json()))
    e.write_text(json.dumps(est.to_json()))
    c.write_text(json.dumps(cfg.to_json()))

    out = subprocess.run(
        [str(BINARY), "--plan", str(p), "--estimate", str(e),
         "--config", str(c), "--now-ms", str(now_ms)],
        capture_output=True, text=True, timeout=30,
    )
    # the binary prints the Decision JSON followed by a human summary line
    text = out.stdout
    end = text.rindex("}") + 1
    return json.loads(text[text.index("{"):end])


needs_binary = pytest.mark.skipif(
    not BINARY.exists(),
    reason=f"Rust supervisor not built at {BINARY} — "
           f"cargo build --release --manifest-path brain/rust/Cargo.toml",
)


# --------------------------------------------------------------------------
# Randomised parity
# --------------------------------------------------------------------------
def random_case(rng):
    """Deliberately near the boundaries. Uniform-random plans would be accepted
    or rejected for boring reasons and never exercise a gate edge."""
    n = int(rng.integers(2, 8))
    est = EstimateSnapshot(
        frame_index=int(rng.integers(0, 500)),
        stamp_unix_ms=NOW - int(rng.integers(0, 2500)),
        pos=[(float(rng.uniform(0, 1)), float(rng.uniform(0, 1)))
             for _ in range(n)],
        cov_trace=[float(rng.choice([0.0005, 0.002, 0.0039, 0.0041, 0.02]))
                   for _ in range(n)],
    )
    k = int(rng.integers(0, n + 2))
    assignments = []
    for _ in range(k):
        veh = int(rng.integers(0, n + 1))          # sometimes out of range
        # cluster near the fence and near each other, to hit both gates
        wp = (float(rng.choice([rng.uniform(-0.2, 1.2), rng.uniform(0.45, 0.55),
                                0.0, 1.0])),
              float(rng.choice([rng.uniform(-0.2, 1.2), rng.uniform(0.45, 0.55),
                                0.0, 1.0])))
        assignments.append(Assignment(vehicle=veh, waypoint_ne=wp))

    plan = Plan(
        plan_id=f"p{rng.integers(0, 10_000)}",
        issued_unix_ms=NOW - int(rng.integers(0, 9000)),
        assignments=assignments,
        min_spacing_m=float(rng.choice([0.001, 0.08, 0.2])) if rng.random() < 0.5
        else None,
    )
    return plan, est


@needs_binary
def test_parity_on_randomised_cases(tmp_path):
    rng = np.random.default_rng(4)
    cfg = SupervisorConfig()
    disagreements = []

    for i in range(120):
        plan, est = random_case(rng)
        mine = validate(plan, est, cfg, NOW)
        theirs = rust_decision(plan, est, cfg, NOW, tmp_path)

        if (mine.accepted != theirs["accepted"]
                or mine.kinds() != [v["kind"] for v in theirs["violations"]]):
            disagreements.append({
                "case": i,
                "python": {"accepted": mine.accepted, "kinds": mine.kinds()},
                "rust": {"accepted": theirs["accepted"],
                         "kinds": [v["kind"] for v in theirs["violations"]]},
                "plan": plan.to_json(),
            })

    assert not disagreements, (
        f"{len(disagreements)}/120 disagreed:\n"
        + json.dumps(disagreements[:3], indent=2)
    )


@needs_binary
def test_parity_on_the_rust_crates_own_test_vectors(tmp_path):
    """The exact scenarios brain/rust/swarm-supervisor/src/lib.rs asserts on."""
    cfg = SupervisorConfig()
    est = EstimateSnapshot(
        frame_index=100, stamp_unix_ms=1_000_000,
        pos=[(0.1 + 0.06 * i, 0.5) for i in range(12)],
        cov_trace=[0.001] * 12,
    )

    cases = {
        "valid_plan_passes": (Plan("p1", 1_000_000, [
            Assignment(0, (0.2, 0.2)), Assignment(1, (0.8, 0.8))]), NOW),
        "hillside_waypoint_rejected": (Plan("p1", 1_000_000, [
            Assignment(0, (1.7, 0.4))]), NOW),
        "fence_boundary_is_outside": (Plan("p1", 1_000_000, [
            Assignment(0, (1.0, 0.5))]), NOW),
        "collision_spacing_rejected": (Plan("p1", 1_000_000, [
            Assignment(0, (0.50, 0.50)), Assignment(1, (0.52, 0.50))]), NOW),
        "plan_cannot_shrink_the_floor": (Plan("p1", 1_000_000, [
            Assignment(0, (0.50, 0.50)), Assignment(1, (0.55, 0.50))],
            min_spacing_m=0.001), NOW),
        "stale_plan_rejected": (Plan("p1", 1_000_000, [
            Assignment(0, (0.5, 0.5))]), NOW + 60_000),
        "empty_plan_rejected": (Plan("p1", 1_000_000, []), NOW),
        "rejection_is_total": (Plan("p1", 1_000_000, [
            Assignment(0, (0.2, 0.2)), Assignment(1, (9.9, 9.9))]), NOW),
    }

    for name, (plan, now) in cases.items():
        mine = validate(plan, est, cfg, now)
        theirs = rust_decision(plan, est, cfg, now, tmp_path)
        assert mine.accepted == theirs["accepted"], name
        assert mine.kinds() == [v["kind"] for v in theirs["violations"]], name


# --------------------------------------------------------------------------
# Behaviour of the mirror itself — runs with or without the binary
# --------------------------------------------------------------------------
def est12():
    return EstimateSnapshot(100, 1_000_000,
                            [(0.1 + 0.06 * i, 0.5) for i in range(12)],
                            [0.001] * 12)


def test_valid_plan_accepted():
    d = validate(Plan("p", 1_000_000,
                      [Assignment(0, (0.2, 0.2)), Assignment(1, (0.8, 0.8))]),
                 est12(), SupervisorConfig(), NOW)
    assert d.accepted and d.violations == []


def test_boundary_is_outside():
    assert not inside_polygon((1.0, 0.5), SupervisorConfig().geofence)
    assert inside_polygon((0.999, 0.5), SupervisorConfig().geofence)


def test_degenerate_fence_admits_nothing():
    assert not inside_polygon((0.5, 0.5), [(0.0, 0.0), (1.0, 1.0)])


def test_plan_cannot_shrink_the_spacing_floor():
    d = validate(Plan("p", 1_000_000,
                      [Assignment(0, (0.50, 0.50)), Assignment(1, (0.55, 0.50))],
                      min_spacing_m=0.001),
                 est12(), SupervisorConfig(), NOW)
    assert "SpacingTooClose" in d.kinds()


def test_plan_may_request_wider_spacing():
    d = validate(Plan("p", 1_000_000,
                      [Assignment(0, (0.30, 0.50)), Assignment(1, (0.45, 0.50))],
                      min_spacing_m=0.25),
                 est12(), SupervisorConfig(), NOW)
    assert "SpacingTooClose" in d.kinds()


def test_uncertain_vehicle_not_commanded():
    e = est12()
    e.cov_trace[3] = 0.5
    d = validate(Plan("p", 1_000_000, [Assignment(3, (0.5, 0.5))]),
                 e, SupervisorConfig(), NOW)
    assert "CovarianceTooHigh" in d.kinds()


def test_stale_inputs_rejected():
    assert "PlanStale" in validate(
        Plan("p", 1_000_000, [Assignment(0, (0.5, 0.5))]),
        est12(), SupervisorConfig(), NOW + 60_000).kinds()

    e = est12()
    e.stamp_unix_ms = 1
    assert "EstimateStale" in validate(
        Plan("p", 1_000_000, [Assignment(0, (0.5, 0.5))]),
        e, SupervisorConfig(), NOW).kinds()


def test_unknown_and_duplicate_vehicles():
    d = validate(Plan("p", 1_000_000, [
        Assignment(40, (0.5, 0.5)), Assignment(1, (0.3, 0.3)),
        Assignment(1, (0.7, 0.7))]), est12(), SupervisorConfig(), NOW)
    assert "UnknownVehicle" in d.kinds() and "DuplicateAssignment" in d.kinds()


def test_rejection_is_total():
    d = validate(Plan("p", 1_000_000, [
        Assignment(0, (0.2, 0.2)), Assignment(1, (9.9, 9.9))]),
        est12(), SupervisorConfig(), NOW)
    assert not d.accepted


def test_clearance_helper():
    assert clearance([(0.0, 0.0), (3.0, 4.0)]) == pytest.approx(5.0)
    assert clearance([(0.0, 0.0)]) == float("inf")
