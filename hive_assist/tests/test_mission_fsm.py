"""D2.3 acceptance: an M->N handover with zero supervisor rejections.

The plan's deliverable. But "zero rejections" is trivially satisfiable by a
gate that never says no, so the suite has to prove two things at once:

  the happy path       a full ANCHOR_INIT -> DISPATCH run submits N plans and
                       every one is accepted, by the REAL Rust supervisor
  the gate has teeth   deliberately bad streams are caught, and the FSM refuses
                       to move rather than moving and apologising

Also carries the regression for the cross-block clearance bug: the first version
of the guard checked each control block only against itself, so a re-configuring
loiter agent closed to 1.16 m of a stationary coalition member while the log
cheerfully reported zero violations.
"""

import json
import pathlib
import subprocess

import numpy as np
import pytest

from hive.mission_fsm import (
    MissionFSM,
    State,
    default_scenario,
    lloyd_targets,
    slew_toward,
)
from hive.cbba import SwarmState, locational_cost
from hive.supervisor_gate import (
    StreamLimits,
    SupervisorConfig,
    check_stream,
    clearance,
)

REPO = pathlib.Path(__file__).resolve().parents[2]
BINARY = REPO / "brain/rust/target/release/swarm-supervisor"
needs_binary = pytest.mark.skipif(
    not BINARY.exists(), reason="Rust supervisor not built")


@pytest.fixture(scope="module")
def flown():
    swarm, cfg = default_scenario()
    fsm = MissionFSM(swarm, cfg)
    log = fsm.run(np.array([16.0, 3.0]), np.array([[13.2, 1.0], [13.2, 5.4]]))
    return fsm, log


# --------------------------------------------------------------------------
# The deliverable
# --------------------------------------------------------------------------
def test_handover_completes(flown):
    fsm, log = flown
    assert fsm.state is State.DISPATCH
    assert [t[1] for t in log.transitions] == [
        "LOITER_MESH", "TASK_INGEST", "AUCTION", "RECONFIG", "DISPATCH"]


def test_zero_supervisor_rejections(flown):
    _, log = flown
    assert log.decisions, "no plans were submitted at all"
    assert log.rejections == [], (
        f"{len(log.rejections)} rejections — the guard let something through:\n"
        + json.dumps(log.rejections[:3], indent=2, default=str))
    assert all(d["accepted"] for d in log.decisions)


def test_zero_guard_failures(flown):
    _, log = flown
    assert log.guard_failures == []


def test_the_swarm_actually_moved(flown):
    """Zero rejections is easy if nothing ever happens."""
    fsm, log = flown
    assert log.ticks >= 40
    assert len(fsm.active) == 2
    assert len(fsm.loiter) == len(fsm.pos) - 2


def test_coalition_reached_its_stations(flown):
    fsm, _ = flown
    stations = np.array([[13.2, 1.0], [13.2, 5.4]])
    reached = np.linalg.norm(fsm.pos[fsm.active] - stations, axis=1)
    assert (reached < 0.5).all(), f"coalition stopped short: {reached}"


@needs_binary
def test_every_submitted_plan_is_accepted_by_the_real_rust_supervisor(
        flown, tmp_path):
    """The end-to-end claim, checked against the binary rather than the mirror.

    Sampled rather than exhaustive — one subprocess per tick would dominate the
    suite runtime, and test_gate_parity already pins mirror == Rust across 120
    randomised cases. This is the belt to that suspenders.
    """
    fsm, log = flown
    idx = np.linspace(0, len(log.submitted) - 1, 10).astype(int)

    for i in idx:
        plan, est, now_ms = log.submitted[i]
        p, e, c = tmp_path / "p.json", tmp_path / "e.json", tmp_path / "c.json"
        p.write_text(json.dumps(plan.to_json()))
        e.write_text(json.dumps(est.to_json()))
        c.write_text(json.dumps(fsm.cfg.supervisor.to_json()))
        out = subprocess.run(
            [str(BINARY), "--plan", str(p), "--estimate", str(e),
             "--config", str(c), "--now-ms", str(now_ms)],
            capture_output=True, text=True, timeout=30)
        text = out.stdout
        d = json.loads(text[text.index("{"):text.rindex("}") + 1])
        assert d["accepted"], f"tick {i}: Rust rejected {d['violations']}"


# --------------------------------------------------------------------------
# Invariants held throughout, not just at the endpoints
# --------------------------------------------------------------------------
def test_clearance_floor_held_every_tick(flown):
    fsm, log = flown
    assert log.min_clearance >= fsm.cfg.limits.d_clear


def test_velocity_limit_held_every_tick(flown):
    fsm, log = flown
    limit = fsm.cfg.limits.v_max * fsm.cfg.limits.dt
    assert log.max_step <= limit + 1e-9


def test_every_agent_is_commanded_every_tick(flown):
    """Mover-vs-holder separation is only checked if the holders are in the
    plan. If a future change trims the plan back to just the movers, this
    fails before the collision does."""
    fsm, log = flown
    n = len(fsm.pos)
    for plan, _, _ in log.submitted:
        assert len(plan.assignments) == n
        assert sorted(a.vehicle for a in plan.assignments) == list(range(n))


# --------------------------------------------------------------------------
# Regression: the cross-block clearance bug
# --------------------------------------------------------------------------
def test_static_agents_are_included_in_the_clearance_check():
    """A stream that is perfectly safe among its own members and drives straight
    through a stationary agent. Without `static` this reports ok."""
    cfg = SupervisorConfig(geofence=[(-20, -20), (20, -20), (20, 20), (-20, 20)],
                           min_spacing_m=1.2)
    limits = StreamLimits(v_max=0.9, d_clear=1.2, dt=1.0)
    # steps of 0.8 m, inside v_max, so the only thing that can fail is clearance
    stream = np.array([[[-2.4, 0.0]], [[-1.6, 0.0]], [[-0.8, 0.0]]])
    held = np.array([[0.0, 0.0]])

    assert check_stream(stream, cfg, limits).ok            # blind to the holder
    v = check_stream(stream, cfg, limits, static=held)
    assert not v.ok and "SpacingTooClose" in v.kinds()


def test_slew_avoids_static_agents():
    limits = StreamLimits(v_max=0.9, d_clear=1.2, dt=1.0)
    held = np.array([[0.0, 0.0]])
    s = slew_toward(np.array([[-6.0, 0.0]]), np.array([[6.0, 0.0]]),
                    limits, horizon=30, static=held)
    d = np.linalg.norm(s[:, 0, :] - held[0], axis=1)
    assert d.min() >= limits.d_clear


# --------------------------------------------------------------------------
# The gate has teeth
# --------------------------------------------------------------------------
def test_dispatch_outside_the_geofence_is_refused():
    swarm, cfg = default_scenario()
    fsm = MissionFSM(swarm, cfg)
    fsm.run(np.array([16.0, 3.0]), np.array([[400.0, 400.0], [402.0, 400.0]]))

    assert fsm.state is State.RECONFIG, "FSM transitioned on a failing guard"
    assert any(g["guard"] == "DISPATCH" for g in fsm.log.guard_failures)
    assert "WaypointOutsideGeofence" in [
        k for g in fsm.log.guard_failures for k in g["kinds"]]
    assert fsm.log.rejections == [], "guard should catch it BEFORE submission"


def test_refused_dispatch_does_not_move_the_coalition():
    """The important half: refusing must also mean not moving."""
    swarm, cfg = default_scenario()
    fsm = MissionFSM(swarm, cfg)
    fsm.anchor_init()
    fsm.ingest_task(np.array([16.0, 3.0]))
    fsm.run_auction()
    fsm.reconfigure()

    before = fsm.pos[fsm.active].copy()
    assert not fsm.dispatch(np.array([[400.0, 400.0], [402.0, 400.0]]))
    assert np.array_equal(fsm.pos[fsm.active], before)


def test_untrusted_agent_cannot_be_elected():
    """Domain 1's covariance reaching Domain 2's auction."""
    swarm, cfg = default_scenario()
    cov = np.full(len(swarm.pos), 0.001)
    cov[2] = cov[3] = 0.9                      # the two that normally win
    swarm = SwarmState(swarm.pos, swarm.battery, swarm.sensor, swarm.region,
                       cov_trace=cov)

    fsm = MissionFSM(swarm, cfg)
    fsm.run(np.array([16.0, 3.0]), np.array([[13.2, 1.0], [13.2, 5.4]]))
    assert 2 not in fsm.active and 3 not in fsm.active
    assert fsm.log.rejections == []


def test_check_stream_rejects_a_too_fast_stream():
    cfg = SupervisorConfig(geofence=[(-20, -20), (20, -20), (20, 20), (-20, 20)])
    limits = StreamLimits(v_max=0.5, d_clear=0.5, dt=1.0)
    stream = np.array([[[0.0, 0.0]], [[5.0, 0.0]]])
    v = check_stream(stream, cfg, limits)
    assert not v.ok and "VelocityTooHigh" in v.kinds()


def test_check_stream_rejects_a_malformed_stream():
    cfg = SupervisorConfig()
    with pytest.raises(ValueError):
        check_stream(np.zeros((4, 3)), cfg, StreamLimits())


# --------------------------------------------------------------------------
# The pieces
# --------------------------------------------------------------------------
def test_slew_never_exceeds_the_step_limit():
    rng = np.random.default_rng(5)
    limits = StreamLimits(v_max=0.4, d_clear=0.9, dt=1.0)
    start = rng.uniform(-8, 8, (7, 2))
    goal = rng.uniform(-8, 8, (7, 2))
    s = slew_toward(start, goal, limits, horizon=50)
    steps = np.linalg.norm(np.diff(s, axis=0), axis=2)
    assert steps.max() <= limits.v_max * limits.dt + 1e-9


def test_slew_actually_arrives():
    limits = StreamLimits(v_max=0.9, d_clear=0.5, dt=1.0)
    start = np.array([[0.0, 0.0], [0.0, 4.0]])
    goal = np.array([[8.0, 0.0], [8.0, 4.0]])
    s = slew_toward(start, goal, limits, horizon=40)
    assert np.linalg.norm(s[-1] - goal, axis=1).max() < 0.05


def test_lloyd_improves_coverage():
    swarm, _ = default_scenario()
    residual = swarm.pos[3:]
    before = locational_cost(residual, swarm.region)
    after = locational_cost(lloyd_targets(residual, swarm.region, 10),
                            swarm.region)
    assert after <= before


def test_reconfigure_closes_the_coverage_hole(flown):
    """The point of RECONFIG: after the coalition leaves, the residual mesh
    should cover the region at least as well as the depleted ring did."""
    fsm, _ = flown
    depleted = fsm.swarm.pos[fsm.loiter]
    assert (locational_cost(fsm.pos[fsm.loiter], fsm.swarm.region)
            <= locational_cost(depleted, fsm.swarm.region) + 1e-9)


def test_clearance_of_a_lone_point_is_infinite():
    assert clearance([(1.0, 2.0)]) == float("inf")
