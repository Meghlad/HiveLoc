"""D4.3 acceptance: the failure mode under packet loss is always "freeze safely".

The plan's Domain 4 stress test, run against the model rather than against netem,
so the claim is host-independent and lives in CI instead of in a lab notebook on
one laptop. The Zephyrus run confirms it on real veth pairs; it should not be the
first place anyone finds out.

Two properties, and the tests keep them separate because the interesting result
is that satisfying one is easy and satisfying both is not:

    never lunges     no commanded step exceeds what the vehicle can fly
    still arrives    the mission completes despite the loss
"""

import numpy as np
import pytest

from hive.loss_model import (
    CONFIGS,
    LinkModel,
    SafeHoldResult,
    SlewGate,
    dispatch_stream,
    run_link,
    sweep,
)
from hive.supervisor_gate import Assignment, EstimateSnapshot, Plan


@pytest.fixture(scope="module")
def swept():
    return sweep(seeds=8, horizon=90)


def _run(loss, gated, replan, seed=0, horizon=90):
    stream, cfg = dispatch_stream(horizon)
    gate = SlewGate(v_max=cfg.limits.v_max, dt=cfg.limits.dt, enabled=gated)
    return run_link(stream, cfg.supervisor, LinkModel(loss=loss), gate,
                    tick_ms=int(cfg.limits.dt * 1000), seed=seed,
                    replan=replan, v_max=cfg.limits.v_max), cfg


# --------------------------------------------------------------------------
# The half that already worked
# --------------------------------------------------------------------------
def test_an_outage_emits_nothing_and_moves_nothing():
    """A totally dead link must produce zero accepted plans and zero motion."""
    stream, cfg = dispatch_stream(60)
    gate = SlewGate(v_max=cfg.limits.v_max, dt=cfg.limits.dt)
    r = run_link(stream, cfg.supervisor, LinkModel(loss=1.0), gate,
                 tick_ms=int(cfg.limits.dt * 1000), v_max=cfg.limits.v_max)
    assert r.accepted == 0
    assert r.held == r.ticks
    assert r.max_commanded_jump == 0.0


def test_stale_estimate_is_what_stops_it():
    """The plan's stated mechanism: a delayed estimate trips EstimateStale."""
    stream, cfg = dispatch_stream(40)
    tick = int(cfg.limits.dt * 1000)
    link = LinkModel(loss=0.0, delay_ms=8_000.0, jitter_ms=0.0, reorder=0.0)
    r = run_link(stream, cfg.supervisor, LinkModel(loss=0.0), SlewGate(),
                 tick_ms=tick, v_max=cfg.limits.v_max)
    assert r.accepted > 0                     # sanity: it works when fresh

    # now make the estimate arrive far too late
    est = EstimateSnapshot(0, 0, [(0.0, 0.0)], [0.001])
    plan = Plan("p", 0, [Assignment(0, (0.0, 0.0))])
    from hive.supervisor_gate import validate
    d = validate(plan, est, cfg.supervisor, 10_000)
    assert "EstimateStale" in d.kinds()
    assert not d.accepted
    assert link.delay_ms > cfg.supervisor.max_estimate_age_ms


# --------------------------------------------------------------------------
# The half that did not: the post-blackout lunge
# --------------------------------------------------------------------------
def test_stale_stream_without_a_gate_lunges(swept):
    """The finding. Every existing gate passes the plan — it is fresh, in
    fence, correctly spaced, from a trusted vehicle — and it commands a step the
    vehicle cannot fly."""
    step = swept["step"]
    worst = max(r["max_jump"] for r in swept["stale_ungated"])
    assert worst > 3 * step, (
        f"expected a lunge, worst jump was only {worst:.2f} m")
    assert any(r["lunged"] > 0 for r in swept["stale_ungated"])


def test_lunge_magnitude_grows_with_loss(swept):
    """It is a systematic consequence of outage length, not a rare event.

    Measured by RANK correlation, not Pearson. The jump is a worst-case over
    seeds of a quantity that only takes integer multiples of the tick step (a
    lunge is always a whole number of skipped ticks) and that saturates once
    outages exceed max_plan_age_ms. A linear correlation on a quantised,
    saturating order statistic understates a trend that is plainly there —
    Pearson reads 0.57 on the same data Spearman reads 0.80.
    """
    from scipy.stats import spearmanr

    jumps = [r["max_jump"] for r in swept["stale_ungated"]]
    assert jumps[-1] > jumps[0]
    assert spearmanr(np.asarray(swept["loss"]), jumps).statistic > 0.7

    half = len(jumps) // 2
    assert np.mean(jumps[half:]) > 1.4 * np.mean(jumps[:half])


def test_the_slew_gate_eliminates_the_lunge(swept):
    step = swept["step"]
    for key in ("stale_gated", "replan_gated"):
        for r in swept[key]:
            assert r["lunged"] == 0, f"{key} at {r['loss']:.0%} loss lunged"
            assert r["max_jump"] <= step + 1e-6, (
                f"{key} at {r['loss']:.0%}: jump {r['max_jump']:.2f} m "
                f"> one tick {step:.2f} m")


def test_the_gate_alone_stalls_the_mission(swept):
    """The other half of the finding, and the reason the gate is not sufficient
    on its own: a stream that keeps advancing away from a frozen vehicle is
    rejected forever."""
    assert swept["stale_gated"][-1]["progress"] < 0.5
    assert swept["stale_gated"][-1]["worst_gap"] > 20


def test_replanning_plus_the_gate_gives_both(swept):
    """The configuration the FSM already implements: every stream is compiled
    from `self.pos`, never from where a previous plan hoped the vehicle was."""
    for r in swept["replan_gated"]:
        assert r["progress"] > 0.98, (
            f"{r['loss']:.0%} loss: only {r['progress']:.0%} progress")
        assert r["lunged"] == 0


def test_slew_gate_budget_is_per_tick_not_per_elapsed_second():
    """Regression on the first version of the budget, which paid a held vehicle
    for waiting and thereby licensed exactly the lunge it existed to stop."""
    gate = SlewGate(v_max=0.9, dt=1.0, tolerance_m=0.25)
    assert gate.budget == pytest.approx(1.15)

    est = EstimateSnapshot(0, 0, [(0.0, 0.0)], [0.001])
    ok = Plan("p", 0, [Assignment(0, (0.9, 0.0))])
    bad = Plan("p", 0, [Assignment(0, (7.2, 0.0))])
    assert gate.check(ok, est) == []
    v = gate.check(bad, est)
    assert v and v[0]["kind"] == "SlewTooLarge"


def test_disabled_gate_checks_nothing():
    est = EstimateSnapshot(0, 0, [(0.0, 0.0)], [0.001])
    plan = Plan("p", 0, [Assignment(0, (500.0, 0.0))])
    assert SlewGate(enabled=False).check(plan, est) == []


# --------------------------------------------------------------------------
# The link model itself
# --------------------------------------------------------------------------
def test_link_drops_at_roughly_the_requested_rate():
    rng = np.random.default_rng(0)
    link = LinkModel(loss=0.25, delay_ms=50, jitter_ms=15, reorder=0.0)
    n = 20_000
    dropped = sum(link.deliver(i * 10.0, rng) is None for i in range(n))
    assert 0.23 < dropped / n < 0.27


def test_link_delay_is_centred_on_the_requested_delay():
    rng = np.random.default_rng(1)
    link = LinkModel(loss=0.0, delay_ms=50.0, jitter_ms=15.0, reorder=0.0)
    d = np.array([link.deliver(0.0, rng) for _ in range(20_000)])
    assert 48.0 < d.mean() < 52.0
    assert 13.0 < d.std() < 17.0


def test_lossless_link_delivers_everything():
    rng = np.random.default_rng(2)
    link = LinkModel(loss=0.0)
    assert all(link.deliver(t * 10.0, rng) is not None for t in range(500))


def test_arrival_time_is_never_before_send():
    rng = np.random.default_rng(3)
    link = LinkModel(loss=0.0, delay_ms=5.0, jitter_ms=50.0, reorder=0.5)
    for t in range(2000):
        a = link.deliver(t * 10.0, rng)
        assert a is None or a >= t * 10.0


# --------------------------------------------------------------------------
# Sweep plumbing
# --------------------------------------------------------------------------
def test_sweep_covers_the_plans_range(swept):
    assert swept["loss"].min() == 0.0
    assert swept["loss"].max() == pytest.approx(0.30)
    for key in CONFIGS:
        assert len(swept[key]) == len(swept["loss"])


def test_result_reports_emit_rate():
    r, _ = _run(0.0, gated=True, replan=True)
    assert isinstance(r, SafeHoldResult)
    assert 0.0 <= r.emit_rate <= 1.0
    assert r.accepted + r.held == r.ticks
