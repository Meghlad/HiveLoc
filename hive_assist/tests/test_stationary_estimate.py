"""D4.8 — the regression `done_till_now` §7.3 named as the missing one.

    "Add a VIO regression test: a stationary vehicle must produce near-zero
     relative pose. That single assertion would have caught this."

The VIO front end is gone, so the assertion is retargeted at what replaced it:
a stationary vehicle, localised by ranges alone, must produce a near-zero
relative pose, must not drift, and — the part that actually matters — must
report a sigma that agrees with its error.

WHY THE RATIO IS THE TEST AND THE ERROR IS NOT. The failure this file exists to
catch was never "the number is large". It was 14.9 m of error reported as 10 cm
of confidence: a ratio of 147, on an estimator whose covariance the supervisor's
trust gate reads and would have certified. A test that asserted only
`error < tol` would have passed the whole way down that road, because the error
was small right up until it was not and the confidence never moved. So the bar
here is two-sided — the estimate must be accurate AND it must know how accurate
it is. An estimator that is quietly under-confident fails these tests too, and
that is deliberate: a sigma nobody can trust in either direction is not a sigma.

These tests run entirely offline. No SITL, no sockets, no GPU. They exercise the
real `RangeWorld` and the real `LiveEstimator` — the same objects the live loop
constructs — so a regression in the flight path fails here first.
"""

from __future__ import annotations

import numpy as np
import pytest

from sim.live_estimator import (
    EstimatorConfig,
    LiveEstimator,
    error_over_sigma,
    multilaterate,
)
from sim.range_world import DEFAULT_ANCHORS, RangeConfig, RangeWorld, UwbModel

SQUARE = np.array([[-4.0, -4.0], [4.0, -4.0], [4.0, 4.0], [-4.0, 4.0]])


def run_static(ticks: int = 150, truth=SQUARE, seed: int = 11,
               cfg: EstimatorConfig | None = None,
               range_cfg: RangeConfig | None = None):
    """Hold a formation perfectly still and estimate it from ranges alone."""
    world = RangeWorld(DEFAULT_ANCHORS, range_cfg, seed=seed)
    est = LiveEstimator(len(truth), DEFAULT_ANCHORS, cfg)
    track = np.zeros((ticks, len(truth), 2))
    for k in range(ticks):
        est.step(world.measure(truth))
        track[k] = est.pos
    return est, track


# --------------------------------------------------------------------------
# The named regression
# --------------------------------------------------------------------------
def test_a_stationary_vehicle_produces_near_zero_relative_pose():
    """The single assertion that would have caught the 7 m VIO lie.

    Frame-to-frame displacement of an airframe that never moved. The old front
    end reported 7.238 m of motion here at full weight while the vehicle sat
    still; ranges cannot do that, because a range to a fixed anchor from a fixed
    point is a constant plus noise and there is nothing to integrate.
    """
    est, track = run_static(ticks=150)
    steps = np.linalg.norm(np.diff(track[20:], axis=0), axis=2)
    assert steps.max() < 0.10, (
        f"stationary vehicle reported {steps.max():.3f} m of frame-to-frame "
        f"motion; the regression this test exists for was 7.238 m")
    # The mean measures ~2.0 cm, which is not zero and should not be: the
    # estimate is a noisy quantity and it re-solves every keyframe, so it
    # dithers at roughly its own uncertainty. The bound is therefore set from
    # the estimator's reported sigma rather than from a round number — an
    # estimate that jitters far beyond what it claims to know is the failure,
    # and a fixed 2 cm threshold would have been a coin flip against a measured
    # 2.006 cm.
    assert steps.mean() < 4.0 * est.sigma_m().mean(), (
        f"mean step {steps.mean() * 100:.2f} cm against a reported sigma of "
        f"{est.sigma_m().mean() * 100:.2f} cm")


def test_a_stationary_swarm_does_not_drift():
    """Accumulated displacement over the whole run, not per-step.

    Per-step motion can be small while the estimate walks steadily away — that
    is exactly what a pinned (rather than anchored) estimator does, and
    `hive/anchored_isam2.py` demonstrates it offline. The anchor is supposed to
    make it impossible here.
    """
    _, track = run_static(ticks=200)
    drift = np.linalg.norm(track[-1] - track[20], axis=1)
    assert drift.max() < 0.10, f"estimate drifted {drift.max():.3f} m at rest"


def test_error_and_sigma_agree_at_rest():
    """error / sigma near 1: the plan's acceptance bar (§4.6), both-sided.

    Above the upper bound is a confidently-wrong estimate, which the supervisor
    would certify. Below the lower bound the estimator is under-claiming, which
    wastes the covariance gate's whole purpose — a sigma that is always
    pessimistic never excludes anything.
    """
    est, _ = run_static(ticks=150)
    ratio = error_over_sigma(est.pos, SQUARE, est.sigma_m())
    assert np.all(np.isfinite(ratio))
    assert 0.2 < ratio.mean() < 4.0, (
        f"error/sigma = {ratio.mean():.2f}; the calibration claim is ~1")


def test_absolute_error_stays_centimetric_at_rest():
    """The accuracy half of the claim, held separately from the calibration
    half so a regression in either one is legible on its own."""
    est, _ = run_static(ticks=150)
    err = np.linalg.norm(est.pos - SQUARE, axis=1)
    assert err.max() < 0.15, f"worst vehicle {err.max():.3f} m from truth"


def test_reported_sigma_is_finite_and_small_at_rest():
    est, _ = run_static(ticks=150)
    sigma = est.sigma_m()
    assert np.all(np.isfinite(sigma))
    assert np.all(sigma < 0.20)
    assert np.all(est.cov_trace < 0.02), (
        "trace exceeds the supervisor's max_cov_trace at rest; the gate would "
        "refuse to command a vehicle that is standing perfectly still")


# --------------------------------------------------------------------------
# The estimate has to survive the radio it was promised
# --------------------------------------------------------------------------
def test_survives_heavy_dropout():
    """30% dropout. The safety prior and the motion factor carry the gaps."""
    cfg = RangeConfig(uwb=UwbModel(p_dropout=0.30))
    est, track = run_static(ticks=150, range_cfg=cfg)
    err = np.linalg.norm(est.pos - SQUARE, axis=1)
    assert np.all(np.isfinite(track)), "estimate went non-finite under dropout"
    assert err.max() < 0.40


def test_outliers_do_not_move_the_estimate_much():
    """The Huber kernel is load-bearing, not decorative.

    10x the outlier rate. Without a robust kernel a 0.5 m multipath spike enters
    at a 1.5 cm noise floor and drags the solution; that is precisely how the
    old stack fused 7 m of garbage "at a 2 cm noise floor with no robust
    kernel".
    """
    clean, _ = run_static(ticks=150)
    noisy, _ = run_static(ticks=150,
                          range_cfg=RangeConfig(uwb=UwbModel(p_outlier=0.30)))
    moved = np.linalg.norm(noisy.pos - clean.pos, axis=1)
    assert moved.max() < 0.30, (
        f"a 10x outlier rate moved the estimate {moved.max():.3f} m; the Huber "
        f"kernel is not doing its job")


def test_a_vehicle_with_no_links_is_reported_as_untrusted_not_confident():
    """The one behaviour that must never regress.

    A vehicle nothing can see must NOT come back with a small covariance. The
    project's thesis is that a confidently wrong estimate is worse than no
    estimate, so 'I do not know' has to be representable and has to be what
    comes out.
    """
    truth = np.array([[0.0, 0.0], [400.0, 400.0]])   # #1 far outside every radius
    world = RangeWorld(DEFAULT_ANCHORS, seed=3)
    est = LiveEstimator(2, DEFAULT_ANCHORS)
    for _ in range(40):
        frame = world.measure(truth)
        assert all(i == 0 for i, _, _ in frame.anchor)
        assert not frame.inter
        est.step(frame)
    assert est.cov_trace[0] < 0.02, "the connected vehicle should be trusted"
    assert est.cov_trace[1] > est.cov_trace[0] * 50, (
        "an unobserved vehicle reported a covariance comparable to an observed "
        "one")


# --------------------------------------------------------------------------
# Cold start and the fixed-lag seam
# --------------------------------------------------------------------------
def test_cold_start_needs_no_initial_guess():
    """The estimator earns its first fix from anchor ranges, not from truth."""
    est, _ = run_static(ticks=30)
    err = np.linalg.norm(est.pos - SQUARE, axis=1)
    assert err.max() < 0.30, (
        f"cold start settled {err.max():.3f} m from truth in 30 keyframes")


def test_multilateration_recovers_a_known_point():
    p = np.array([3.0, -7.0])
    ranges = [(k, float(np.linalg.norm(p - a)))
              for k, a in enumerate(DEFAULT_ANCHORS)]
    assert np.linalg.norm(multilaterate(DEFAULT_ANCHORS, ranges) - p) < 1e-6


def test_the_fixed_lag_rebuild_does_not_move_the_estimate():
    """The seam has to be invisible in the output, not merely documented.

    A rebuild drops cross-covariances with the discarded past. If that showed up
    as a step in the position, the loop would be commanding a jump every
    `window - retain` keyframes.
    """
    cfg = EstimatorConfig(window=12, retain=6)
    est, track = run_static(ticks=120, cfg=cfg)
    assert est.rebuilds >= 5, "the test did not actually exercise a rebuild"
    steps = np.linalg.norm(np.diff(track[20:], axis=0), axis=2)
    assert steps.max() < 0.10, (
        f"the largest single-tick move was {steps.max():.3f} m — if that lands "
        f"on a rebuild keyframe, the seam is visible in the output")


def test_the_graph_stays_bounded():
    """Runs long enough that an unbounded graph would show up as a slowdown."""
    cfg = EstimatorConfig(window=20, retain=10)
    est, _ = run_static(ticks=300, cfg=cfg)
    assert len(est._frames) <= cfg.window
    assert est.rebuilds >= 20


def test_window_must_exceed_retain():
    """retain == window rebuilds on every keyframe: 150 ms against a 4 ms
    steady state, measured. The config refuses it rather than being slow."""
    with pytest.raises(ValueError):
        EstimatorConfig(window=10, retain=10)


# --------------------------------------------------------------------------
# Motion, so "it only works standing still" cannot be true silently
# --------------------------------------------------------------------------
def test_a_moving_formation_stays_calibrated():
    """The failure the restructure was designed to delete.

    The old stack was calibrated at rest (ratio 0.78) and detonated in flight
    (ratio 147) because the odometry term was visual. The odometry term is now a
    constant-velocity prior over ranges, so this must hold while moving.
    """
    world = RangeWorld(DEFAULT_ANCHORS, seed=23)
    est = LiveEstimator(len(SQUARE), DEFAULT_ANCHORS)
    ratios, errs = [], []
    for k in range(180):
        truth = SQUARE + np.array([0.04 * k, 0.02 * k])      # 4 cm/tick drift
        est.step(world.measure(truth))
        if k > 30:
            ratios.append(error_over_sigma(est.pos, truth, est.sigma_m()).mean())
            errs.append(np.linalg.norm(est.pos - truth, axis=1).max())
    assert max(errs) < 0.40, f"worst in-motion error {max(errs):.3f} m"
    assert np.mean(ratios) < 8.0, (
        f"in-motion error/sigma = {np.mean(ratios):.1f}; the old stack's "
        f"in-flight number was 147 and that is the regression")
