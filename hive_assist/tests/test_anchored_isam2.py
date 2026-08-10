"""D1.3 acceptance: the anchor holds the estimate to the world over a mission.

`test_nullspace.py` settles the rank question at one instant. These tests are
about what happens over 120 keyframes, which is where the plan's actual claim
lives — that a surveyed anchor beats reference-drone pinning not on conditioning
but on *drift*.

The run is short (a few seconds) so it stays in the ordinary test suite rather
than becoming a thing nobody runs.
"""

import numpy as np
import pytest

from hive.anchored_isam2 import AnchoredWorld, run


@pytest.fixture(scope="module")
def runs():
    world = AnchoredWorld(n_keyframes=90)
    frames = world.measurements()          # identical data for both modes
    return world, {m: run(world, m, frames) for m in ("pinned", "anchored")}


SETTLE = 20        # keyframes of cold start to step past before judging drift


def test_anchored_tracks_truth(runs):
    _, r = runs
    assert r["anchored"]["final_rmse"] < 0.25


def test_pinned_drifts_away(runs):
    """Not a bug in the pinned run — the expected behaviour of an estimate whose
    only absolute reference is a single prior receding into the past."""
    _, r = runs
    assert r["pinned"]["final_rmse"] > 4 * r["anchored"]["final_rmse"]


def test_pinned_drift_is_monotone_not_noise(runs):
    """A drift, not a diffusion: compare the mean error early vs late, past the
    cold start. Noise would leave these roughly equal."""
    _, r = runs
    e = r["pinned"]["rmse"]
    early = e[SETTLE:SETTLE + 20].mean()
    late = e[-20:].mean()
    assert late > 2.0 * early


def test_anchored_does_not_drift(runs):
    _, r = runs
    e = r["anchored"]["rmse"]
    early = e[SETTLE:SETTLE + 20].mean()
    late = e[-20:].mean()
    assert late < 1.6 * early, "anchored run should be flat, not creeping"


def test_pinned_becomes_overconfident(runs):
    """The consequence that matters downstream: the pinned estimator's reported
    sigma stops tracking its real error, so the Rust supervisor's covariance
    gate would happily command a vehicle that is a metre out of place."""
    _, r = runs
    p = r["pinned"]
    ratio = p["rmse"][-1] / p["sigma"][-1]
    assert ratio > 2.5, f"expected overconfidence, got ratio {ratio:.2f}"


def test_anchored_stays_calibrated(runs):
    _, r = runs
    a = r["anchored"]
    ratio = a["rmse"][SETTLE:] / a["sigma"][SETTLE:]
    assert ratio.mean() < 2.5, "anchored estimate should roughly know its own error"


def test_anchored_sigma_is_bounded(runs):
    """The covariance claim from the plan, over time rather than at an instant."""
    _, r = runs
    s = r["anchored"]["sigma"][SETTLE:]
    assert s.max() < 0.20
    assert s[-10:].mean() < 1.5 * s[:10].mean()


def test_incremental_cost_stays_bounded(runs):
    """iSAM2's whole reason for being: per-keyframe cost must not grow with
    mission length, or none of this flies in real time."""
    _, r = runs
    ms = r["anchored"]["update_ms"][SETTLE:]
    first, last = ms[:20].mean(), ms[-20:].mean()
    assert last < 3.0 * first, f"update cost grew {last/first:.1f}x"


def test_both_modes_use_identical_measurements(runs):
    """The comparison is only meaningful if the anchor is the sole difference."""
    world, _ = runs
    a = world.measurements()
    b = world.measurements()
    assert len(a) == len(b)
    assert np.allclose([m for _, _, m in a[5]["range"]],
                       [m for _, _, m in b[5]["range"]])


def test_unknown_mode_rejected(runs):
    world, _ = runs
    with pytest.raises(ValueError):
        run(world, "freestyle")
