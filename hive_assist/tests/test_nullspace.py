"""D1.2 acceptance: dim ker(H) is what the anchor configuration says it is.

The plan's deliverable was "show dim ker(H): 3 with anchors removed -> 0 with the
single anchor factor present". These tests do that, and also lock in the two
corrections the numbers forced:

  * a single range-only anchor leaves dim ker(H) = 1, not 0, no matter how much
    the swarm flies — rotation about the anchor is an exact symmetry
  * a bearing only removes it if measured in the anchor's *surveyed* frame

Both are asserted as properties, not just as rank counts, so a future change that
accidentally breaks gauge invariance (the classic way to get a flattering rank)
fails here rather than silently making the estimator look better than it is.
"""

import math

import numpy as np
import pytest

from hive.anchor_factor import (
    AnchorBearingFactor,
    AnchorRangeFactor,
    InterAgentBearingFactor,
    InterAgentRangeFactor,
    OdometryFactor,
    PriorFactor,
    gauge_generators,
    linearize_all,
    state_dim,
)
from hive.nullspace import (
    CONFIGS,
    RANK_TOL,
    Scenario,
    centroid_uncertainty,
    kernel_basis,
    rank_report,
    residual_energy,
)


@pytest.fixture(scope="module")
def scn():
    return Scenario()


# --------------------------------------------------------------------------
# Jacobians: everything below is meaningless if these are wrong
# --------------------------------------------------------------------------
def _fd_jacobian(f, x, nk, n, eps=1e-6):
    j = np.zeros((f.dim(), n))
    for k in range(n):
        xp, xm = x.copy(), x.copy()
        xp[k] += eps
        xm[k] -= eps
        j[:, k] = (np.atleast_1d(f.residual(xp, nk))
                   - np.atleast_1d(f.residual(xm, nk))) / (2 * eps)
    return j


@pytest.mark.parametrize("factor", [
    OdometryFactor(sigma=0.05, agent=1, t0=2,
                   delta_body=np.array([0.3, -0.2]), delta_theta=0.1),
    InterAgentRangeFactor(sigma=0.1, a=0, b=2, t=1, meas=1.7),
    InterAgentBearingFactor(sigma=0.02, a=2, b=1, t=3, meas=0.4),
    AnchorRangeFactor(sigma=0.08, anchor=np.array([1.3, -0.7]), agent=1, t=0,
                      meas=2.2),
    AnchorBearingFactor(sigma=0.02, anchor=np.array([1.3, -0.7]), agent=0, t=2,
                        meas=-0.3, frame="anchor"),
    AnchorBearingFactor(sigma=0.02, anchor=np.array([1.3, -0.7]), agent=2, t=1,
                        meas=0.9, frame="body"),
    PriorFactor(sigma=1.0, agent=0, t=0, sigma_theta=1.0),
], ids=lambda f: f"{type(f).__name__}{getattr(f, 'frame', '')}")
def test_analytic_jacobian_matches_finite_difference(factor):
    rng = np.random.default_rng(0)
    m, t = 3, 4
    n = state_dim(m, t)
    x = rng.normal(scale=2.0, size=n)
    err = np.abs(factor.jacobian(x, t, n) - _fd_jacobian(factor, x, t, n)).max()
    assert err < 1e-7


def test_bad_bearing_frame_rejected():
    with pytest.raises(ValueError):
        AnchorBearingFactor(anchor=np.zeros(2), frame="world")


# --------------------------------------------------------------------------
# The ladder
# --------------------------------------------------------------------------
@pytest.mark.parametrize("config", list(CONFIGS))
def test_kernel_dimension_matches_expectation(scn, config):
    r = rank_report(scn, config)
    assert r["kernel_dim"] == r["expected"], (
        f"{r['label']}: got dim ker(H) = {r['kernel_dim']}, "
        f"expected {r['expected']}"
    )


@pytest.mark.parametrize("config", list(CONFIGS))
def test_rank_decision_is_not_threshold_sensitive(scn, config):
    """The rank claim must survive moving the threshold by orders of magnitude,
    or it is a numerical artefact rather than a structural result."""
    r = rank_report(scn, config)
    assert r["gap"] > 1e6, f"{r['label']}: singular-value gap only {r['gap']:.1e}"
    assert r["largest_dropped_sv"] < RANK_TOL * 1e-3 or r["kernel_dim"] == 0


def test_headline_claim_three_to_zero(scn):
    """The plan's original sentence, verbatim: 3 without the anchor, 0 with it."""
    assert rank_report(scn, "no_anchor")["kernel_dim"] == 3
    assert rank_report(scn, "mesh_anchor_bearing")["kernel_dim"] == 0


# --------------------------------------------------------------------------
# Gauge invariance of the base graph — the property the result rests on
# --------------------------------------------------------------------------
def test_base_graph_is_exactly_gauge_invariant(scn):
    """Odometry + inter-agent ranges must be blind to a global SE(2) shift.

    If any of them leaks a global reference, the anchor gets credit for
    information it did not supply and the whole study is worthless.
    """
    x = scn.truth()
    gens = gauge_generators(x, scn.n_agents, scn.n_keyframes)
    _, j = linearize_all(scn.factors("no_anchor"), x, scn.n_agents,
                         scn.n_keyframes)
    scale = np.linalg.norm(j)
    for name, g in zip(("tx", "ty", "yaw"), gens):
        leak = np.linalg.norm(j @ g) / (np.linalg.norm(g) * scale)
        assert leak < 1e-12, f"{name} generator leaks into the base graph"


def test_inter_agent_bearing_is_also_gauge_invariant(scn):
    """Body-frame inter-agent bearings are optional in the plan; check they do
    not smuggle in a global yaw if enabled."""
    x = scn.truth()
    facs = scn.factors("no_anchor")
    for t in range(scn.n_keyframes):
        for i in range(scn.n_agents):
            j_ = (i + 1) % scn.n_agents
            from hive.anchor_factor import pose, rot
            pi, thi = pose(x, i, t, scn.n_keyframes)
            pj, _ = pose(x, j_, t, scn.n_keyframes)
            d = rot(thi).T @ (pj - pi)
            facs.append(InterAgentBearingFactor(
                sigma=0.02, a=i, b=j_, t=t, meas=math.atan2(d[1], d[0])))

    _, j = linearize_all(facs, x, scn.n_agents, scn.n_keyframes)
    gens = gauge_generators(x, scn.n_agents, scn.n_keyframes)
    scale = np.linalg.norm(j)
    for g in gens:
        assert np.linalg.norm(j @ g) / (np.linalg.norm(g) * scale) < 1e-12


# --------------------------------------------------------------------------
# Naming the surviving freedom, not just counting it
# --------------------------------------------------------------------------
def test_range_only_leaves_exactly_rotation_about_the_anchor(scn):
    x = scn.truth()
    g = gauge_generators(x, scn.n_agents, scn.n_keyframes, centre=scn.anchor)[2]
    k = kernel_basis(scn, "mesh_motion")

    assert k.shape[0] == 1
    alignment = float(np.linalg.norm(k @ (g / np.linalg.norm(g))))
    assert alignment == pytest.approx(1.0, abs=1e-9), (
        "the surviving direction is not rotation about the anchor"
    )


def test_rotation_about_the_map_origin_is_not_free(scn):
    """The pivot is the surveyed object, not the coordinate origin. If this
    passed, the result would be about our choice of frame, not about the
    anchor."""
    x = scn.truth()
    g_origin = gauge_generators(x, scn.n_agents, scn.n_keyframes)[2]
    assert residual_energy(scn, "mesh_motion", g_origin) > 1.0


def test_translation_is_not_free_once_the_anchor_ranges(scn):
    x = scn.truth()
    gens = gauge_generators(x, scn.n_agents, scn.n_keyframes)
    for g in gens[:2]:
        assert residual_energy(scn, "mesh_motion", g) > 1.0
        assert residual_energy(scn, "no_anchor", g) < 1e-10


# --------------------------------------------------------------------------
# The two corrections, asserted directly
# --------------------------------------------------------------------------
def test_motion_baseline_does_not_rescue_yaw(scn):
    """More flying, more range data, same kernel. This is the plan's open
    question answered: the motion-baseline route to global yaw does not exist
    for a single anchor."""
    static = rank_report(scn, "mesh_static")["kernel_dim"]
    moving = rank_report(scn, "mesh_motion")["kernel_dim"]
    assert static == moving == 1

    # and the extra keyframes really did add rows, so this is not a no-op
    assert (rank_report(scn, "mesh_motion")["n_factor_rows"]
            > rank_report(scn, "mesh_static")["n_factor_rows"])


def test_baseline_is_what_recovers_translation(scn):
    """What motion *does* buy: one agent, one keyframe leaves 2 free; adding a
    baseline — temporal or spatial — takes it to 1."""
    assert rank_report(scn, "one_agent_static")["kernel_dim"] == 2
    assert rank_report(scn, "one_agent_motion")["kernel_dim"] == 1   # temporal
    assert rank_report(scn, "mesh_static")["kernel_dim"] == 1        # spatial


def test_bearing_only_counts_in_the_anchors_own_frame(scn):
    """A vehicle-side AoA measurement of the anchor is invariant to rotation
    about the anchor, so it cannot remove that rotation. A surveyed ground
    station's bearing can."""
    assert rank_report(scn, "mesh_body_bearing")["kernel_dim"] == 1
    assert rank_report(scn, "mesh_anchor_bearing")["kernel_dim"] == 0


def test_second_surveyed_anchor_also_works(scn):
    """The bearing-free alternative, for sites where surveying a heading is
    harder than surveying a second point."""
    assert rank_report(scn, "two_anchors_range")["kernel_dim"] == 0


# --------------------------------------------------------------------------
# What it costs in metres
# --------------------------------------------------------------------------
def test_range_only_is_pinned_radially_and_free_tangentially(scn):
    v = centroid_uncertainty(scn, "mesh_motion")
    assert v["radial_m"] < 0.10, "anchor ranges should pin the radial direction"
    assert v["tangential_m"] > 100.0, "tangential must be prior-limited"


def test_surveyed_bearing_bounds_both_directions(scn):
    for cfg in ("mesh_anchor_bearing", "two_anchors_range"):
        v = centroid_uncertainty(scn, cfg)
        assert v["radial_m"] < 0.10
        assert v["tangential_m"] < 0.10, f"{cfg} left the swarm free to swing"


def test_anchorless_is_unbounded_in_both(scn):
    v = centroid_uncertainty(scn, "no_anchor")
    assert v["radial_m"] > 100.0 and v["tangential_m"] > 100.0
