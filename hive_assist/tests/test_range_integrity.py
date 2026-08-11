"""H3.2 — the range mesh is the trust root, so prove the monitor earns its place.

The load-bearing test here is `test_the_asymmetry_is_real`: a symmetric filter
would either reject ordinary NLOS or accept a distance-reduction attack, and the
whole design rests on treating those two directions differently.
"""

from __future__ import annotations

import numpy as np
import pytest

from security.audit_log import AuditLog
from security.range_integrity import RangeIntegrityConfig, RangeMonitor

ANCHORS = np.array([[-25.0, -25.0], [25.0, -25.0],
                    [25.0, 25.0], [-25.0, 25.0]])


class Frame:
    """Minimal stand-in for `sim.range_world.RangeFrame`.

    The monitor only needs `.inter` and `.anchor`, which is deliberate — the
    security layer must not import from `sim/`, so that it still works in a
    deployment where the range simulator does not exist.
    """

    def __init__(self, inter=None, anchor=None):
        self.inter = list(inter or [])
        self.anchor = list(anchor or [])


def truth_frame(pos, noise=0.0, rng=None):
    """Exact (or lightly noised) ranges for a fleet at `pos`."""
    pos = np.asarray(pos, dtype=float).reshape(-1, 2)
    rng = rng or np.random.default_rng(0)
    anchor, inter = [], []
    for i, p in enumerate(pos):
        for k, a in enumerate(ANCHORS):
            d = float(np.linalg.norm(p - a)) + (rng.normal(0, noise) if noise else 0.0)
            anchor.append((i, k, d))
    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            d = float(np.linalg.norm(pos[i] - pos[j]))
            inter.append((i, j, d + (rng.normal(0, noise) if noise else 0.0)))
    return Frame(inter, anchor)


@pytest.fixture()
def fleet():
    return np.array([[0.0, 0.0], [6.0, 0.0], [0.0, 6.0]])


# --------------------------------------------------------------------------
# baseline
# --------------------------------------------------------------------------
def test_clean_ranges_pass_untouched(fleet):
    mon = RangeMonitor(3, ANCHORS)
    f = truth_frame(fleet)
    v = mon.check(f, ref_pos=fleet, dt_s=0.1)
    assert v.n_dropped == 0
    assert len(v.anchor) == len(f.anchor)
    assert len(v.inter) == len(f.inter)


def test_realistic_noise_does_not_trip_it(fleet):
    """σ 1.5 cm thermal plus one-sided NLOS is the *expected* condition."""
    mon = RangeMonitor(3, ANCHORS)
    rng = np.random.default_rng(7)
    for _ in range(20):
        f = truth_frame(fleet, noise=0.015, rng=rng)
        # one-sided NLOS, exactly as sim/range_world models it
        f.anchor = [(i, k, d + abs(rng.normal(0, 0.05))) for i, k, d in f.anchor]
        v = mon.check(f, ref_pos=fleet, dt_s=0.1)
        assert v.n_dropped == 0, v.kinds()


# --------------------------------------------------------------------------
# THE asymmetry
# --------------------------------------------------------------------------
def test_the_asymmetry_is_real(fleet):
    """Same magnitude of error, opposite signs, opposite verdicts.

    +0.40 m is an ordinary multipath spike and must survive. −0.40 m has no
    benign explanation — radio does not take shortcuts — and is also the
    dangerous direction, because reduction drags the estimate toward the
    attacker. A symmetric filter cannot express this.
    """
    cfg = RangeIntegrityConfig()
    assert cfg.max_short_m < cfg.max_long_m, "the asymmetry is the design"

    long_mon = RangeMonitor(3, ANCHORS, cfg)
    f = truth_frame(fleet)
    f.anchor[0] = (0, 0, f.anchor[0][2] + 0.40)
    assert long_mon.check(f, ref_pos=fleet).n_dropped == 0

    short_mon = RangeMonitor(3, ANCHORS, cfg)
    f2 = truth_frame(fleet)
    f2.anchor[0] = (0, 0, f2.anchor[0][2] - 0.40)
    v = short_mon.check(f2, ref_pos=fleet)
    assert v.n_dropped == 1
    assert v.violations[0].kind == "residual"
    assert v.violations[0].detail["side"] == "short"


def test_distance_reduction_attack_is_dropped_not_merely_flagged(fleet):
    mon = RangeMonitor(3, ANCHORS)
    f = truth_frame(fleet)
    spoofed = (0, 2, f.anchor[2][2] - 3.0)
    f.anchor[2] = spoofed
    v = mon.check(f, ref_pos=fleet)
    assert spoofed not in v.anchor, "a corrupted link reached the estimator"


# --------------------------------------------------------------------------
# rate — works with no position estimate at all
# --------------------------------------------------------------------------
def test_rate_spike_caught_without_any_reference_position(fleet):
    """Rate in isolation: only two anchor links, so the geometry check cannot
    run and cannot be the thing that catches it."""
    mon = RangeMonitor(3, ANCHORS)
    first = truth_frame(fleet)
    first.anchor = [r for r in first.anchor if r[0] == 0 and r[1] < 2]
    mon.check(first, ref_pos=None, dt_s=0.1)

    f = truth_frame(fleet)
    f.anchor = [r for r in f.anchor if r[0] == 0 and r[1] < 2]
    f.anchor[0] = (0, 0, f.anchor[0][2] + 12.0)     # 12 m in one 100 ms tick
    v = mon.check(f, ref_pos=None, dt_s=0.1)
    assert [x.kind for x in v.violations] == ["rate"]


def test_geometry_preempts_rate_when_it_can_attribute(fleet):
    """Ordering is intentional: when both would fire, prefer the check that
    names the specific link over the one that only says 'something moved'."""
    mon = RangeMonitor(3, ANCHORS)
    mon.check(truth_frame(fleet), ref_pos=None, dt_s=0.1)
    f = truth_frame(fleet)
    f.anchor[0] = (0, 0, f.anchor[0][2] + 12.0)
    v = mon.check(f, ref_pos=None, dt_s=0.1)
    assert [x.kind for x in v.violations] == ["geometry"]
    assert v.violations[0].peer == 0


def test_legal_motion_does_not_trip_the_rate_check():
    mon = RangeMonitor(1, ANCHORS)
    p = np.array([[0.0, 0.0]])
    mon.check(truth_frame(p), ref_pos=None, dt_s=0.1)
    for k in range(1, 15):                           # 4 m/s, inside v_max
        p2 = np.array([[0.4 * k, 0.0]])
        v = mon.check(truth_frame(p2), ref_pos=None, dt_s=0.1)
        assert v.n_dropped == 0, v.kinds()


# --------------------------------------------------------------------------
# geometry — the cold-start defence
# --------------------------------------------------------------------------
def test_geometry_identifies_the_corrupted_link_with_no_prior_estimate():
    """Four anchors give the redundancy needed to name the liar, not just
    notice the lie — and this works on the very first frame, which is when an
    attacker most wants to be believed."""
    pos = np.array([[3.0, 4.0]])
    mon = RangeMonitor(1, ANCHORS)
    f = truth_frame(pos)
    f.anchor[1] = (0, 1, f.anchor[1][2] + 6.0)
    v = mon.check(f, ref_pos=None, dt_s=0.1)

    blamed = [x for x in v.violations if x.kind == "geometry"]
    assert len(blamed) == 1
    assert blamed[0].peer == 1, "blamed the wrong anchor"
    assert all(rec[1] != 1 for rec in v.anchor)


def test_three_links_detects_but_refuses_to_blame():
    """With no redundancy left after removing one, attribution is impossible.

    Not merely hard — impossible. Three ranges and two unknowns means dropping
    any one of them leaves an exactly-determined system that fits *perfectly*,
    so every candidate looks equally guilty (measured: all three leave-one-out
    fits give rms 0.0000). Naming one would be picking whichever the loop
    visited first. An honest 'something here is wrong' beats a confident wrong
    answer, so the vehicle is flagged and nothing is dropped.
    """
    pos = np.array([[3.0, 4.0]])
    mon = RangeMonitor(1, ANCHORS)
    f = truth_frame(pos)
    f.anchor = f.anchor[:3]
    f.anchor[1] = (0, 1, f.anchor[1][2] + 12.0)
    v = mon.check(f, ref_pos=None, dt_s=0.1)

    kinds = [x.kind for x in v.violations]
    assert "geometry_inconsistent" in kinds
    assert not any(x.kind == "geometry" for x in v.violations)
    assert len(v.anchor) == 3, "flagged, but nothing was dropped on a guess"


def test_three_anchors_have_a_measured_blind_spot():
    """The limitation, pinned with the number, so it is not rediscovered later.

    One DOF of redundancy absorbs a surprising amount: at 3 anchors a +6 m
    corruption produces rms 0.507 m and passes the geometry gate entirely.
    ~+12 m (rms 1.68) is where detection begins. This is the concrete argument
    for four anchor masts rather than three — the same conclusion the D1
    null-space ladder reached from observability, arrived at independently
    from integrity.
    """
    pos = np.array([[3.0, 4.0]])
    mon = RangeMonitor(1, ANCHORS)
    f = truth_frame(pos)
    f.anchor = f.anchor[:3]
    f.anchor[1] = (0, 1, f.anchor[1][2] + 6.0)
    v = mon.check(f, ref_pos=None, dt_s=0.1)
    assert v.n_dropped == 0, "if this now fires, the threshold changed"


def test_consistent_ranges_are_never_blamed():
    pos = np.array([[3.0, 4.0]])
    mon = RangeMonitor(1, ANCHORS)
    for _ in range(5):
        v = mon.check(truth_frame(pos), ref_pos=None, dt_s=0.1)
        assert v.n_dropped == 0


# --------------------------------------------------------------------------
# triangle inequality — needs no positions whatsoever
# --------------------------------------------------------------------------
def test_impossible_triangle_is_caught():
    """A fabricated mesh whose links are individually plausible.

    6 and 6 cannot bound a 40. No anchor, no estimate, no geometry solve
    involved — pure internal contradiction.
    """
    mon = RangeMonitor(3, ANCHORS)
    f = Frame(inter=[(0, 1, 6.0), (0, 2, 6.0), (1, 2, 40.0)])
    v = mon.check(f, ref_pos=None, dt_s=0.1)
    assert [x.kind for x in v.violations] == ["triangle"]
    assert (1, 2, 40.0) not in v.inter


def test_valid_triangle_survives(fleet):
    mon = RangeMonitor(3, ANCHORS)
    f = Frame(inter=truth_frame(fleet).inter)
    assert mon.check(f, ref_pos=None, dt_s=0.1).n_dropped == 0


# --------------------------------------------------------------------------
# plumbing
# --------------------------------------------------------------------------
def test_violations_reach_the_audit_log(tmp_path, fleet):
    log = AuditLog(tmp_path / "sec.jsonl", b"k" * 32, sync=False)
    mon = RangeMonitor(3, ANCHORS, audit=log)
    f = truth_frame(fleet)
    f.anchor[0] = (0, 0, f.anchor[0][2] - 3.0)
    mon.check(f, ref_pos=fleet)

    v = log.verify()
    assert v.ok and v.kinds.get("range_rejected") == 1


def test_report_counts_by_kind(fleet):
    mon = RangeMonitor(3, ANCHORS)
    f = truth_frame(fleet)
    f.anchor[0] = (0, 0, f.anchor[0][2] - 0.40)   # under the geometry gate
    mon.check(f, ref_pos=fleet)
    r = mon.report()
    assert r["dropped"] == 1 and r["by_kind"]["residual"] == 1


def test_a_rejected_link_does_not_poison_the_rate_baseline(fleet):
    """Same ratchet defence as the estimate gate: a dropped measurement must
    not become the reference the next one is compared against."""
    mon = RangeMonitor(3, ANCHORS)
    good = truth_frame(fleet)
    mon.check(good, ref_pos=fleet)
    base = mon._last_anchor[(0, 0)]

    bad = truth_frame(fleet)
    bad.anchor[0] = (0, 0, base - 5.0)
    mon.check(bad, ref_pos=fleet)
    assert mon._last_anchor[(0, 0)] == base
