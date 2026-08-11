"""H4.3 — an injected teleport is refused and the vehicle holds.

The plan's acceptance test for A2. The headline case is `test_teleport_is_refused`,
but the one that actually justifies the module is `test_confidently_wrong_*`: an
estimate that is wrong while *reporting that it is precise* is the failure mode
every downstream covariance gate is blind to, and the one this project's own
history produced without any attacker involved.
"""

from __future__ import annotations

import numpy as np
import pytest

from security.audit_log import AuditLog
from security.plausibility import EstimateGate, PlausibilityConfig

DT = 100.0          # ms per tick — 10 Hz, the estimator's keyframe rate
GOOD_TRACE = 2.0 * (0.017 ** 2)      # the live estimator's ~1.7 cm/axis sigma


def feed(gate: EstimateGate, track, trace=GOOD_TRACE, t0=1_000_000.0):
    """Push a sequence of fleet positions through the gate, one tick each."""
    out = []
    for k, pos in enumerate(track):
        stamp = t0 + k * DT
        out.append(gate.check(pos, [trace] * len(pos),
                              stamp_unix_ms=stamp, now_unix_ms=stamp))
    return out


def straight_line(n_ticks: int, v_mps: float = 1.0):
    """One vehicle flying east at a constant, legal speed."""
    return [np.array([[v_mps * (k * DT / 1000.0), 0.0]])
            for k in range(n_ticks)]


# --------------------------------------------------------------------------
# the baseline: honest estimates are not molested
# --------------------------------------------------------------------------
def test_a_normal_flight_is_never_rejected():
    gate = EstimateGate(1, PlausibilityConfig())
    verdicts = feed(gate, straight_line(40))
    assert all(v.trusted[0] for v in verdicts)
    assert gate.report()["total"] == 0


def test_a_hard_but_legal_manoeuvre_is_allowed():
    """The false-positive guard. A real vehicle accelerating must not trip."""
    cfg = PlausibilityConfig()
    gate = EstimateGate(1, cfg)
    track, p, v = [], np.zeros(2), np.zeros(2)
    for _ in range(30):
        v = np.minimum(v + np.array([cfg.a_max_mps2 * 0.9, 0.0]) * (DT / 1000.0),
                       cfg.v_max_mps)
        p = p + v * (DT / 1000.0)
        track.append(p.reshape(1, 2).copy())
    assert all(v_.trusted[0] for v_ in feed(gate, track))


# --------------------------------------------------------------------------
# H4.3 proper — the spoof
# --------------------------------------------------------------------------
def test_teleport_is_refused():
    gate = EstimateGate(1, PlausibilityConfig())
    track = straight_line(10)
    track.append(np.array([[400.0, 400.0]]))          # the injected jump
    verdicts = feed(gate, track)

    assert verdicts[-1].trusted[0] is np.False_ or not verdicts[-1].trusted[0]
    assert "teleport" in verdicts[-1].kinds() or "off_map" in verdicts[-1].kinds()


def test_a_teleport_inside_the_footprint_is_still_refused():
    """Not merely an `off_map` bounds check — the motion itself is impossible."""
    gate = EstimateGate(1, PlausibilityConfig())
    track = straight_line(10)
    last = track[-1][0, 0]
    track.append(np.array([[last + 25.0, 0.0]]))       # 25 m in 100 ms
    v = feed(gate, track)[-1]
    assert not v.trusted[0]
    assert v.kinds() == ["teleport"]
    assert v.violations[0].detail["implied_v_mps"] > 100.0


def test_the_vehicle_holds_rather_than_chasing_the_spoof():
    """The 'and holds' half of the plan's acceptance criterion.

    `publish()` re-sends the last good position for an untrusted vehicle, so
    'refused' has to mean the fan-out never sees the bad number. Here that is
    expressed as: the gate's own accepted history is unchanged by the spoof.
    """
    gate = EstimateGate(1, PlausibilityConfig())
    feed(gate, straight_line(10))
    before = gate._p1[0].copy()

    stamp = 1_000_000.0 + 10 * DT
    gate.check(np.array([[500.0, 500.0]]), [GOOD_TRACE],
               stamp_unix_ms=stamp, now_unix_ms=stamp)

    assert np.allclose(gate._p1[0], before), (
        "a rejected estimate became the baseline; an attacker can now walk the "
        "vehicle anywhere in individually-legal steps")


def test_ratcheting_walk_is_refused_not_absorbed():
    """The attack the history rule exists to stop.

    Repeated spoofs, each one a legal step from the *previous spoof* but not
    from the last accepted truth. If rejected estimates updated the baseline,
    every one of these would pass.
    """
    gate = EstimateGate(1, PlausibilityConfig())
    feed(gate, straight_line(10))
    t = 1_000_000.0 + 10 * DT

    refused = 0
    for k in range(1, 12):
        pos = np.array([[10.0 + 20.0 * k, 0.0]])
        v = gate.check(pos, [GOOD_TRACE], stamp_unix_ms=t + k * DT,
                       now_unix_ms=t + k * DT)
        refused += 0 if v.trusted[0] else 1
    assert refused == 11


# --------------------------------------------------------------------------
# the one that matters
# --------------------------------------------------------------------------
def test_confidently_wrong_estimate_is_caught_by_sigma_lie():
    """14.9 m of error reported as 10 cm of confidence — the project's own bug.

    Small enough to clear the teleport budget outright would be too easy; this
    sits inside the kinematic limit but far outside what the reported sigma
    claims, which is precisely the case a covariance gate certifies as good.
    """
    cfg = PlausibilityConfig()
    gate = EstimateGate(1, cfg)
    feed(gate, straight_line(12, v_mps=1.0))

    # 0.4 m off prediction in one 100 ms tick: v = 4 m/s, under the 5 m/s
    # ceiling, so `teleport` does NOT fire. But sigma says 1.7 cm.
    t = 1_000_000.0 + 12 * DT
    last = gate._p1[0].copy()
    v = gate.check(np.array([[last[0] + 0.5, 0.4]]), [GOOD_TRACE],
                   stamp_unix_ms=t, now_unix_ms=t)

    assert not v.trusted[0]
    assert v.kinds() == ["sigma_lie"]
    assert v.violations[0].detail["ratio"] > cfg.sigma_ratio_max


def test_an_honest_large_sigma_is_not_a_lie():
    """The check is about disagreement, not about magnitude.

    Same physical jump as the test above, but the estimator says it is unsure.
    That is an estimator behaving correctly and must not be punished — error and
    sigma agree, which is the project's actual acceptance criterion.
    """
    cfg = PlausibilityConfig(max_cov_trace=1.0)
    gate = EstimateGate(1, cfg)
    honest = 2.0 * (0.25 ** 2)
    feed(gate, straight_line(12, v_mps=1.0), trace=honest)

    t = 1_000_000.0 + 12 * DT
    last = gate._p1[0].copy()
    v = gate.check(np.array([[last[0] + 0.5, 0.4]]), [honest],
                   stamp_unix_ms=t, now_unix_ms=t)
    assert v.trusted[0], "an honestly-uncertain estimate was refused"


# --------------------------------------------------------------------------
# the remaining checks
# --------------------------------------------------------------------------
def test_nan_is_refused_before_anything_else():
    gate = EstimateGate(1)
    v = gate.check(np.array([[np.nan, 0.0]]), [GOOD_TRACE],
                   stamp_unix_ms=1e6, now_unix_ms=1e6)
    assert not v.trusted[0] and v.kinds() == ["not_finite"]


def test_stale_estimate_is_refused():
    gate = EstimateGate(1, PlausibilityConfig(max_age_ms=1000))
    v = gate.check(np.array([[0.0, 0.0]]), [GOOD_TRACE],
                   stamp_unix_ms=1e6, now_unix_ms=1e6 + 5000)
    assert not v.trusted[0] and v.kinds() == ["stale"]


def test_covariance_beyond_the_trust_bound_is_refused():
    gate = EstimateGate(1, PlausibilityConfig(max_cov_trace=0.02))
    v = gate.check(np.array([[0.0, 0.0]]), [10.0],
                   stamp_unix_ms=1e6, now_unix_ms=1e6)
    assert not v.trusted[0] and v.kinds() == ["cov_exceeded"]


def test_hysteresis_delays_recovery():
    cfg = PlausibilityConfig(recover_ticks=5)
    gate = EstimateGate(1, cfg)
    feed(gate, straight_line(10))

    t = 1_000_000.0 + 10 * DT
    gate.check(np.array([[300.0, 300.0]]), [GOOD_TRACE],
               stamp_unix_ms=t, now_unix_ms=t)          # trip

    base = gate._p1[0].copy()
    trusted = []
    for k in range(1, 9):
        pos = np.array([[base[0] + 0.1 * k, 0.0]])
        st = t + k * DT
        trusted.append(bool(gate.check(pos, [GOOD_TRACE], stamp_unix_ms=st,
                                       now_unix_ms=st).trusted[0]))
    # Five clean ticks are served as penalty, then trust returns.
    assert trusted[:5] == [False] * 5
    assert all(trusted[5:])


def test_only_the_spoofed_vehicle_loses_trust():
    gate = EstimateGate(3, PlausibilityConfig())
    for k in range(8):
        st = 1e6 + k * DT
        gate.check(np.array([[k * 0.1, 0.0], [5.0, k * 0.1], [0.0, 5.0]]),
                   [GOOD_TRACE] * 3, stamp_unix_ms=st, now_unix_ms=st)
    st = 1e6 + 8 * DT
    v = gate.check(np.array([[0.8, 0.0], [400.0, 400.0], [0.0, 5.0]]),
                   [GOOD_TRACE] * 3, stamp_unix_ms=st, now_unix_ms=st)
    assert list(v.trusted) == [True, False, True]


def test_rejections_reach_the_audit_log(tmp_path):
    log = AuditLog(tmp_path / "sec.jsonl", b"k" * 32, sync=False)
    gate = EstimateGate(1, PlausibilityConfig(), audit=log)
    feed(gate, straight_line(6))
    st = 1e6 + 6 * DT
    gate.check(np.array([[900.0, 900.0]]), [GOOD_TRACE],
               stamp_unix_ms=st, now_unix_ms=st)

    v = log.verify()
    assert v.ok
    assert v.kinds.get("estimate_rejected") == 1
