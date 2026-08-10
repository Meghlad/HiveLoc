"""D4.9 — the live coalition standoff: auction -> stations -> flyable rungs.

The offline D3 tests (`test_approach.py`) already prove the geometry. What is
new here is the LIVE hand-over: a Dubins curve is flown by `follow()`'s own
integrator offline, but live it has to survive being chopped into per-tick
setpoints that the supervisor gate will accept one at a time. That conversion is
where a curve can silently become an illegal lunge, so it is what these tests
are about.
"""

from __future__ import annotations

import numpy as np
import pytest

from hive.cbba import comms_adjacency, consensus_elect
from hive.standoff import TASKS, dubins, stations_for, walk_waypoints
from hive.supervisor_gate import check_stream
from sim.orchestrator import Orchestrator, OrchestratorConfig, formation

# Far enough out that a 6 m loiter mesh starts entirely OUTSIDE the 12 m
# inspection perimeter. D3's figure uses (16, 3) with ONE agent that starts
# 20.6 m away; a mesh centred on the anchor does not, and vehicle 0 would begin
# 10.4 m from the target — already inside the circle it is supposed to stop at.
X_TAC = np.array([25.0, 8.0])
X_TAC_TOO_CLOSE = np.array([16.0, 3.0])


class FakeCommander:
    """Enough of `VehicleCommander` for the planner. Never transmits."""

    armed = True

    def setpoint_ned(self, **kw):
        raise AssertionError("no test here may reach the wire")


def make_orch(n: int = 4, spacing: float = 6.0) -> Orchestrator:
    return Orchestrator([FakeCommander() for _ in range(n)],
                        OrchestratorConfig(spacing_m=spacing))


# -- the rung ladder -------------------------------------------------------
def test_rungs_never_exceed_one_tick_of_travel():
    """The property the supervisor's VelocityTooHigh gate is checking."""
    orch = make_orch()
    start = formation(4, "square", 6.0)
    goals, _, _ = orch.dispatch_coalition([0, 1], start, X_TAC)
    step = np.linalg.norm(np.diff(goals, axis=0), axis=2).max()
    assert step <= orch.cfg.limits.v_max + 1e-9


def test_paths_of_different_length_get_the_same_number_of_rungs():
    """Unequal ladders would desynchronise the coalition: `step_toward` only
    advances once EVERY vehicle has arrived, so the shorter path would sit at
    its final rung waiting rather than holding its station."""
    short = dubins((0.0, 0.0, 0.0), (10.0, 0.0, 0.0), 4.0)
    long_ = dubins((0.0, 0.0, 0.0), (-30.0, 12.0, np.pi), 4.0)
    assert long_.length > short.length
    rungs = walk_waypoints([short, long_], 0.6)
    assert rungs.shape[1] == 2                      # (K, A, 2)
    per_path_steps = np.linalg.norm(np.diff(rungs, axis=0), axis=2)
    assert per_path_steps.max() <= 0.6 + 1e-9


def test_rungs_start_at_the_vehicle_and_end_on_the_station():
    orch = make_orch()
    start = formation(4, "square", 6.0)
    goals, stations, _ = orch.dispatch_coalition([0, 1], start, X_TAC)
    for slot, agent in enumerate([0, 1]):
        assert np.linalg.norm(goals[0, agent] - start[agent]) < 1e-6
        assert np.linalg.norm(goals[-1, agent] - stations[slot]) < 1e-6


def test_uninvolved_vehicles_hold_their_slot_for_the_whole_stream():
    orch = make_orch()
    start = formation(4, "square", 6.0)
    goals, _, _ = orch.dispatch_coalition([0, 1], start, X_TAC)
    held = orch.hold()
    for agent in (2, 3):
        assert np.allclose(goals[:, agent, :], held[agent])


def test_walk_waypoints_rejects_a_nonpositive_step():
    p = dubins((0.0, 0.0, 0.0), (10.0, 0.0, 0.0), 4.0)
    with pytest.raises(ValueError):
        walk_waypoints([p], 0.0)


# -- the standoff claim itself ---------------------------------------------
@pytest.mark.parametrize("task", sorted(TASKS))
def test_every_coalition_station_sits_on_the_standoff_perimeter(task):
    for station, _ in stations_for(X_TAC, task, 2, spread_deg=45.0):
        r = np.linalg.norm(np.asarray(station) - X_TAC)
        assert r == pytest.approx(TASKS[task].standoff_m, abs=1e-9)


def test_the_coalition_never_flies_onto_the_target():
    """D3's load-bearing claim, restated for two agents on the live ladder.

    The bar is the perimeter itself, not the 0.7x the offline test uses: these
    rungs ARE the commanded setpoints, so anything inside `standoff_m` is a
    setpoint that closes on the target.
    """
    orch = make_orch()
    start = formation(4, "square", 6.0)
    goals, _, _ = orch.dispatch_coalition([0, 1], start, X_TAC)
    d_s = TASKS["inspection"].standoff_m
    for agent in (0, 1):
        closest = np.linalg.norm(goals[:, agent, :] - X_TAC, axis=1).min()
        assert closest >= d_s - orch.cfg.limits.v_max


def test_a_target_inside_the_loiter_mesh_is_refused_not_flown():
    """The scenario that exposed this: with the target only 16 m out, vehicle 0
    begins 10.4 m from it — already inside the 12 m perimeter. Flying that
    anyway would have silently retired the claim above."""
    orch = make_orch()
    start = formation(4, "square", 6.0)
    assert np.linalg.norm(start[0] - X_TAC_TOO_CLOSE) < 12.0
    with pytest.raises(ValueError, match="standoff perimeter"):
        orch.dispatch_coalition([0, 1], start, X_TAC_TOO_CLOSE)


def test_the_approach_starts_pointing_at_its_station():
    """A hop shorter than 2R with an arbitrary start heading forces Dubins into
    a loop, and that loop can swing through the perimeter. This is the fix."""
    orch = make_orch()
    start = formation(4, "square", 6.0)
    _, stations, paths = orch.dispatch_coalition([0, 1], start, X_TAC)
    for slot, agent in enumerate([0, 1]):
        want = np.arctan2(stations[slot][1] - start[agent][1],
                          stations[slot][0] - start[agent][0])
        assert paths[slot].start[2] == pytest.approx(want)


# -- the gate ---------------------------------------------------------------
def test_every_rung_transition_passes_the_supervisor_guard():
    """The whole stream is pre-validated, holders included. A guard failure here
    is the difference between 'the coalition flew' and 'the coalition refused'."""
    orch = make_orch()
    start = formation(4, "square", 6.0)
    goals, _, _ = orch.dispatch_coalition([0, 1], start, X_TAC)
    vehicles = list(range(4))
    for i in range(len(goals) - 1):
        stream = orch._slew(goals[i], goals[i + 1])
        v = check_stream(stream, orch.supervisor_cfg, orch.cfg.limits, vehicles)
        assert v.ok, f"rung {i}: {v.kinds()} min_clearance {v.min_clearance}"


@pytest.mark.parametrize("n,slots,spread", [
    (4, 2, 45.0), (5, 2, 45.0), (6, 3, 45.0), (6, 3, 90.0),
    (8, 3, 45.0), (8, 4, 60.0), (9, 4, 60.0), (12, 4, 60.0),
])
def test_larger_coalitions_do_not_stall_against_the_guard(n, slots, spread):
    """The regression for the reported failure: at three movers the coalition's
    own paths crossed, the guard refused every rung, and the drones stopped in
    mid-air part way to the perimeter having reported nothing wrong.

    Every rung transition must pass, for movers AND holders, or the mission
    silently becomes a hover.
    """
    orch = make_orch(n)
    start = formation(n, "square", 6.0)
    coalition = sorted(orch.elect(start, X_TAC, n_slots=slots,
                                  cov_trace=np.full(n, 3e-4)).winners)
    goals, _, _ = orch.dispatch_coalition(coalition, start, X_TAC,
                                          spread_deg=spread)
    vehicles = list(range(n))
    for i in range(len(goals) - 1):
        v = check_stream(orch._slew(goals[i], goals[i + 1]),
                         orch.supervisor_cfg, orch.cfg.limits, vehicles)
        assert v.ok, (f"n={n} slots={slots}: rung {i} {v.kinds()} "
                      f"min_clearance {v.min_clearance}")


def test_stations_are_assigned_to_minimise_total_transit():
    """Index-order pairing is what made the paths cross."""
    from sim.orchestrator import assign_stations
    starts = np.array([[0.0, 0.0], [10.0, 0.0]])
    stations = np.array([[10.0, 1.0], [0.0, 1.0]])   # deliberately swapped
    assert assign_stations(starts, stations) == [1, 0]


def test_assignment_is_never_worse_than_index_order():
    rng = np.random.default_rng(7)
    from sim.orchestrator import assign_stations
    for _ in range(50):
        k = int(rng.integers(2, 6))
        starts = rng.uniform(-20, 20, size=(k, 2))
        stations = rng.uniform(-20, 20, size=(k, 2))
        cost = np.linalg.norm(starts[:, None, :] - stations[None, :, :], axis=2)
        order = assign_stations(starts, stations)
        assert sorted(order) == list(range(k))          # a permutation
        assert cost[range(k), order].sum() <= cost[range(k), range(k)].sum() + 1e-9


def test_a_conflicting_coalition_is_refused_rather_than_stalled():
    """Forced head-on geometry: two agents on opposite sides sent to stations
    that require them through the same corridor. The old code flew this and
    froze; it must now name the pair instead."""
    orch = make_orch(2)
    orch.cfg.limits.d_clear = 30.0      # an impossible floor forces the conflict
    start = np.array([[0.0, 3.0], [0.0, -3.0]])
    with pytest.raises(ValueError, match="conflict"):
        orch.dispatch_coalition([0, 1], start, X_TAC)


def test_coalition_stations_clear_each_other_and_the_holders():
    orch = make_orch()
    start = formation(4, "square", 6.0)
    goals, stations, _ = orch.dispatch_coalition([0, 1], start, X_TAC)
    assert (np.linalg.norm(stations[0] - stations[1])
            > orch.cfg.limits.d_clear)
    final = goals[-1]
    d = np.linalg.norm(final[:, None, :] - final[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    assert d.min() > orch.cfg.limits.d_clear


# -- the auction ------------------------------------------------------------
def test_the_auction_elects_exactly_two_and_they_are_the_closest():
    orch = make_orch()
    start = formation(4, "square", 6.0)
    el = orch.elect(start, X_TAC, n_slots=2, cov_trace=np.full(4, 3e-4))
    assert len(el.winners) == 2
    assert el.converged
    # Distance dominates the bid, so the two nearest the target must win.
    order = np.argsort(np.linalg.norm(start - X_TAC, axis=1))
    assert set(el.winners) == set(order[:2].tolist())


# Measured over 60 seeded draws at 0.3 m noise, AFTER the live weights. These
# are the real rates, not targets: n=6 is the weak case and is deliberately
# recorded as such rather than smoothed over. Six vehicles is the size where the
# ring is dense enough to obstruct a crossing mover but not dense enough for the
# auction to have an obviously-nearest pair, so it refuses more often than
# either n=8 or n=12. Closing that gap needs transit deconfliction, not weights.
_NOISE_BUDGET = [
    (6, 2, 14),      # measured 11/60
    (8, 3, 5),       # measured  2/60
    (12, 4, 7),      # measured  4/60
]


@pytest.mark.parametrize("n,slots,max_refused", _NOISE_BUDGET)
def test_the_election_is_stable_under_estimator_noise(n, slots, max_refused):
    """The live failure: D2.1's default weights go flat at this scenario's
    scale, so the coverage term decided the winner and 5 cm of noise elected
    almost every coalition there is — 54 distinct ones out of 56 possible at
    n=8 — most of which must cross the loiter mesh and are refused.
    """
    orch = make_orch(n)
    base = formation(n, "square", 6.0)
    rng = np.random.default_rng(11)
    refused, seen = 0, set()
    for _ in range(60):
        pos = base + rng.normal(0, 0.3, size=base.shape)
        co = sorted(orch.elect(pos, X_TAC, n_slots=slots,
                               cov_trace=np.zeros(n)).winners)
        seen.add(tuple(co))
        try:
            orch.dispatch_coalition(co, pos, X_TAC)
        except ValueError:
            refused += 1
    assert len(seen) <= max(8, slots * 3), \
        f"election unstable: {len(seen)} distinct coalitions over 60 draws"
    assert refused <= max_refused, \
        f"{refused}/60 elected coalitions were unflyable (budget {max_refused})"


def test_live_weights_make_distance_dominate():
    """The specific defect: at 20-32 m the default bid spread was 2% against a
    55% spread in distance, so ranking was decided by noise."""
    from hive.cbba import BidWeights, SwarmState, bid_scores
    from sim.orchestrator import LIVE_BID_WEIGHTS

    orch = make_orch(8)
    pos = formation(8, "square", 6.0)
    state = SwarmState(pos=pos, battery=np.ones(8), sensor=np.ones(8),
                       region=orch.hold(), cov_trace=np.zeros(8),
                       max_cov_trace=orch.supervisor_cfg.max_cov_trace)
    dist = np.linalg.norm(pos - X_TAC, axis=1)
    default = bid_scores(state, X_TAC, BidWeights())
    live = bid_scores(state, X_TAC, LIVE_BID_WEIGHTS)

    def spread(b):
        return (b.max() - b.min()) / abs(b.max())

    # Measured: 0.0229 -> 0.2211, a 10x widening of the signal the ranking
    # rests on. Not "distance is everything" — battery and sensor still carry
    # their weight — just enough that noise no longer casts the deciding vote.
    assert spread(default) < 0.05          # the defect, pinned
    assert spread(live) > 0.15             # distance now actually discriminates
    # and the ranking must still be exactly nearest-first
    assert np.argsort(-live).tolist() == np.argsort(dist).tolist()


def test_an_untrusted_agent_cannot_join_the_coalition():
    """Domain 1 reaching into Domain 2: an agent whose covariance would trip the
    supervisor's trust gate must never be elected, because the plan naming it
    would be REJECTED on arrival."""
    orch = make_orch()
    start = formation(4, "square", 6.0)
    order = np.argsort(np.linalg.norm(start - X_TAC, axis=1))
    nearest = int(order[0])
    cov = np.full(4, 3e-4)
    cov[nearest] = orch.supervisor_cfg.max_cov_trace * 10.0
    el = orch.elect(start, X_TAC, n_slots=2, cov_trace=cov)
    assert nearest not in el.winners


def test_election_agrees_with_a_direct_consensus_call():
    """The live wrapper must not quietly diverge from D2.1's own entry point."""
    orch = make_orch()
    start = formation(4, "square", 6.0)
    el = orch.elect(start, X_TAC, n_slots=2, cov_trace=np.zeros(4))
    from hive.cbba import SwarmState, bid_scores
    scores = bid_scores(SwarmState(pos=start, battery=np.ones(4),
                                   sensor=np.ones(4), region=orch.hold(),
                                   cov_trace=np.zeros(4),
                                   max_cov_trace=orch.supervisor_cfg.max_cov_trace),
                        X_TAC)
    direct = consensus_elect(scores, comms_adjacency(start, 30.0), 2)
    assert sorted(el.winners) == sorted(direct.winners)


# -- the QGC markers --------------------------------------------------------
def test_markers_are_the_target_plus_one_station_per_agent():
    from sim import qgc_markers
    stations = [np.array([4.0, 3.5]), np.array([7.1, -5.1])]
    marks = qgc_markers.markers_for_dispatch(X_TAC, stations, [0, 1])
    assert len(marks) == 3
    assert (marks[0].x, marks[0].y) == tuple(X_TAC)
    assert "v0" in marks[1].name and "v1" in marks[2].name


def test_marker_geodetic_conversion_round_trips():
    """A marker that lands in the wrong place is worse than no marker: it would
    be read as the estimator being wrong."""
    from sim.ground_truth_bridge import default_frame
    from sim.qgc_markers import Marker, _to_degE7

    frame = default_frame()
    m = Marker("t", 16.0, 3.0)
    lat_e7, lon_e7 = _to_degE7(frame, m)
    back = frame.from_geodetic_2d(lat_e7 / 1e7, lon_e7 / 1e7)
    # degE7 quantisation is ~1.1 cm; anything under a centimetre is the encoding
    # floor, not a frame error.
    assert np.linalg.norm(back - np.array([16.0, 3.0])) < 0.02
