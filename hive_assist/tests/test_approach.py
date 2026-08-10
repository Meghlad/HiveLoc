"""D3 acceptance: e -> 0, arrival at s_i, every setpoint supervisor-valid.

The plan's three deliverables, plus the things that turned out to matter more
than the plan anticipated:

  it ARRIVES        a vector field nulls cross-track error and nothing else.
                    Run to gamma = 1 without a terminal phase and the vehicle
                    flies along the tangent line straight past the station with
                    e = 0.000 the whole way. "e -> 0" and "arrives" are two
                    separate claims and both get their own test.
  it is FLYABLE     a Dubins path tighter than v_cruise/omega_max is a reference
                    the aircraft cannot track.
  it never HOMES    the whole point of Domain 3's reframing. Asserted as a
                    property of the trajectory, not just of the endpoint.
"""

import math

import numpy as np
import pytest

from hive.standoff import (
    TASKS,
    GuidanceGains,
    GuidanceLimits,
    TaskGeometry,
    dispatch_on_signal,
    dubins,
    min_turn_radius,
    setpoint_stream,
    standoff_station,
    stations_for,
    validate_approach,
    wrap,
)
from hive.supervisor_gate import StreamLimits, SupervisorConfig

X_TAC = np.array([16.0, 3.0])
START = (-2.0, 7.8, math.radians(155.0))


def fence():
    return SupervisorConfig(
        geofence=[(-40.0, -40.0), (40.0, -40.0), (40.0, 40.0), (-40.0, 40.0)],
        min_spacing_m=1.2)


# --------------------------------------------------------------------------
# Dubins: the curvature-bounded path
# --------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(12))
def test_dubins_lands_on_the_requested_pose(seed):
    rng = np.random.default_rng(seed)
    s = (rng.uniform(-20, 20), rng.uniform(-20, 20), rng.uniform(-math.pi, math.pi))
    e = (rng.uniform(-20, 20), rng.uniform(-20, 20), rng.uniform(-math.pi, math.pi))
    r = float(rng.uniform(1.0, 6.0))

    path = dubins(s, e, r)
    end, heading = path.sample(1.0)
    assert np.linalg.norm(end - np.array(e[:2])) < 1e-9
    assert abs(wrap(heading - e[2])) < 1e-9

    start, h0 = path.sample(0.0)
    assert np.linalg.norm(start - np.array(s[:2])) < 1e-9
    assert abs(wrap(h0 - s[2])) < 1e-9


def test_dubins_respects_the_curvature_bound():
    rng = np.random.default_rng(3)
    for _ in range(40):
        s = (rng.uniform(-15, 15), rng.uniform(-15, 15),
             rng.uniform(-math.pi, math.pi))
        e = (rng.uniform(-15, 15), rng.uniform(-15, 15),
             rng.uniform(-math.pi, math.pi))
        r = float(rng.uniform(2.0, 5.0))
        pl = dubins(s, e, r).polyline(800)

        d = np.diff(pl, axis=0)
        seg = np.linalg.norm(d, axis=1)
        th = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
        k = np.abs(np.diff(th)) / np.maximum(seg[1:], 1e-9)
        assert k.max() * r < 1.02          # 1.0 plus sampling slop


def test_all_six_dubins_words_get_used():
    """LRL was written with a wrong q term and the endpoint verification
    silently discarded it, so across 3000 random pose pairs LRL was chosen zero
    times while its mirror image RLR was chosen 142. A correct implementation
    cannot produce that asymmetry, and this test is what would catch it again.
    """
    rng = np.random.default_rng(0)
    seen = {}
    for _ in range(1500):
        s = (rng.uniform(-20, 20), rng.uniform(-20, 20),
             rng.uniform(-math.pi, math.pi))
        e = (rng.uniform(-20, 20), rng.uniform(-20, 20),
             rng.uniform(-math.pi, math.pi))
        w = dubins(s, e, float(rng.uniform(1.0, 6.0))).word
        seen[w] = seen.get(w, 0) + 1

    assert set(seen) == {"LSL", "RSR", "LSR", "RSL", "RLR", "LRL"}, seen
    ratio = seen["LRL"] / seen["RLR"]
    assert 0.4 < ratio < 2.5, f"LRL/RLR asymmetry {ratio:.2f}: {seen}"


def test_dubins_is_at_least_as_short_as_the_straight_line():
    path = dubins((0, 0, 0), (10, 0, 0), 3.0)
    assert path.length >= 10.0 - 1e-9
    assert path.length == pytest.approx(10.0, abs=1e-9)   # collinear: straight


def test_zero_radius_rejected():
    with pytest.raises(ValueError):
        dubins((0, 0, 0), (5, 5, 1.0), 0.0)


def test_curvature_matches_the_word():
    path = dubins((0, 0, 0), (6, 6, math.pi / 2), 3.0)
    for g in np.linspace(0, 1, 25):
        k = path.curvature(float(g))
        assert abs(k) <= 1.0 / path.radius + 1e-12


# --------------------------------------------------------------------------
# 3.1 Station placement
# --------------------------------------------------------------------------
@pytest.mark.parametrize("task", list(TASKS))
def test_station_sits_at_the_task_standoff_radius(task):
    s, _ = standoff_station(X_TAC, task)
    assert np.linalg.norm(s - X_TAC) == pytest.approx(TASKS[task].standoff_m,
                                                      abs=1e-9)


def test_station_is_not_the_target():
    for task in TASKS:
        s, _ = standoff_station(X_TAC, task)
        assert np.linalg.norm(s - X_TAC) > 1.0


def test_default_approach_faces_the_target():
    s, h = standoff_station(X_TAC, "inspection")
    to_target = math.atan2(*(X_TAC - s)[::-1])
    assert abs(wrap(h - to_target)) < 1e-9


def test_explicit_approach_heading_is_honoured():
    _, h = standoff_station(X_TAC, "delivery")
    assert abs(wrap(h - math.radians(TASKS["delivery"].approach_deg))) < 1e-9


def test_multiple_stations_are_distinct_and_uncoupled():
    """The plan's 'a second signal arrives' case. Separate positions, separate
    approaches, and nothing that ties their timing together."""
    st = stations_for(X_TAC, "inspection", 3, spread_deg=60)
    assert len(st) == 3
    pts = np.array([s for s, _ in st])
    d = np.linalg.norm(pts[:, None] - pts[None, :], axis=2)
    np.fill_diagonal(d, np.inf)
    assert d.min() > 2.0
    for s, _ in st:
        assert np.linalg.norm(s - X_TAC) == pytest.approx(
            TASKS["inspection"].standoff_m, abs=1e-9)


# --------------------------------------------------------------------------
# 3.2 The approach
# --------------------------------------------------------------------------
@pytest.mark.parametrize("task", list(TASKS))
def test_approach_arrives_on_station(task):
    d = dispatch_on_signal(0, START, X_TAC, task)
    assert d.result.arrived, f"{task} never arrived"
    assert d.result.final_error < GuidanceGains().arrive_radius


@pytest.mark.parametrize("task", list(TASKS))
def test_cross_track_converges(task):
    r = dispatch_on_signal(0, START, X_TAC, task).result
    assert abs(r.cross_track[-1]) < 0.25
    assert np.abs(r.cross_track).max() < 1.0


def test_the_vehicle_does_not_fly_past_the_station():
    """The failure the terminal capture phase exists to prevent. Without it the
    vehicle sailed 1955 m beyond the station reporting e = 0.000 m."""
    d = dispatch_on_signal(0, START, X_TAC, "sampling")
    dist = np.linalg.norm(d.result.pos - d.station, axis=1)
    assert dist[-1] < 0.35
    # and it never wandered far past on the way in
    assert dist[np.argmin(dist):].max() < 1.0


@pytest.mark.parametrize("task", list(TASKS))
def test_the_trajectory_never_approaches_the_target(task):
    """Domain 3's central reframing, as a property of the whole trajectory."""
    r = dispatch_on_signal(0, START, X_TAC, task).result
    closest = np.linalg.norm(r.pos - X_TAC, axis=1).min()
    assert closest > 0.7 * TASKS[task].standoff_m, (
        f"{task}: came within {closest:.1f} m of the target itself")


def test_speed_and_turn_rate_stay_inside_the_envelope():
    lim = GuidanceLimits()
    r = dispatch_on_signal(0, START, X_TAC, "inspection", limits=lim).result
    assert r.speed.max() <= lim.v_max + 1e-9
    assert r.speed.min() >= lim.v_min - 1e-9

    dpsi = np.abs(np.diff(np.unwrap(r.heading))) / lim.dt
    assert dpsi.max() <= lim.omega_max + 1e-6


def test_an_unflyable_turn_radius_is_refused():
    """v_cruise/omega_max is a hard floor. A tighter reference is one the
    vehicle rides wide of on every turn, which looks like a tuning problem and
    is not one."""
    lim = GuidanceLimits(v_cruise=5.0, omega_max=1.2)
    assert min_turn_radius(lim) == pytest.approx(5.0 / 1.2)
    with pytest.raises(ValueError):
        dispatch_on_signal(0, START, X_TAC, "inspection", turn_radius=2.0,
                           limits=lim)


def test_default_turn_radius_is_flyable():
    lim = GuidanceLimits()
    d = dispatch_on_signal(0, START, X_TAC, "inspection", limits=lim)
    assert d.path.radius >= min_turn_radius(lim)


def test_dispatch_is_independent_of_other_agents():
    """Two go-signals, two stations, no coupling: running one dispatch must not
    change the other's trajectory in any way."""
    a1 = dispatch_on_signal(0, START, X_TAC, "inspection")
    other = (5.0, -9.0, math.radians(40.0))
    dispatch_on_signal(1, other, X_TAC, "delivery")
    a2 = dispatch_on_signal(0, START, X_TAC, "inspection")
    assert np.array_equal(a1.result.pos, a2.result.pos)


# --------------------------------------------------------------------------
# 3.3 Every setpoint passes the same gate as everything else
# --------------------------------------------------------------------------
@pytest.mark.parametrize("task", list(TASKS))
def test_every_emitted_setpoint_is_supervisor_valid(task):
    lim = GuidanceLimits()
    d = dispatch_on_signal(0, START, X_TAC, task, limits=lim)
    v = validate_approach(
        d.result, fence(),
        StreamLimits(v_max=lim.v_max * lim.dt, d_clear=1.2, dt=1.0))
    assert v.ok, v.kinds()


def test_the_approach_clears_a_loitering_agent():
    """The dispatched agent still shares airspace with the mesh it left."""
    lim = GuidanceLimits()
    d = dispatch_on_signal(0, START, X_TAC, "inspection", limits=lim)
    a = 2 * np.pi * np.arange(12) / 12
    loiter = np.stack([8 * np.cos(a), 8 * np.sin(a)], axis=1)

    v = validate_approach(
        d.result, fence(),
        StreamLimits(v_max=lim.v_max * lim.dt, d_clear=0.8, dt=1.0),
        static=loiter)
    assert v.min_clearance > 0.0


def test_setpoint_stream_shape():
    d = dispatch_on_signal(0, START, X_TAC, "inspection")
    s = setpoint_stream(d.result)
    assert s.ndim == 3 and s.shape[1] == 1 and s.shape[2] == 2
    assert len(s) == len(d.result.pos)


def test_custom_task_geometry():
    g = TaskGeometry("survey", standoff_m=20.0, bearing_deg=45.0)
    s, h = standoff_station(X_TAC, g)
    assert np.linalg.norm(s - X_TAC) == pytest.approx(20.0)
    assert s[0] > X_TAC[0] and s[1] > X_TAC[1]
