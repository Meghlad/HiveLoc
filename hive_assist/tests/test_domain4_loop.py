"""Domain 4 plumbing — the parts that fail SILENTLY in a live run.

`test_stationary_estimate.py` covers the estimate. This file covers everything
around it, and the selection is not arbitrary: each of these is a mistake that
produces a run which looks completely successful. A loop reading the EKF's own
output converges beautifully. A setpoint emitted on a rejected plan flies. A
translation past the anchor footprint holds a confident estimate of a position
nothing is measuring. None of them announce themselves.

No sockets are opened here. The MAVLink-touching classes only connect in
`connect()`, so their decoding and their invariants are testable offline, which
is the reason they were written that way.
"""

from __future__ import annotations

import numpy as np
import pytest

from hive.supervisor_gate import Assignment, Decision, Plan, SupervisorConfig
from sim.ground_truth_bridge import (
    FORBIDDEN_SOURCES,
    GroundTruthBridge,
    default_frame,
    endpoint_url,
)
from sim.orchestrator import (
    GateViolation,
    Orchestrator,
    OrchestratorConfig,
    formation,
)
from sim.range_world import DEFAULT_ANCHORS, RangeConfig, RangeWorld


# --------------------------------------------------------------------------
# D4.1 — ground truth, and the two ways of getting it wrong
# --------------------------------------------------------------------------
class _Msg:
    def __init__(self, kind, **kw):
        self._kind = kind
        self.__dict__.update(kw)

    def get_type(self):
        return self._kind


def test_simstate_is_decoded_as_deg_e7():
    m = _Msg("SIMSTATE", lat=-353632610, lng=1491652300, yaw=0.25)
    lat, lon, yaw = GroundTruthBridge._decode(m)
    assert lat == pytest.approx(-35.363261, abs=1e-9)
    assert lon == pytest.approx(149.16523, abs=1e-9)
    assert yaw == pytest.approx(0.25)


def test_sim_state_uses_the_integer_extensions_only():
    """SIM_STATE.lat is a float32 holding a degE7 VALUE.

    ArduPilot writes `state.latitude * 1.0e7` into a float field
    (libraries/SITL/SITL.cpp). Near -3.5e8 the float32 ulp is 32, so the value
    is quantised to 32 degE7 units — a worst-case 16 units, about 18 cm of
    latitude, against SIMSTATE's exact 1.1 cm. Twelve times the UWB sigma the
    ranges carry, which would make ground truth the dominant error term in the
    loop. The decoder must take lat_int/lon_int and nothing else.
    """
    m = _Msg("SIM_STATE", lat=-353632610.0, lon=1491652300.0, yaw=0.1,
             lat_int=-353632610, lon_int=1491652300)
    lat, lon, _ = GroundTruthBridge._decode(m)
    assert lat == pytest.approx(-35.363261, abs=1e-9)

    # The size of what is being dodged, measured rather than asserted from
    # memory: one ulp of float32 at this magnitude, in metres of latitude.
    ulp_deg_e7 = float(np.spacing(np.float32(353632610.0)))
    assert ulp_deg_e7 == 32.0
    worst_case_m = (ulp_deg_e7 / 2) * 1e-7 * 111_320
    assert worst_case_m > 0.15                      # ~18 cm
    assert worst_case_m > 10 * (1e-7 * 111_320)     # >10x SIMSTATE's 1.1 cm


def test_sim_state_without_extensions_is_refused_not_guessed():
    assert GroundTruthBridge._decode(_Msg("SIM_STATE", yaw=0.0)) is None
    assert GroundTruthBridge._decode(
        _Msg("SIM_STATE", yaw=0.0, lat_int=0, lon_int=0)) is None


def test_the_ekfs_own_output_is_never_a_truth_source():
    """The plan's §4.2 correctness trap, pinned.

    Generating ranges from LOCAL_POSITION_NED closes the loop on itself: the
    estimator corrects the EKF using measurements derived from that same EKF.
    It converges, and it proves nothing.
    """
    for kind in ("LOCAL_POSITION_NED", "GLOBAL_POSITION_INT", "ODOMETRY"):
        assert kind in FORBIDDEN_SOURCES
        assert GroundTruthBridge._decode(_Msg(kind, x=1.0, y=2.0)) is None


def test_unseen_vehicles_are_nan_not_the_origin():
    """Zero is a legal position — it is the anchor. A vehicle never heard from
    must not be indistinguishable from one parked on the origin."""
    bridge = GroundTruthBridge(default_frame(), 3)
    xy = bridge.truths()
    assert xy.shape == (3, 2)
    assert np.all(np.isnan(xy))


def test_endpoint_ports_match_the_supervisors_defaults():
    """14551 + 10*i is ArduPilot's -I convention AND the Rust gate's
    --base-port/--port-stride default. One number, one meaning."""
    assert endpoint_url(0) == "udpin:127.0.0.1:14551"
    assert endpoint_url(3) == "udpin:127.0.0.1:14581"


def test_tacframe_origin_is_the_anchor():
    f = default_frame()
    p = f.from_geodetic_2d(f.anchor_lat_deg, f.anchor_lon_deg)
    assert np.linalg.norm(p) < 1e-6


# --------------------------------------------------------------------------
# D4.2 — the radio
# --------------------------------------------------------------------------
def test_nlos_bias_is_one_sided():
    """NLOS lengthens a path and never shortens it, so the error is not
    zero-mean and cannot be averaged away. If this ever became symmetric the
    Huber kernel downstream would look unnecessary."""
    world = RangeWorld(DEFAULT_ANCHORS, seed=1)
    truth = np.array([[0.0, 0.0], [5.0, 0.0]])
    errs = []
    for _ in range(400):
        for i, j, m in world.measure(truth).inter:
            errs.append(m - np.linalg.norm(truth[i] - truth[j]))
    assert np.mean(errs) > 0.005, "NLOS bias should push the mean range LONG"


def test_a_nan_vehicle_generates_no_links():
    world = RangeWorld(DEFAULT_ANCHORS, seed=1)
    truth = np.array([[0.0, 0.0], [np.nan, np.nan]])
    frame = world.measure(truth)
    assert all(i == 0 and j == 0 for i, j, _ in frame.inter)
    assert all(i == 0 for i, _, _ in frame.anchor)


def test_dropout_rate_matches_the_model():
    world = RangeWorld(DEFAULT_ANCHORS, RangeConfig(), seed=2)
    truth = np.array([[0.0, 0.0], [5.0, 0.0], [0.0, 5.0]])
    yields = [world.measure(truth).link_yield for _ in range(300)]
    assert np.mean(yields) == pytest.approx(1.0 - RangeConfig().uwb.p_dropout,
                                            abs=0.03)


def test_anchor_footprint_is_reported_honestly():
    # The anchors sit at (+/-25, +/-25), so the origin is 35.4 m from the
    # NEAREST of them. r_anchor has to exceed that for the centre of the
    # operating area to be inside the footprint at all — which is the geometry
    # constraint S2 is really about, stated as a number.
    world = RangeWorld(DEFAULT_ANCHORS, RangeConfig(r_anchor=40.0), seed=1)
    assert world.anchor_reach(np.array([[0.0, 0.0]]))[0] == pytest.approx(
        np.hypot(25.0, 25.0))
    assert world.inside_footprint(np.array([[0.0, 0.0]])).all()
    assert not world.inside_footprint(np.array([[500.0, 500.0]])).any()


# --------------------------------------------------------------------------
# D4.5 — the gate invariant
# --------------------------------------------------------------------------
class _FakeCommander:
    def __init__(self, index):
        self.index = index
        self.armed = True
        self.sent = []

    def setpoint_ned(self, north, east, down):
        self.sent.append((north, east, down))


def _orch(n=2):
    return Orchestrator([_FakeCommander(i) for i in range(n)],
                        OrchestratorConfig(alt_m=2.5))


def _plan(plan_id="p1"):
    return Plan(plan_id=plan_id, issued_unix_ms=1_000_000,
                assignments=[Assignment(0, (1.0, 2.0)),
                             Assignment(1, (5.0, 6.0))],
                min_spacing_m=1.2)


def test_a_rejected_plan_emits_exactly_zero_packets():
    orch = _orch()
    plan = _plan()
    reject = Decision(plan_id=plan.plan_id, accepted=False,
                      violations=[{"kind": "WaypointOutsideGeofence"}])
    with pytest.raises(GateViolation):
        orch.emit(plan, reject)
    assert all(not c.sent for c in orch.commanders)


def test_a_decision_for_a_different_plan_is_not_an_accept():
    """A stale ACCEPT is the subtle version of the same bug: the gate really did
    say yes, just to a plan that is no longer the one being flown."""
    orch = _orch()
    stale = Decision(plan_id="p0", accepted=True, violations=[])
    with pytest.raises(GateViolation):
        orch.emit(_plan("p1"), stale)
    assert all(not c.sent for c in orch.commanders)


def test_an_accepted_plan_is_transmitted_north_east_down():
    """Plan waypoints are TacFrame ENU; MAVLink wants NED. The swap happens in
    exactly two places (here and hive/supervisor_io.py) so there are exactly two
    places to look when a vehicle flies at right angles to its plan."""
    orch = _orch()
    plan = _plan()
    ok = Decision(plan_id=plan.plan_id, accepted=True, violations=[])
    assert orch.emit(plan, ok) == 2
    assert orch.commanders[0].sent == [(2.0, 1.0, -2.5)]     # (north, east, down)
    assert orch.commanders[1].sent == [(6.0, 5.0, -2.5)]


def test_unarmed_vehicles_are_never_streamed_at():
    """Streaming setpoints at a vehicle that never armed prints a cheerful log
    and silently invalidates every measurement taken after it."""
    orch = _orch()
    orch.commanders[1].armed = False
    plan = _plan()
    ok = Decision(plan_id=plan.plan_id, accepted=True, violations=[])
    assert orch.emit(plan, ok) == 1
    assert not orch.commanders[1].sent


def test_translation_beyond_the_anchor_footprint_is_refused():
    """S2's real constraint. Outside the footprint the swarm's absolute position
    is pinned by nothing, and the estimate stays confident anyway — which is the
    exact failure mode this project exists to refuse. That is S3's flying
    anchor, and it is scoped separately for a reason."""
    orch = _orch(4)
    world = RangeWorld(DEFAULT_ANCHORS, RangeConfig(r_anchor=40.0), seed=1)
    current = formation(4, "square", 6.0)
    orch.translate(current, (8.0, 0.0), world)          # inside: fine
    with pytest.raises(ValueError, match="anchor footprint"):
        orch.translate(current, (900.0, 900.0), world)


def test_the_python_precheck_reads_the_same_config_as_the_rust_gate():
    """Two configs that drift apart produce an inexplicable REJECT on a plan the
    planner was certain about."""
    import json

    from sim.orchestrator import SUPERVISOR_CFG

    raw = json.loads(SUPERVISOR_CFG.read_text())
    cfg = Orchestrator._load_supervisor_cfg()
    assert isinstance(cfg, SupervisorConfig)
    assert cfg.min_spacing_m == raw["min_spacing_m"]
    assert cfg.max_cov_trace == raw["max_cov_trace"]
    assert [list(p) for p in cfg.geofence] == raw["geofence"]


# --------------------------------------------------------------------------
# Formation geometry
# --------------------------------------------------------------------------
@pytest.mark.parametrize("n", [1, 2, 3, 4, 6])
def test_formations_respect_the_supervisors_spacing_floor(n):
    cfg = Orchestrator._load_supervisor_cfg()
    pts = formation(n, "square", 6.0)
    assert len(pts) == n
    if n > 1:
        d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        assert d.min() > cfg.min_spacing_m


def test_formation_centre_is_where_it_was_asked_to_be():
    pts = formation(4, "square", 6.0, centre=(8.0, -3.0))
    assert np.allclose(pts.mean(axis=0), [8.0, -3.0], atol=1e-9)
