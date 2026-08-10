"""D4 — the file hand-off to the native supervisor.

Two properties, both of which fail SILENTLY in the sim if broken, which is why
they are pinned here rather than left to the SITL run to notice.

1. THE FRAME SWAP. Everything upstream is ENU (TacFrame is ENU about the
   surveyed anchor). The Rust struct field is `waypoint_ne` and main.rs maps
   [0] -> x (North), [1] -> y (East). Writing ENU straight into it mirrors every
   commanded position about x=y: an easting becomes a northing, the vehicle
   flies at right angles to the plan, and it does so INSIDE the geofence and
   past every gate, because a square fence is symmetric under that swap.

2. NO NON-FINITE JSON. An indeterminate marginal covariance is real (iSAM2
   raises on it), and the natural encoding is float('inf'). json.dumps writes
   that as a bare `Infinity` token, which serde_json refuses — so one vehicle's
   bad marginal would stop the supervisor parsing the estimate at all and take
   the WHOLE fleet down with a parse error, instead of excluding that one
   vehicle via CovarianceTooHigh.
"""

from __future__ import annotations

import json

import pytest

from hive import supervisor_io
from hive.supervisor_gate import Assignment, EstimateSnapshot, Plan


@pytest.fixture
def shared(tmp_path, monkeypatch):
    monkeypatch.setattr(supervisor_io, "HIVE_DIR", tmp_path)
    monkeypatch.setattr(supervisor_io, "ESTIMATE_PATH", tmp_path / "estimate.json")
    monkeypatch.setattr(supervisor_io, "PLAN_PATH", tmp_path / "plan.json")
    monkeypatch.setattr(supervisor_io, "DECISION_PATH", tmp_path / "decision.json")
    return tmp_path


def test_enu_to_ne_swaps_axes():
    assert supervisor_io.enu_to_ne((3.0, 7.0)) == [7.0, 3.0]


def test_plan_waypoints_are_written_north_east(shared):
    """A vehicle 8 m due EAST must be written as north=0, east=8."""
    plan = Plan(plan_id="p", issued_unix_ms=1_000,
                assignments=[Assignment(0, (8.0, 0.0))])
    supervisor_io.publish_plan(plan)

    wp = json.loads((shared / "plan.json").read_text())["assignments"][0]["waypoint_ne"]
    assert wp == [0.0, 8.0], "waypoint_ne must be [North, East], not ENU"


def test_estimate_positions_are_written_north_east(shared):
    est = EstimateSnapshot(frame_index=0, stamp_unix_ms=1_000,
                           pos=[(8.0, 0.0), (0.0, 5.0)], cov_trace=[0.001, 0.001])
    supervisor_io.publish_estimate(est)

    pos = json.loads((shared / "estimate.json").read_text())["pos"]
    assert pos == [[0.0, 8.0], [5.0, 0.0]]


def test_swap_is_an_involution_not_a_rotation():
    """Applying it twice returns the original — a mirror, not a 90 deg turn.

    Worth pinning because a rotation would also 'look right' on the axis-aligned
    cases above while being wrong everywhere off-axis.
    """
    p = (2.0, -5.0)
    assert tuple(supervisor_io.enu_to_ne(supervisor_io.enu_to_ne(p))) == p


def test_non_finite_cov_trace_is_refused_loudly(shared):
    """inf must raise here rather than produce JSON the Rust side cannot read."""
    est = EstimateSnapshot(frame_index=0, stamp_unix_ms=1_000,
                           pos=[(0.0, 0.0)], cov_trace=[float("inf")])
    with pytest.raises(ValueError):
        supervisor_io.publish_estimate(est)


def test_write_is_atomic_leaving_no_partial_file(shared):
    """The supervisor polls on its own clock; a torn read must be impossible."""
    est = EstimateSnapshot(frame_index=0, stamp_unix_ms=1_000,
                           pos=[(1.0, 2.0)], cov_trace=[0.001])
    supervisor_io.publish_estimate(est)
    assert not list(shared.glob("*.tmp")), "temp file left behind"
    json.loads((shared / "estimate.json").read_text())      # parses cleanly
