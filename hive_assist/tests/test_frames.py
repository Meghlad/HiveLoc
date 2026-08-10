"""D1.0 acceptance: the geodetic <-> TacFrame transform is rigid and exact.

The plan's bar is "round-trip error < survey tolerance". That is a weak bar and
we clear it by seven orders of magnitude, so these tests check the properties
that actually matter downstream instead: that the transform preserves distances
and angles (it must, or the supervisor's geofence and spacing gates are
measuring something other than metres), and that it is genuinely constant.
"""

import math

import numpy as np
import pytest

from hive.frames import (
    TacFrame,
    ecef_to_geodetic,
    enu_to_ned,
    geodetic_to_ecef,
    ned_to_enu,
    r_ecef_to_enu,
    rz,
)

SURVEY_TOL_M = 0.02          # the plan's acceptance bar
RIGID_TOL_M = 1e-6           # what a rigid transform should actually deliver


def frame() -> TacFrame:
    return TacFrame(
        anchor_lat_deg=47.397_742,
        anchor_lon_deg=8.545_594,
        anchor_alt_m=488.0,
        yaw_offset_deg=31.5,
    )


# --------------------------------------------------------------------------
# The plan's stated deliverable
# --------------------------------------------------------------------------
def test_anchor_maps_to_origin():
    f = frame()
    p = f.from_geodetic(f.anchor_lat_deg, f.anchor_lon_deg, f.anchor_alt_m)
    assert np.linalg.norm(p) < RIGID_TOL_M


@pytest.mark.parametrize("dlat,dlon,dalt", [
    (0.0010, 0.0013, 0.0), (-0.0021, 0.0004, 25.0), (0.0, -0.0030, -10.0),
    (0.0005, 0.0005, 300.0), (-0.0009, -0.0011, -75.0),
])
def test_round_trip_beats_survey_tolerance(dlat, dlon, dalt):
    f = frame()
    lat = f.anchor_lat_deg + dlat
    lon = f.anchor_lon_deg + dlon
    alt = f.anchor_alt_m + dalt

    p = f.from_geodetic(lat, lon, alt)
    lat2, lon2, alt2 = f.to_geodetic(p)
    p2 = f.from_geodetic(lat2, lon2, alt2)

    err = np.linalg.norm(p2 - p)
    assert err < SURVEY_TOL_M, "fails the plan's acceptance bar"
    assert err < RIGID_TOL_M, "transform is not behaving as rigid"


def test_ecef_round_trip():
    for lat, lon, alt in [(47.4, 8.5, 488.0), (-33.9, 151.2, 58.0),
                          (0.0, 0.0, 0.0), (78.2, 15.6, 12.0)]:
        x, y, z = geodetic_to_ecef(lat, lon, alt)
        lat2, lon2, alt2 = ecef_to_geodetic(x, y, z)
        assert abs(lat2 - lat) < 1e-9
        assert abs(lon2 - lon) < 1e-9
        assert abs(alt2 - alt) < 1e-6


# --------------------------------------------------------------------------
# Rigidity: what the supervisor's gates depend on
# --------------------------------------------------------------------------
def test_transform_preserves_distance():
    """Two targets 'n' metres apart in the world must be n metres apart in
    TacFrame, or min_spacing_m in the supervisor means nothing."""
    f = frame()
    a = f.from_geodetic(47.398_0, 8.546_0, 488.0)
    b = f.from_geodetic(47.399_0, 8.547_5, 500.0)

    # ground truth separation, straight-line in ECEF
    ea = geodetic_to_ecef(47.398_0, 8.546_0, 488.0)
    eb = geodetic_to_ecef(47.399_0, 8.547_5, 500.0)

    assert np.linalg.norm(b - a) == pytest.approx(np.linalg.norm(eb - ea),
                                                  abs=RIGID_TOL_M)


def test_rotation_is_orthonormal():
    f = frame()
    r = f.r_ecef_to_tac
    assert np.allclose(r @ r.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(r) == pytest.approx(1.0, abs=1e-12)   # proper, no flip


def test_enu_triad_points_the_right_way():
    """North of the anchor must be +y in ENU; east must be +x. A sign error
    here would fly the whole swarm mirrored and every test above would still
    pass, so it gets its own check."""
    f = TacFrame(anchor_lat_deg=47.4, anchor_lon_deg=8.5, anchor_alt_m=0.0)
    north = f.from_geodetic(47.401, 8.5, 0.0)
    east = f.from_geodetic(47.4, 8.501, 0.0)

    assert north[1] > 100.0 and abs(north[0]) < 1.0    # +North, no East
    assert east[0] > 50.0 and abs(east[1]) < 1.0       # +East, no North


def test_yaw_offset_rotates_as_declared():
    """yaw_offset_deg is CCW from ENU East to TacFrame +x, so TacFrame is ENU
    post-multiplied by Rz(-yaw). Checked on the rotation itself: a geodetic
    probe point cannot resolve this exactly, because a constant-latitude step
    is not due east in ENU (meridian convergence leaks ~0.5 mm of north into a
    75 m step), and that real geometry would mask a small algebra error."""
    common = dict(anchor_lat_deg=47.4, anchor_lon_deg=8.5, anchor_alt_m=0.0)
    plain = TacFrame(**common, yaw_offset_deg=0.0)
    turned = TacFrame(**common, yaw_offset_deg=90.0)

    assert np.allclose(turned.r_ecef_to_tac, rz(-math.pi / 2) @ plain.r_ecef_to_tac,
                       atol=1e-12)

    # and the rotation is rigid: same point, same distance, bearing turned by 90
    p0 = plain.from_geodetic(47.4, 8.501, 0.0)
    p1 = turned.from_geodetic(47.4, 8.501, 0.0)
    assert np.linalg.norm(p0) == pytest.approx(np.linalg.norm(p1), abs=1e-9)
    turn = math.degrees(math.atan2(p1[1], p1[0]) - math.atan2(p0[1], p0[0]))
    assert turn == pytest.approx(-90.0, abs=1e-9)


def test_r_ecef_to_enu_rows_are_the_triad():
    r = r_ecef_to_enu(0.0, 0.0)     # on the equator at the prime meridian
    assert np.allclose(r[0], [0, 1, 0])      # East  -> +Y_ecef
    assert np.allclose(r[1], [0, 0, 1])      # North -> +Z_ecef
    assert np.allclose(r[2], [1, 0, 0])      # Up    -> +X_ecef


# --------------------------------------------------------------------------
# Constancy: the property Domain 1's null-space argument leans on
# --------------------------------------------------------------------------
def test_transform_is_constant_and_frame_is_frozen():
    f = frame()
    first = f.r_ecef_to_tac.copy()
    for _ in range(50):
        f.from_geodetic(47.399, 8.547, 490.0)
    assert np.array_equal(f.r_ecef_to_tac, first)

    with pytest.raises(Exception):        # dataclass(frozen=True)
        f.anchor_lat_deg = 0.0            # type: ignore[misc]


# --------------------------------------------------------------------------
# NED bridge to MAVLink
# --------------------------------------------------------------------------
def test_enu_ned_bridge():
    v = np.array([1.0, 2.0, 3.0])                 # E=1, N=2, U=3
    assert np.allclose(enu_to_ned(v), [2.0, 1.0, -3.0])
    assert np.allclose(ned_to_enu(enu_to_ned(v)), v)


def test_planar_helper_drops_up():
    f = frame()
    p3 = f.from_geodetic(47.398_8, 8.546_9, f.anchor_alt_m)
    p2 = f.from_geodetic_2d(47.398_8, 8.546_9)
    assert np.allclose(p2, p3[:2])
