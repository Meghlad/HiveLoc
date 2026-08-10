"""D1.0 — the geodetic → TacFrame transform.

The handoff boundary of this whole project is one message: an external detector
says "the thing is at (lat, lon)". Everything downstream — the estimator, the
supervisor's geofence, MAVLink's LOCAL_NED setpoints — speaks a local metric
frame. This module is the single place those two worlds meet.

WHY THE TRANSFORM IS *KNOWN*, NOT ESTIMATED. The anchor A is GPS-surveyed. Its
geodetic position is external truth, fixed before the mission starts. So
T_geo->tac is a rigid, constant element of SE(3): no drift, no state, nothing to
estimate, no covariance. Target ingest is a one-shot projection and then we
never touch geodesy again for the rest of the flight. That property is what
makes Domain 1's null-space argument work at all — an anchor that had to be
estimated would carry the gauge freedom right back in with it.

    WGS84 geodetic (lat, lon, alt)
        |  geodetic_to_ecef        [ellipsoid model]
        v
    ECEF (x, y, z)                 [earth-centred, earth-fixed]
        |  R_ecef_to_enu(A) . (X - A_ecef)
        v
    ENU about the anchor           [East, North, Up, metres]
        |  Rz(-yaw_offset)
        v
    TacFrame                       [what everything else speaks]

TacFrame is ENU rotated about Up by the survey's yaw offset, so a site can align
its x-axis with a runway, a pier, a building face — whatever the operators
actually think in — rather than with magnetic east.

Units: radians in, metres out. The public API takes degrees because that is what
detectors emit; conversion happens at the boundary, once.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# --------------------------------------------------------------------------
# WGS84
# --------------------------------------------------------------------------
WGS84_A = 6_378_137.0                      # semi-major axis (m)
WGS84_F = 1.0 / 298.257_223_563            # flattening
WGS84_B = WGS84_A * (1.0 - WGS84_F)        # semi-minor axis (m)
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)       # first eccentricity squared
WGS84_EP2 = (WGS84_A**2 - WGS84_B**2) / WGS84_B**2   # second eccentricity sq.


def geodetic_to_ecef(lat_deg: float, lon_deg: float, alt_m: float = 0.0) -> np.ndarray:
    """WGS84 geodetic -> ECEF metres."""
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    sin_lat, cos_lat = math.sin(lat), math.cos(lat)
    # radius of curvature in the prime vertical
    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    return np.array([
        (n + alt_m) * cos_lat * math.cos(lon),
        (n + alt_m) * cos_lat * math.sin(lon),
        (n * (1.0 - WGS84_E2) + alt_m) * sin_lat,
    ])


def ecef_to_geodetic(x: float, y: float, z: float) -> tuple[float, float, float]:
    """ECEF metres -> WGS84 (lat_deg, lon_deg, alt_m).

    Bowring's closed-form solution. Accurate to well under a micrometre for any
    altitude we will ever fly, and — unlike the naive fixed-point iteration —
    it has no convergence branch to get wrong at high latitude.
    """
    lon = math.atan2(y, x)
    p = math.hypot(x, y)
    if p < 1e-9:                       # on the spin axis: lat is +/-90, lon undefined
        lat = math.copysign(math.pi / 2.0, z)
        alt = abs(z) - WGS84_B
        return math.degrees(lat), math.degrees(lon), alt

    theta = math.atan2(z * WGS84_A, p * WGS84_B)
    sin_t, cos_t = math.sin(theta), math.cos(theta)
    lat = math.atan2(
        z + WGS84_EP2 * WGS84_B * sin_t**3,
        p - WGS84_E2 * WGS84_A * cos_t**3,
    )
    sin_lat = math.sin(lat)
    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    alt = p / math.cos(lat) - n
    return math.degrees(lat), math.degrees(lon), alt


def r_ecef_to_enu(lat_deg: float, lon_deg: float) -> np.ndarray:
    """Rotation from ECEF into the local ENU triad at a reference point."""
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    s_lat, c_lat = math.sin(lat), math.cos(lat)
    s_lon, c_lon = math.sin(lon), math.cos(lon)
    return np.array([
        [-s_lon,          c_lon,         0.0 ],   # East
        [-s_lat * c_lon, -s_lat * s_lon, c_lat],  # North
        [ c_lat * c_lon,  c_lat * s_lon, s_lat],  # Up
    ])


def rz(angle_rad: float) -> np.ndarray:
    """Right-handed rotation about the Up/Z axis."""
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def enu_to_ned(v: np.ndarray) -> np.ndarray:
    """[E, N, U] -> [N, E, D]. MAVLink's LOCAL_NED wants the latter."""
    v = np.asarray(v, dtype=float)
    return np.array([v[..., 1], v[..., 0], -v[..., 2]]).T if v.ndim > 1 else \
        np.array([v[1], v[0], -v[2]])


def ned_to_enu(v: np.ndarray) -> np.ndarray:
    """[N, E, D] -> [E, N, U]. Its own inverse, but named for the direction."""
    return enu_to_ned(v)


# --------------------------------------------------------------------------
# TacFrame
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class TacFrame:
    """A surveyed anchor plus a yaw offset: the mission's metric frame.

    Frozen on purpose. A TacFrame that could be mutated mid-mission is a
    silently-moving origin, which is exactly the failure mode the surveyed
    anchor exists to eliminate.

    Attributes
    ----------
    anchor_lat_deg, anchor_lon_deg, anchor_alt_m
        The surveyed anchor A. Origin of TacFrame by construction.
    yaw_offset_deg
        Counter-clockwise angle from ENU East to the TacFrame +x axis. 0 means
        TacFrame *is* ENU.
    survey_sigma_m
        1-sigma uncertainty of the survey itself (m). Not used by the transform
        — it is carried here so Domain 1 can set the anchor factor's Sigma_A
        honestly instead of inventing a tight number. See anchor_factor.py.
    """

    anchor_lat_deg: float
    anchor_lon_deg: float
    anchor_alt_m: float = 0.0
    yaw_offset_deg: float = 0.0
    survey_sigma_m: float = 0.02

    # -- cached pieces of the (constant) transform ------------------------
    @property
    def anchor_ecef(self) -> np.ndarray:
        return geodetic_to_ecef(
            self.anchor_lat_deg, self.anchor_lon_deg, self.anchor_alt_m
        )

    @property
    def r_ecef_to_tac(self) -> np.ndarray:
        """The full ECEF -> TacFrame rotation, ENU triad then survey yaw."""
        return rz(-math.radians(self.yaw_offset_deg)) @ r_ecef_to_enu(
            self.anchor_lat_deg, self.anchor_lon_deg
        )

    # -- the one-shot ingest ---------------------------------------------
    def from_geodetic(self, lat_deg: float, lon_deg: float,
                      alt_m: float = 0.0) -> np.ndarray:
        """Project a geodetic coordinate into TacFrame metres.

        This is *the* function the plan's input contract runs through: an
        external detector's {lat, lon, alt} becomes X_tac, and nothing
        downstream ever sees a degree again.
        """
        d_ecef = geodetic_to_ecef(lat_deg, lon_deg, alt_m) - self.anchor_ecef
        return self.r_ecef_to_tac @ d_ecef

    def to_geodetic(self, p_tac: np.ndarray) -> tuple[float, float, float]:
        """TacFrame metres -> geodetic. The inverse; used for logging and for
        the round-trip test that proves the transform is actually rigid."""
        p_tac = np.asarray(p_tac, dtype=float)
        ecef = self.anchor_ecef + self.r_ecef_to_tac.T @ p_tac
        return ecef_to_geodetic(*ecef)

    def from_geodetic_2d(self, lat_deg: float, lon_deg: float) -> np.ndarray:
        """Planar convenience: drop Up. Domains 2 and 3 are planar studies."""
        return self.from_geodetic(lat_deg, lon_deg, self.anchor_alt_m)[:2]


# --------------------------------------------------------------------------
# Self-check
# --------------------------------------------------------------------------
if __name__ == "__main__":
    # A surveyed anchor with a non-trivial yaw offset, so the check exercises
    # every stage rather than accidentally passing on an identity rotation.
    frame = TacFrame(
        anchor_lat_deg=47.397_742,     # the PX4 SITL default origin, so the
        anchor_lon_deg=8.545_594,      # numbers here match what Domain 4 flies
        anchor_alt_m=488.0,
        yaw_offset_deg=31.5,
        survey_sigma_m=0.02,
    )

    print(f"anchor ECEF        : {frame.anchor_ecef}")
    print(f"anchor -> TacFrame : {frame.from_geodetic(frame.anchor_lat_deg, frame.anchor_lon_deg, frame.anchor_alt_m)}")

    # a target ~120 m north-east of the anchor
    target = frame.from_geodetic(47.398_8, 8.546_9, 488.0)
    print(f"target X_tac       : {target}  (|.| = {np.linalg.norm(target):.2f} m)")

    back = frame.to_geodetic(target)
    print(f"round trip         : {back}")
    err = np.linalg.norm(np.array(frame.from_geodetic(*back)) - target)
    print(f"round-trip error   : {err:.3e} m   (survey sigma {frame.survey_sigma_m} m)")
