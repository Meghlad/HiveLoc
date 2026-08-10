"""D1.2 — does a single surveyed anchor really collapse the gauge to zero?

The plan asserts it. This module *measures* it: build one swarm scenario, swap
only the anchor factors, and read off dim ker(H) for each configuration. The
answer turns out to be more interesting than the plan assumed, and the point of
doing it numerically is that the surprise shows up as a number instead of as a
bug three domains later.

    configuration                          expected  dim ker(H)
    -------------------------------------------------------------
    no anchor                                        3   tx, ty, yaw
    1 anchor, range only, single keyframe            3   under-determined
    1 anchor, range only, all keyframes              1   yaw about the anchor
    1 anchor, range + BODY-frame bearing             1   yaw survives
    1 anchor, range + ANCHOR-frame bearing           0   full rank
    2 separated anchors, range only                  0   full rank

READ THE TWO SURPRISES.

1. A motion baseline does NOT rescue yaw from a single range-only anchor. Rotate
   the entire solution — every agent, every keyframe, every heading — about the
   anchor and every range to the anchor is preserved, every inter-agent range is
   preserved, every body-frame odometry residual is preserved. The symmetry is
   exact, so the null vector is exact, and no amount of flying around changes
   that. What motion *does* buy is translation: it takes the single-keyframe
   case from 3 free directions down to 1.

2. A bearing only helps if it is expressed in an externally-known orientation.
   The anchor is a surveyed ground station bolted to the earth, so a bearing
   measured *at the anchor* is real yaw information. An AoA antenna on the
   vehicle measures the anchor in the vehicle's own drifting body frame, and
   that is invariant to rotation about the anchor too — it buys nothing.

So the plan's open question ("bearing channel, or yaw from ranges over a motion
baseline?") has a definite answer: **the motion-baseline route does not exist
for a single anchor.** Either survey the anchor's heading and get a real bearing
channel out of it, or survey a second anchor. Both are cheap; neither is
optional.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from hive.anchor_factor import (
    AnchorBearingFactor,
    AnchorRangeFactor,
    InterAgentRangeFactor,
    OdometryFactor,
    PriorFactor,
    gauge_generators,
    information_matrix,
    linearize_all,
    pose,
    rot,
    state_dim,
    wrap,
)

# Rank tolerance. Applied to the singular values of J, not the eigenvalues of
# H: forming H squares the condition number, and we would rather not throw away
# eight digits before asking a rank question. The gauge directions sit at ~1e-16
# relative while the smallest real direction sits above 1e-3 relative, so the
# gap is ~13 orders of magnitude and the exact threshold is not load-bearing.
# `rank_report` returns the gap so that claim is checkable, not assumed.
RANK_TOL = 1e-8


# --------------------------------------------------------------------------
# Scenario
# --------------------------------------------------------------------------
@dataclass
class Scenario:
    """A loiter mesh flying an arc, with a surveyed anchor off to one side.

    Ground truth only — every factor is built at its noise-free measurement, so
    the residual is zero and H is evaluated exactly at the solution. That is
    what an observability question wants: no linearisation error, no optimiser
    convergence to argue about, just the Jacobian's rank at the true state.
    """

    n_agents: int = 6
    n_keyframes: int = 5
    ring_radius: float = 8.0
    arc_per_frame: float = 0.18          # rad of loiter travel per keyframe
    anchor: np.ndarray = field(default_factory=lambda: np.array([12.0, -5.0]))
    anchor_2: np.ndarray = field(default_factory=lambda: np.array([-9.0, 11.0]))

    sigma_odom: float = 0.05
    sigma_odom_theta: float = 0.01
    sigma_range: float = 0.10
    sigma_anchor_range: float = 0.08     # survey + UWB, honestly sized
    sigma_anchor_bearing: float = 0.02   # ~1.1 deg

    def truth(self) -> np.ndarray:
        """Ground-truth state. Agents ride a ring, nose along the tangent."""
        x = np.zeros(state_dim(self.n_agents, self.n_keyframes))
        for i in range(self.n_agents):
            phase = 2.0 * math.pi * i / self.n_agents
            for t in range(self.n_keyframes):
                a = phase + self.arc_per_frame * t
                k = 3 * (i * self.n_keyframes + t)
                x[k] = self.ring_radius * math.cos(a)
                x[k + 1] = self.ring_radius * math.sin(a)
                x[k + 2] = wrap(a + math.pi / 2.0)      # tangent heading
        return x

    # -- factor construction, all at ground truth -------------------------
    def _odometry(self, x):
        out = []
        for i in range(self.n_agents):
            for t in range(self.n_keyframes - 1):
                p0, th0 = pose(x, i, t, self.n_keyframes)
                p1, th1 = pose(x, i, t + 1, self.n_keyframes)
                out.append(OdometryFactor(
                    sigma=self.sigma_odom, agent=i, t0=t,
                    delta_body=rot(th0).T @ (p1 - p0),
                    delta_theta=wrap(th1 - th0),
                    sigma_theta=self.sigma_odom_theta,
                ))
        return out

    def _inter_agent_ranges(self, x):
        out = []
        for t in range(self.n_keyframes):
            for i in range(self.n_agents):
                for j in range(i + 1, self.n_agents):
                    pi, _ = pose(x, i, t, self.n_keyframes)
                    pj, _ = pose(x, j, t, self.n_keyframes)
                    out.append(InterAgentRangeFactor(
                        sigma=self.sigma_range, a=i, b=j, t=t,
                        meas=float(np.linalg.norm(pi - pj)),
                    ))
        return out

    def _anchor_ranges(self, x, anchor, keyframes, agents=None):
        agents = range(self.n_agents) if agents is None else agents
        out = []
        for t in keyframes:
            for i in agents:
                p, _ = pose(x, i, t, self.n_keyframes)
                out.append(AnchorRangeFactor(
                    sigma=self.sigma_anchor_range, anchor=anchor, agent=i, t=t,
                    meas=float(np.linalg.norm(p - anchor)),
                ))
        return out

    def _anchor_bearings(self, x, anchor, keyframes, frame, agents=None):
        agents = range(self.n_agents) if agents is None else agents
        out = []
        for t in keyframes:
            for i in agents:
                p, th = pose(x, i, t, self.n_keyframes)
                d = (p - anchor) if frame == "anchor" else rot(th).T @ (anchor - p)
                out.append(AnchorBearingFactor(
                    sigma=self.sigma_anchor_bearing, anchor=anchor, agent=i, t=t,
                    meas=math.atan2(d[1], d[0]), frame=frame,
                ))
        return out

    # -- the configurations under test ------------------------------------
    def factors(self, config: str):
        """Base graph (odometry + inter-agent ranges) plus the anchor set named
        by `config`. Only the anchor part ever changes."""
        x = self.truth()
        a, a2 = self.anchor, self.anchor_2
        all_kf = list(range(self.n_keyframes))
        one = [0]
        base = self._odometry(x) + self._inter_agent_ranges(x)

        if config == "no_anchor":
            extra = []
        elif config == "one_agent_static":
            extra = self._anchor_ranges(x, a, one, agents=one)
        elif config == "one_agent_motion":
            extra = self._anchor_ranges(x, a, all_kf, agents=one)
        elif config == "mesh_static":
            extra = self._anchor_ranges(x, a, one)
        elif config == "mesh_motion":
            extra = self._anchor_ranges(x, a, all_kf)
        elif config == "mesh_body_bearing":
            extra = (self._anchor_ranges(x, a, all_kf)
                     + self._anchor_bearings(x, a, all_kf, "body"))
        elif config == "mesh_anchor_bearing":
            extra = (self._anchor_ranges(x, a, all_kf)
                     + self._anchor_bearings(x, a, all_kf, "anchor"))
        elif config == "two_anchors_range":
            extra = (self._anchor_ranges(x, a, all_kf)
                     + self._anchor_ranges(x, a2, all_kf))
        else:
            raise ValueError(f"unknown config {config!r}")
        return base + extra


# label, expected dim ker(H), one-line reason
CONFIGS = {
    "no_anchor":           ("no anchor",                        3,
                            "full SE(2) gauge: tx, ty, yaw"),
    "one_agent_static":    ("1 anchor, 1 agent, 1 keyframe",     2,
                            "one scalar range on a 3-DoF body"),
    "one_agent_motion":    ("1 anchor, 1 agent, all keyframes",  1,
                            "TEMPORAL baseline recovers translation"),
    "mesh_static":         ("1 anchor, all agents, 1 keyframe",  1,
                            "SPATIAL baseline does the same, instantly"),
    "mesh_motion":         ("1 anchor, all agents, all keyframes", 1,
                            "more range data, same kernel: yaw never dies"),
    "mesh_body_bearing":   ("  + BODY-frame bearing (vehicle AoA)", 1,
                            "invariant to rotation about the anchor"),
    "mesh_anchor_bearing": ("  + ANCHOR-frame bearing (surveyed)", 0,
                            "external heading kills yaw -> full rank"),
    "two_anchors_range":   ("2 surveyed anchors, range only",    0,
                            "second known point kills yaw -> full rank"),
}


# --------------------------------------------------------------------------
# Rank
# --------------------------------------------------------------------------
def rank_report(scn: Scenario, config: str) -> dict:
    """dim ker(H) plus the evidence that the number is trustworthy."""
    x = scn.truth()
    factors = scn.factors(config)
    n = state_dim(scn.n_agents, scn.n_keyframes)

    _, j = linearize_all(factors, x, scn.n_agents, scn.n_keyframes)
    sv = np.linalg.svd(j, compute_uv=False)
    sv_rel = sv / sv[0]

    rank = int((sv_rel > RANK_TOL).sum())
    kernel_dim = n - rank

    # the gap that makes the threshold uncontroversial
    below = sv_rel[sv_rel <= RANK_TOL]
    gap = (sv_rel[rank - 1] / below.max()) if below.size and rank else math.inf

    return {
        "config": config,
        "label": CONFIGS[config][0],
        "expected": CONFIGS[config][1],
        "state_dim": n,
        "n_factor_rows": j.shape[0],
        "rank": rank,
        "kernel_dim": kernel_dim,
        "smallest_kept_sv": float(sv_rel[rank - 1]) if rank else 0.0,
        "largest_dropped_sv": float(below.max()) if below.size else 0.0,
        "gap": float(gap),
        "singular_values": sv,
    }


def kernel_basis(scn: Scenario, config: str) -> np.ndarray:
    """Orthonormal basis of ker(H), as rows."""
    x = scn.truth()
    _, j = linearize_all(scn.factors(config), x, scn.n_agents, scn.n_keyframes)
    _, sv, vt = np.linalg.svd(j)
    n = state_dim(scn.n_agents, scn.n_keyframes)
    keep = n - int((sv / sv[0] > RANK_TOL).sum())
    return vt[n - keep:] if keep else np.zeros((0, n))


def residual_energy(scn: Scenario, config: str, direction: np.ndarray) -> float:
    """‖J d‖ / ‖d‖ — how much cost a unit move along `direction` incurs.

    Zero means the direction is free. This is the physically direct way to ask
    "is the swarm allowed to slide this way", with no rank threshold involved.
    """
    x = scn.truth()
    _, j = linearize_all(scn.factors(config), x, scn.n_agents, scn.n_keyframes)
    d = np.asarray(direction, dtype=float)
    return float(np.linalg.norm(j @ d) / np.linalg.norm(d))


# --------------------------------------------------------------------------
# Covariance along the (former) gauge directions
# --------------------------------------------------------------------------
def gauge_covariance(scn: Scenario, config: str,
                     prior_sigma: float = 1e3) -> dict[str, float]:
    """Variance of the gauge coefficient along each generator.

    An unobservable direction has infinite variance, which does not plot. So we
    add one deliberately useless prior — sigma = 1 km — to make H invertible in
    every configuration. Then:

      * a direction the data pins reads a small, data-driven variance
      * a direction the data leaves free reads the prior's variance, ~1e6

    The prior is six orders of magnitude weaker than any real measurement, so it
    cannot rescue an observable direction or contaminate one; it only turns
    "infinite" into "1e6" so bounded and unbounded land on one axis.

    Generators are taken about the anchor, because that is where the surviving
    rotational freedom actually lives.
    """
    x = scn.truth()
    factors = scn.factors(config) + [
        PriorFactor(sigma=prior_sigma, agent=0, t=0, sigma_theta=prior_sigma)
    ]
    h = information_matrix(factors, x, scn.n_agents, scn.n_keyframes)
    cov = np.linalg.inv(h)

    gens = gauge_generators(x, scn.n_agents, scn.n_keyframes, centre=scn.anchor)
    names = ["translation x", "translation y", "yaw about anchor"]
    out = {}
    for name, g in zip(names, gens):
        # delta_x = alpha * g  =>  var(alpha) = g^T Sigma g / (g^T g)^2
        gg = float(g @ g)
        out[name] = float(g @ cov @ g) / (gg * gg)
    return out


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
def print_ladder(scn: Scenario) -> list[dict]:
    rows = [rank_report(scn, c) for c in CONFIGS]
    n = rows[0]["state_dim"]

    print(f"\nAnchor null-space ladder   "
          f"({scn.n_agents} agents x {scn.n_keyframes} keyframes, dim x = {n})")
    print("=" * 100)
    print(f"{'configuration':<40}{'rank':>6}{'ker':>5}{'exp':>5}   {'why'}")
    print("-" * 100)
    for r in rows:
        ok = "" if r["kernel_dim"] == r["expected"] else "   <-- MISMATCH"
        print(f"{r['label']:<40}{r['rank']:>6}{r['kernel_dim']:>5}"
              f"{r['expected']:>5}   {CONFIGS[r['config']][2]}{ok}")
    print("=" * 100)
    worst = min(r["gap"] for r in rows)
    print(f"smallest singular-value gap across all rows: {worst:.1e}  "
          f"(rank threshold {RANK_TOL:g} is nowhere near load-bearing)")
    return rows


def print_free_directions(scn: Scenario) -> None:
    """Name the surviving freedom in the range-only case, rather than just
    counting it: project ker(H) onto rotation-about-the-anchor."""
    x = scn.truth()
    g_anchor = gauge_generators(scn.truth(), scn.n_agents, scn.n_keyframes,
                                centre=scn.anchor)[2]
    g_anchor = g_anchor / np.linalg.norm(g_anchor)

    print("\nWhat is still free, and what does it rotate about?")
    print("=" * 100)
    for c in ("mesh_static", "mesh_motion", "mesh_body_bearing",
              "mesh_anchor_bearing", "two_anchors_range"):
        k = kernel_basis(scn, c)
        if k.shape[0] == 0:
            print(f"{CONFIGS[c][0]:<40}  nothing free — full rank")
            continue
        # |cos| between the surviving kernel and rotation-about-the-anchor
        align = float(np.linalg.norm(k @ g_anchor))
        e = residual_energy(scn, c, g_anchor)
        print(f"{CONFIGS[c][0]:<40}  dim {k.shape[0]}, "
              f"alignment with rotation-about-anchor = {align:.9f}, "
              f"‖J n‖/‖n‖ = {e:.1e}")
    print("=" * 100)

    # and the contrast that makes it a *surveyed anchor* result
    g_origin = gauge_generators(x, scn.n_agents, scn.n_keyframes)[2]
    e_origin = residual_energy(scn, "mesh_motion", g_origin)
    print(f"rotation about the map ORIGIN instead: ‖J n‖/‖n‖ = {e_origin:.3e} — "
          f"NOT free.\nThe pivot is the anchor itself, which is the whole point: "
          f"the surviving\nfreedom is anchored to a surveyed object, not to an "
          f"arbitrary coordinate choice.")


def centroid_uncertainty(scn: Scenario, config: str,
                         prior_sigma: float = 1e3) -> dict[str, float]:
    """Radial / tangential position uncertainty of the swarm centroid, measured
    about the anchor.

    This is the practically important consequence of a surviving yaw freedom,
    and the reason the raw gauge-variance table looks confusing at first: with a
    range-only anchor the rank says translation is "recovered", but the swarm
    can still swing *tangentially* around the anchor at no cost. Radially it is
    pinned to centimetres; tangentially it is free, and the resulting position
    error scales with the anchor-to-swarm distance.

    Ranges tell you how far away you are. Only a bearing (or a second anchor)
    tells you which way round.
    """
    x = scn.truth()
    factors = scn.factors(config) + [
        PriorFactor(sigma=prior_sigma, agent=0, t=0, sigma_theta=prior_sigma)
    ]
    h = information_matrix(factors, x, scn.n_agents, scn.n_keyframes)
    cov = np.linalg.inv(h)

    # centroid of the final keyframe
    t = scn.n_keyframes - 1
    n = state_dim(scn.n_agents, scn.n_keyframes)
    a = np.zeros((2, n))
    for i in range(scn.n_agents):
        k = 3 * (i * scn.n_keyframes + t)
        a[0, k] = a[1, k + 1] = 1.0 / scn.n_agents
    c2 = a @ cov @ a.T

    centroid = np.array([sum(pose(x, i, t, scn.n_keyframes)[0]
                             for i in range(scn.n_agents))]).ravel() / scn.n_agents
    d = centroid - scn.anchor
    r_hat = d / np.linalg.norm(d)
    t_hat = np.array([-r_hat[1], r_hat[0]])

    return {
        "radial_m": float(math.sqrt(r_hat @ c2 @ r_hat)),
        "tangential_m": float(math.sqrt(t_hat @ c2 @ t_hat)),
        "anchor_dist_m": float(np.linalg.norm(d)),
    }


def print_centroid(scn: Scenario) -> dict[str, dict[str, float]]:
    print("\nSwarm-centroid position uncertainty, resolved about the anchor")
    print("=" * 100)
    print(f"{'configuration':<40}{'sigma radial [m]':>20}"
          f"{'sigma tangential [m]':>22}")
    print("-" * 100)
    out = {}
    for c in CONFIGS:
        v = centroid_uncertainty(scn, c)
        out[c] = v
        print(f"{CONFIGS[c][0]:<40}{v['radial_m']:>20.3e}"
              f"{v['tangential_m']:>22.3e}")
    print("=" * 100)
    d = out["mesh_motion"]["anchor_dist_m"]
    print(f"centroid sits {d:.1f} m from the anchor. Range-only pins the RADIAL "
          f"direction to\ncentimetres and leaves the TANGENTIAL one free — you "
          f"know how far away you are,\nnot which way round. That is the yaw "
          f"null-space showing up as a position error.")
    return out


def print_covariance(scn: Scenario) -> dict[str, dict[str, float]]:
    print("\nVariance along each gauge direction "
          "(weak 1 km prior added so every row is invertible)")
    print("=" * 100)
    print(f"{'configuration':<40}{'trans x [m2]':>16}{'trans y [m2]':>16}"
          f"{'yaw@anchor [rad2]':>20}")
    print("-" * 100)
    out = {}
    for c in CONFIGS:
        v = gauge_covariance(scn, c)
        out[c] = v
        print(f"{CONFIGS[c][0]:<40}{v['translation x']:>16.2e}"
              f"{v['translation y']:>16.2e}{v['yaw about anchor']:>20.2e}")
    print("=" * 100)
    print("~1e5-1e7 = the prior talking, i.e. the data says nothing "
          "(UNBOUNDED).\n<1e-2      = pinned by measurement (BOUNDED).")
    return out


def main() -> None:
    scn = Scenario()
    rows = print_ladder(scn)
    print_free_directions(scn)
    print_covariance(scn)
    centroid = print_centroid(scn)

    from hive.plots import plot_nullspace_study
    path = plot_nullspace_study(scn, rows, centroid)
    print(f"\nfigure -> {path}")


if __name__ == "__main__":
    main()
