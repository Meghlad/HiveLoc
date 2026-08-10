"""D1.3 — anchor factors inside the real incremental estimator.

`nullspace.py` proves the rank claim on a static Jacobian. That is necessary and
not sufficient: rank is a statement about one instant, and the failure this whole
domain exists to prevent is a *drift*, which only shows up over a mission. So
this module runs the same anchored graph through iSAM2, one keyframe at a time,
against the alternative the plan explicitly argues with:

  anchored     surveyed anchor supplies range + bearing at every keyframe
  pinned       the previous iteration's approach — declare drone 0 the origin,
               pin it with a tight prior, and let everything else float

Both are full-rank. Both produce a covariance. Only one of them stays attached to
the world, and that is the entire argument:

    a reference *drone* removes the SINGULARITY (a convention)
    a surveyed *anchor* removes the DRIFT (external information)

The pinned run is not badly implemented — it is correctly implemented and still
walks away, because relative measurements have nothing to walk away *from*. Its
covariance grows honestly while it drifts, which is the tell: the estimator knows
it is losing the world and has no measurement that could stop it. The anchored
run holds both flat, because the anchor re-supplies absolute information at every
single keyframe rather than once at t=0.

Reuses the Brain's `CustomFactor` idiom (see brain/src/estimation/day8_isam2.py)
so the two estimators stay recognisably the same code.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

import gtsam
from gtsam.symbol_shorthand import X


# --------------------------------------------------------------------------
# World
# --------------------------------------------------------------------------
@dataclass
class AnchoredWorld:
    """A loiter mesh orbiting the origin with a surveyed anchor off to one side.

    `odom_bias` is the important knob. Real VIO does not make zero-mean mistakes
    — scale error and heading error are systematic, so the integrated position
    walks in a preferred direction. Set it to zero and the pinned run merely
    diffuses; set it to a realistic value and it *leaves*, which is the failure
    mode worth showing.
    """

    n_agents: int = 8
    n_keyframes: int = 120
    ring_radius: float = 8.0
    omega: float = 0.035                 # rad of loiter per keyframe
    anchor: np.ndarray = field(default_factory=lambda: np.array([12.0, -5.0]))

    r_comm: float = 11.0                 # inter-agent ranging radius
    r_anchor: float = 30.0               # anchor ranging radius

    sigma_odom: float = 0.04
    sigma_range: float = 0.10
    sigma_anchor_range: float = 0.08
    sigma_anchor_bearing: float = 0.02   # rad, ~1.1 deg
    odom_bias: np.ndarray = field(default_factory=lambda: np.array([0.006, 0.004]))

    seed: int = 11

    def truth(self) -> np.ndarray:
        """(T, M, 2) ground-truth positions."""
        t = np.arange(self.n_keyframes)[:, None]
        phase = 2.0 * math.pi * np.arange(self.n_agents)[None, :] / self.n_agents
        a = phase + self.omega * t
        return np.stack([self.ring_radius * np.cos(a),
                         self.ring_radius * np.sin(a)], axis=-1)

    def measurements(self):
        """Per-keyframe measurement bundles, generated once and shared by both
        runs so the comparison isolates the anchor and nothing else."""
        rng = np.random.default_rng(self.seed)
        truth = self.truth()
        frames = []
        for t in range(self.n_keyframes):
            ranges = []
            for i in range(self.n_agents):
                for j in range(i + 1, self.n_agents):
                    d = float(np.linalg.norm(truth[t, i] - truth[t, j]))
                    if d <= self.r_comm:
                        ranges.append((i, j, d + rng.normal(0, self.sigma_range)))

            odom = []
            if t > 0:
                for i in range(self.n_agents):
                    d = truth[t, i] - truth[t - 1, i]
                    odom.append((i, d + self.odom_bias
                                 + rng.normal(0, self.sigma_odom, 2)))

            anchor_r, anchor_b = [], []
            for i in range(self.n_agents):
                d = truth[t, i] - self.anchor
                dist = float(np.linalg.norm(d))
                if dist <= self.r_anchor:
                    anchor_r.append((i, dist + rng.normal(0, self.sigma_anchor_range)))
                    anchor_b.append((i, math.atan2(d[1], d[0])
                                     + rng.normal(0, self.sigma_anchor_bearing)))
            frames.append({"range": ranges, "odom": odom,
                           "anchor_range": anchor_r, "anchor_bearing": anchor_b})
        return frames


# --------------------------------------------------------------------------
# Factors — same CustomFactor idiom as the Brain's day-8 estimator
# --------------------------------------------------------------------------
def _key(i: int, t: int, n_agents: int) -> int:
    return X(t * n_agents + i)


def range_factor(ki, kj, meas, model):
    def err(this, v, h):
        pi, pj = v.atPoint2(ki), v.atPoint2(kj)
        d = pi - pj
        dist = float(np.linalg.norm(d)) + 1e-9
        if h is not None:
            u = (d / dist).reshape(1, 2)
            h[0], h[1] = u, -u
        return np.array([dist - meas])
    return gtsam.CustomFactor(model, [ki, kj], err)


def odom_factor(k0, k1, delta, model):
    def err(this, v, h):
        p0, p1 = v.atPoint2(k0), v.atPoint2(k1)
        if h is not None:
            h[0], h[1] = -np.eye(2), np.eye(2)
        return (p1 - p0) - delta
    return gtsam.CustomFactor(model, [k0, k1], err)


def anchor_range_factor(ki, anchor, meas, model):
    """Kills global translation. Leaves rotation about the anchor free — which
    is exactly why the bearing factor below is not optional. See nullspace.py."""
    def err(this, v, h):
        d = v.atPoint2(ki) - anchor
        dist = float(np.linalg.norm(d)) + 1e-9
        if h is not None:
            h[0] = (d / dist).reshape(1, 2)
        return np.array([dist - meas])
    return gtsam.CustomFactor(model, [ki], err)


def anchor_bearing_factor(ki, anchor, meas, model):
    """The surveyed ground station reports the agent's bearing in its OWN fixed,
    surveyed frame. That is what makes it real yaw information — a vehicle-side
    AoA measurement in the vehicle's own frame would add nothing (proven in
    tests/test_nullspace.py). Depends only on position, so the graph stays
    Point2 exactly like the Brain's."""
    def err(this, v, h):
        d = v.atPoint2(ki) - anchor
        n2 = float(d @ d) + 1e-12
        if h is not None:
            h[0] = np.array([[-d[1], d[0]]]) / n2
        r = math.atan2(d[1], d[0]) - meas
        return np.array([(r + math.pi) % (2 * math.pi) - math.pi])
    return gtsam.CustomFactor(model, [ki], err)


def prior_factor(ki, p0, model):
    def err(this, v, h):
        if h is not None:
            h[0] = np.eye(2)
        return v.atPoint2(ki) - p0
    return gtsam.CustomFactor(model, [ki], err)


# --------------------------------------------------------------------------
# The incremental run
# --------------------------------------------------------------------------
# A prior this weak (50 m on an 8 m ring) cannot inform anything. It exists only
# so a keyframe where an agent loses every link still has a determined system,
# the same safety net the Brain's day-8 estimator uses.
SAFETY_PRIOR_SIGMA = 50.0
# The "declare drone 0 the origin" convention, applied once at t = 0.
PIN_SIGMA = 0.01


def run(world: AnchoredWorld, mode: str, frames=None) -> dict:
    """Run iSAM2 over the mission.

    mode="anchored"  anchor range + surveyed bearing at every keyframe
    mode="pinned"    tight prior on agent 0 at t=0, nothing else absolute
    """
    if mode not in ("anchored", "pinned"):
        raise ValueError(f"mode must be 'anchored' or 'pinned', got {mode!r}")

    frames = world.measurements() if frames is None else frames
    truth = world.truth()
    m, tt = world.n_agents, world.n_keyframes

    huber = gtsam.noiseModel.Robust.Create(
        gtsam.noiseModel.mEstimator.Huber.Create(1.345),
        gtsam.noiseModel.Isotropic.Sigma(1, world.sigma_range))
    odom_model = gtsam.noiseModel.Isotropic.Sigma(2, world.sigma_odom)
    ar_model = gtsam.noiseModel.Isotropic.Sigma(1, world.sigma_anchor_range)
    ab_model = gtsam.noiseModel.Isotropic.Sigma(1, world.sigma_anchor_bearing)
    safety = gtsam.noiseModel.Isotropic.Sigma(2, SAFETY_PRIOR_SIGMA)
    pin = gtsam.noiseModel.Isotropic.Sigma(2, PIN_SIGMA)

    isam = gtsam.ISAM2(gtsam.ISAM2Params())
    online = np.zeros((tt, m, 2))
    sigma = np.zeros(tt)
    update_ms = np.zeros(tt)

    for t in range(tt):
        f = frames[t]
        g = gtsam.NonlinearFactorGraph()
        vals = gtsam.Values()

        # introduce this keyframe's variables at the dead-reckoned guess
        odom_by_agent = dict(f["odom"])
        for i in range(m):
            if t == 0:
                guess = truth[0, i]                    # survey-time initialisation
            else:
                guess = online[t - 1, i] + odom_by_agent.get(i, np.zeros(2))
            vals.insert(_key(i, t, m), np.asarray(guess, dtype=float))
            g.add(prior_factor(_key(i, t, m), np.asarray(guess, float), safety))

        for (i, j, meas) in f["range"]:
            g.add(range_factor(_key(i, t, m), _key(j, t, m), meas, huber))

        for (i, delta) in f["odom"]:
            g.add(odom_factor(_key(i, t - 1, m), _key(i, t, m), delta, odom_model))

        if mode == "anchored":
            for (i, meas) in f["anchor_range"]:
                g.add(anchor_range_factor(_key(i, t, m), world.anchor, meas, ar_model))
            for (i, meas) in f["anchor_bearing"]:
                g.add(anchor_bearing_factor(_key(i, t, m), world.anchor, meas, ab_model))
        elif t == 0:
            # the convention: drone 0 IS the origin, by declaration
            g.add(prior_factor(_key(0, 0, m), truth[0, 0].astype(float), pin))

        t0 = time.perf_counter()
        isam.update(g, vals)
        update_ms[t] = (time.perf_counter() - t0) * 1e3

        est = isam.calculateEstimate()
        for i in range(m):
            online[t, i] = est.atPoint2(_key(i, t, m))

        # marginal sigma of a mid-swarm agent, as the estimator reports it
        try:
            cov = isam.marginalCovariance(_key(m // 2, t, m))
            sigma[t] = math.sqrt(max(np.trace(cov), 0.0) / 2.0)
        except Exception:                    # indeterminate: carry the last value
            sigma[t] = sigma[t - 1] if t else float("nan")

    err = np.linalg.norm(online - truth, axis=2)     # (T, M)
    return {
        "mode": mode,
        "online": online,
        "truth": truth,
        "rmse": np.sqrt((err ** 2).mean(axis=1)),
        "final_rmse": float(np.sqrt((err[-1] ** 2).mean())),
        "sigma": sigma,
        "update_ms": update_ms,
    }


def main() -> None:
    world = AnchoredWorld()
    frames = world.measurements()            # identical data for both runs

    res = {mode: run(world, mode, frames) for mode in ("pinned", "anchored")}

    print(f"\nAnchored iSAM2  —  {world.n_agents} agents, "
          f"{world.n_keyframes} keyframes, odom bias "
          f"{np.linalg.norm(world.odom_bias)*100:.1f} cm/keyframe")
    print("=" * 86)
    print(f"{'run':<34}{'final RMSE':>13}{'peak RMSE':>13}"
          f"{'final sigma':>14}{'median update':>14}")
    print("-" * 86)
    labels = {"pinned": "pinned (drone 0 = origin)",
              "anchored": "anchored (range + surveyed bearing)"}
    for mode in ("pinned", "anchored"):
        r = res[mode]
        print(f"{labels[mode]:<34}{r['final_rmse']:>12.3f}m"
              f"{r['rmse'].max():>12.3f}m{r['sigma'][-1]:>13.3f}m"
              f"{np.median(r['update_ms']):>12.2f}ms")
    print("=" * 86)

    ratio = res["pinned"]["final_rmse"] / max(res["anchored"]["final_rmse"], 1e-9)
    print(f"The pinned run ends {ratio:.0f}x further from truth. Both were "
          f"full-rank the whole way;\nonly one of them was attached to anything "
          f"external. Pinning fixes the gauge at t=0;\nthe anchor re-supplies it "
          f"at every keyframe — that is the difference the plan is about.")

    from hive.plots import plot_anchored_isam2
    t = np.arange(world.n_keyframes)
    path = plot_anchored_isam2(
        t, res["anchored"]["rmse"], res["pinned"]["rmse"],
        res["anchored"]["sigma"], res["pinned"]["sigma"],
        res["pinned"]["final_rmse"],
    )
    print(f"\nfigure -> {path}")


if __name__ == "__main__":
    main()
