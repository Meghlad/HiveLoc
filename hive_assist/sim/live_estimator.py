"""D4.3 — Domain 1's estimator, running live over N vehicles.

This is `hive/anchored_isam2.py`'s factor graph with the mission's data flowing
through it instead of a scripted world. The factor TYPES are unchanged, which is
the point: the thing that was proven offline is the thing that flies.

    J(x) = SUM ||e_motion||^2_Sm     constant-velocity prior, p_{t-2} - 2p_{t-1} + p_t
         + SUM ||e_range(i,j)||^2    inter-agent range, Huber
         + SUM ||e_anchor(i,A)||^2   surveyed anchor -> agent range, Huber   <- the pin

WHAT IS NOT HERE, AND WHY THAT IS THE WHOLE RESTRUCTURE. No IMU preintegration.
No RGB-D BetweenFactorPose3. No 15-DOF NavState. No attitude, no gravity, no
bias states. The autopilot's EKF3 already fuses IMU, attitude and gravity, and
does it well; this estimator's sole output is a POSITION in TacFrame, which EKF3
consumes as VISION_POSITION_ESTIMATE. Every red row in the old forensic log —
the 14.9 m in-flight lie at 10 cm sigma, the 7 m of reported motion on a
stationary airframe — came from the vision path that used to sit where this
comment is. Ranges are geometry. There is nothing here to hallucinate with.

THE MOTION PRIOR IS A SLOT, NOT A CLAIM. `e_motion` is a soft
constant-velocity constraint: "you did not teleport". It is what connects
keyframes and keeps an under-constrained frame non-singular, and it is
deliberately loose (sigma_motion defaults to 10 cm, an order of magnitude above
the range sigma) so that it shapes and never dominates. On real hardware it is
swapped for UWB-rate odometry behind the same factor slot without touching
anything else in this file.

COLD START IS COLD. `step()` does not need to be told where the vehicles are.
The first frame is multilaterated from anchor ranges alone (`multilaterate`
below), so the estimator earns its initial fix from measurements rather than
being handed the answer it is then scored against. Passing `init_xy` is
supported for the at-rest calibration case and is off by default.

GRAPH GROWTH. iSAM2 is incremental but the graph is not bounded, and this build
of gtsam exposes neither `marginalizeLeaves` nor `IncrementalFixedLagSmoother`.
So the bound is enforced by REBUILD: past `window` keyframes the smoother is
reconstructed over the retained window, seeded with a Gaussian prior on the
oldest retained keyframe taken from its own marginal covariance. That is the
standard fixed-lag approximation and it has a standard cost — the prior treats
the discarded past as independent, so cross-covariances with it are dropped and
the retained window is very slightly over-confident at the seam. Measured on the
stationary regression, the seam moves the reported sigma by under a millimetre;
it is written here because a reader deserves to know the approximation exists
rather than to find it.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

import gtsam
from gtsam.symbol_shorthand import X

from hive.supervisor_gate import EstimateSnapshot

# A vehicle whose marginal is indeterminate has no honest variance. It gets a
# large FINITE sentinel rather than inf: the supervisor's covariance gate then
# excludes exactly that vehicle (CovarianceTooHigh), whereas inf would be
# serialised as a bare `Infinity` token that serde_json refuses, taking the whole
# fleet's estimate down with a parse error. See hive/supervisor_io.py.
INDETERMINATE_TRACE = 1.0e6


@dataclass
class EstimatorConfig:
    """Sigmas in metres. Defaults match the range world's radio."""

    sigma_range: float = 0.015
    sigma_anchor_range: float = 0.015
    sigma_motion: float = 0.10
    huber_k: float = 1.345

    # Weak enough to inform nothing (the operating area is ~50 m across), strong
    # enough that a keyframe in which some vehicle loses every single link is
    # still a determined system rather than an iSAM2 IndeterminantLinearSystem.
    # This is the Brain's day-8 safety net, at metre scale.
    safety_prior_sigma: float = 25.0

    # Fixed-lag window. `window` is the size the graph is allowed to reach;
    # `retain` is what survives a trim. They are separate numbers because
    # setting retain = window makes the smoother rebuild on EVERY keyframe
    # (append one, exceed by one, trim one), which measured 150 ms per step
    # against a 3.9 ms steady state — a 40x self-inflicted cost that looks
    # exactly like "iSAM2 is slow". Trimming to half amortises one rebuild over
    # `window - retain` cheap steps.
    #
    # 30 keyframes is ~3 s of lag at 10 Hz. A constant-velocity prior over
    # range measurements has no use for more: the information from a keyframe
    # three seconds old has already propagated into the current one, and every
    # extra keyframe is paid for in the rebuild.
    window: int = 30
    retain: int = 15
    relinearize_threshold: float = 0.01
    relinearize_skip: int = 1

    def __post_init__(self):
        if self.retain < 3:
            raise ValueError("retain must be >= 3 to keep a motion factor")
        if self.window <= self.retain:
            raise ValueError("window must exceed retain, or every step rebuilds")


def multilaterate(anchors, ranges, guess=None, iters: int = 12) -> np.ndarray:
    """Position from anchor ranges alone: Gauss-Newton on the range residual.

    `ranges` is [(anchor_index, metres), ...]. Needs three non-collinear
    anchors to be unique in the plane; with two it converges to whichever of the
    two intersections the initial guess is nearer, and with one it slides along
    the circle. The caller checks the count — this function does not pretend a
    rank-deficient problem has an answer, it just returns the least-squares
    point and lets the covariance downstream tell the truth about it.
    """
    a = np.asarray(anchors, dtype=float)
    if not ranges:
        return np.zeros(2) if guess is None else np.asarray(guess, float)
    p = (np.asarray(guess, dtype=float) if guess is not None
         else a[[k for k, _ in ranges]].mean(axis=0))
    for _ in range(iters):
        jac = np.zeros((len(ranges), 2))
        res = np.zeros(len(ranges))
        for row, (k, m) in enumerate(ranges):
            d = p - a[k]
            dist = float(np.linalg.norm(d)) + 1e-9
            jac[row] = d / dist
            res[row] = dist - m
        try:
            step, *_ = np.linalg.lstsq(jac, -res, rcond=None)
        except np.linalg.LinAlgError:            # pragma: no cover - defensive
            break
        p = p + step
        if float(np.linalg.norm(step)) < 1e-9:
            break
    return p


# --------------------------------------------------------------------------
# Factors — the same CustomFactor idiom as hive/anchored_isam2.py
# --------------------------------------------------------------------------
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


def anchor_range_factor(ki, anchor, meas, model):
    def err(this, v, h):
        d = v.atPoint2(ki) - anchor
        dist = float(np.linalg.norm(d)) + 1e-9
        if h is not None:
            h[0] = (d / dist).reshape(1, 2)
        return np.array([dist - meas])
    return gtsam.CustomFactor(model, [ki], err)


def motion_factor(k0, k1, k2, model):
    """p_{t-2} - 2 p_{t-1} + p_t = 0. Zero discrete acceleration, softly."""
    def err(this, v, h):
        p0, p1, p2 = v.atPoint2(k0), v.atPoint2(k1), v.atPoint2(k2)
        if h is not None:
            h[0], h[1], h[2] = np.eye(2), -2.0 * np.eye(2), np.eye(2)
        return p2 - 2.0 * p1 + p0
    return gtsam.CustomFactor(model, [k0, k1, k2], err)


def prior_factor(ki, p0, model):
    def err(this, v, h):
        if h is not None:
            h[0] = np.eye(2)
        return v.atPoint2(ki) - p0
    return gtsam.CustomFactor(model, [ki], err)


# --------------------------------------------------------------------------
# The live estimator
# --------------------------------------------------------------------------
@dataclass
class StepTiming:
    build_ms: float = 0.0
    update_ms: float = 0.0
    marginal_ms: float = 0.0
    rebuild_ms: float = 0.0          # non-zero only on a trim keyframe

    @property
    def total_ms(self) -> float:
        return self.build_ms + self.update_ms + self.marginal_ms + self.rebuild_ms


class LiveEstimator:
    """iSAM2 over N live vehicles. One `step()` per range frame."""

    def __init__(self, n: int, anchors, cfg: EstimatorConfig | None = None,
                 init_xy=None) -> None:
        self.n = int(n)
        self.anchors = np.asarray(anchors, dtype=float).reshape(-1, 2)
        self.cfg = cfg or EstimatorConfig()

        self.t = -1                                  # keyframe index
        self.pos = np.full((self.n, 2), np.nan)
        self.cov_trace = np.full(self.n, INDETERMINATE_TRACE)
        self.timing = StepTiming()
        self.rebuilds = 0

        self._seed = (None if init_xy is None
                      else np.asarray(init_xy, dtype=float).reshape(self.n, 2))
        self._history: list[np.ndarray] = []          # retained keyframe estimates
        self._frames: list = []                       # retained measurement frames
        self._isam = self._new_isam()
        self._window_start = 0

        self._models()

    # -- setup ------------------------------------------------------------
    def _models(self) -> None:
        c = self.cfg
        self.m_range = gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber.Create(c.huber_k),
            gtsam.noiseModel.Isotropic.Sigma(1, c.sigma_range))
        self.m_anchor = gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber.Create(c.huber_k),
            gtsam.noiseModel.Isotropic.Sigma(1, c.sigma_anchor_range))
        self.m_motion = gtsam.noiseModel.Isotropic.Sigma(2, c.sigma_motion)
        self.m_safety = gtsam.noiseModel.Isotropic.Sigma(2, c.safety_prior_sigma)

    def _new_isam(self) -> gtsam.ISAM2:
        p = gtsam.ISAM2Params()
        p.setRelinearizeThreshold(self.cfg.relinearize_threshold)
        p.relinearizeSkip = self.cfg.relinearize_skip
        return gtsam.ISAM2(p)

    def key(self, i: int, t: int) -> int:
        return X(t * self.n + i)

    # -- prediction -------------------------------------------------------
    def _predict(self, i: int) -> np.ndarray:
        """Where vehicle i probably is now, before this frame is used.

        Constant velocity from the last two estimates — the same extrapolation
        the motion factor encodes, so the linearisation point and the prior
        agree. Disagreeing costs an iteration on every keyframe.
        """
        if len(self._history) >= 2:
            return 2.0 * self._history[-1][i] - self._history[-2][i]
        if len(self._history) == 1:
            return self._history[-1][i].copy()
        if self._seed is not None:
            return self._seed[i].copy()
        return np.zeros(2)

    def _cold_start(self, frame) -> np.ndarray:
        """First keyframe: multilaterate each vehicle from its anchor ranges."""
        by_vehicle: dict[int, list] = {i: [] for i in range(self.n)}
        for i, k, m in frame.anchor:
            by_vehicle[i].append((k, m))
        out = np.zeros((self.n, 2))
        for i in range(self.n):
            rows = by_vehicle[i]
            seed = self._seed[i] if self._seed is not None else None
            if len(rows) >= 2:
                out[i] = multilaterate(self.anchors, rows, guess=seed)
            elif seed is not None:
                out[i] = seed
            else:
                # Nothing to stand on. The centroid of the anchors is the least
                # committal guess; the safety prior is 25 m wide, so this is a
                # placeholder the very next keyframe will overwrite, not a claim.
                out[i] = self.anchors.mean(axis=0)
        return out

    # -- the tick ---------------------------------------------------------
    def step(self, frame, stamp_unix_ms: int | None = None) -> EstimateSnapshot:
        """One keyframe: measurements in, EstimateSnapshot out."""
        t0 = time.perf_counter()
        self.t += 1
        t = self.t

        if t == 0:
            guess = self._cold_start(frame)
        else:
            guess = np.array([self._predict(i) for i in range(self.n)])

        graph = gtsam.NonlinearFactorGraph()
        values = gtsam.Values()

        for i in range(self.n):
            k = self.key(i, t)
            values.insert(k, guess[i].astype(float))
            graph.add(prior_factor(k, guess[i].astype(float), self.m_safety))

        for i, j, m in frame.inter:
            graph.add(range_factor(self.key(i, t), self.key(j, t), m,
                                   self.m_range))
        for i, k_a, m in frame.anchor:
            graph.add(anchor_range_factor(self.key(i, t), self.anchors[k_a], m,
                                          self.m_anchor))

        if t - self._window_start >= 2:
            for i in range(self.n):
                graph.add(motion_factor(self.key(i, t - 2), self.key(i, t - 1),
                                        self.key(i, t), self.m_motion))

        t1 = time.perf_counter()
        self._isam.update(graph, values)
        t2 = time.perf_counter()

        est = self._isam.calculateEstimate()
        pos = np.array([est.atPoint2(self.key(i, t)) for i in range(self.n)])
        self.pos = pos

        for i in range(self.n):
            try:
                cov = self._isam.marginalCovariance(self.key(i, t))
                trace = float(np.trace(cov))
                self.cov_trace[i] = (trace if math.isfinite(trace) and trace >= 0
                                     else INDETERMINATE_TRACE)
            except Exception:
                # Indeterminate marginal: real, and the honest report is "I do
                # not know", which the gate reads as "do not command this one".
                self.cov_trace[i] = INDETERMINATE_TRACE
        t3 = time.perf_counter()

        self._history.append(pos.copy())
        self._frames.append(frame)
        rebuild_ms = 0.0
        if len(self._history) > self.cfg.window:
            t4 = time.perf_counter()
            self._rebuild()
            rebuild_ms = (time.perf_counter() - t4) * 1e3
            self.rebuilds += 1

        self.timing = StepTiming(build_ms=(t1 - t0) * 1e3,
                                 update_ms=(t2 - t1) * 1e3,
                                 marginal_ms=(t3 - t2) * 1e3,
                                 rebuild_ms=rebuild_ms)

        stamp = (int(time.time() * 1000) if stamp_unix_ms is None
                 else int(stamp_unix_ms))
        return EstimateSnapshot(
            frame_index=t,
            stamp_unix_ms=stamp,
            pos=[(float(p[0]), float(p[1])) for p in pos],
            cov_trace=[float(c) for c in self.cov_trace],
        )

    # -- fixed lag --------------------------------------------------------
    def _rebuild(self) -> None:
        """Reconstruct the smoother over the retained window.

        Keeps the last `window` keyframes' measurements and re-plays them into a
        fresh ISAM2, seeded with a Gaussian prior on the oldest retained
        keyframe drawn from its own marginal covariance. Keys are global
        (`t * n + i`) and are NOT reused, so nothing from the discarded past can
        collide with anything in the new window.
        """
        keep = self.cfg.retain
        drop = len(self._history) - keep
        if drop <= 0:
            return

        # Seed covariance must be captured BEFORE the old smoother is discarded.
        seed_t = self.t - keep + 1
        seed_cov = []
        for i in range(self.n):
            try:
                seed_cov.append(np.asarray(
                    self._isam.marginalCovariance(self.key(i, seed_t)), float))
            except Exception:
                seed_cov.append(np.eye(2) * (self.cfg.safety_prior_sigma ** 2))

        self._history = self._history[drop:]
        self._frames = self._frames[drop:]
        self._window_start = seed_t
        self._isam = self._new_isam()

        graph = gtsam.NonlinearFactorGraph()
        values = gtsam.Values()
        for idx, frame in enumerate(self._frames):
            t = seed_t + idx
            for i in range(self.n):
                k = self.key(i, t)
                values.insert(k, self._history[idx][i].astype(float))
                if idx == 0:
                    graph.add(prior_factor(
                        k, self._history[idx][i].astype(float),
                        gtsam.noiseModel.Gaussian.Covariance(seed_cov[i])))
                else:
                    graph.add(prior_factor(k, self._history[idx][i].astype(float),
                                           self.m_safety))
            for i, j, m in frame.inter:
                graph.add(range_factor(self.key(i, t), self.key(j, t), m,
                                       self.m_range))
            for i, k_a, m in frame.anchor:
                graph.add(anchor_range_factor(self.key(i, t),
                                              self.anchors[k_a], m, self.m_anchor))
            if idx >= 2:
                for i in range(self.n):
                    graph.add(motion_factor(self.key(i, t - 2),
                                            self.key(i, t - 1),
                                            self.key(i, t), self.m_motion))
        self._isam.update(graph, values)

    # -- reporting --------------------------------------------------------
    def sigma_m(self) -> np.ndarray:
        """Per-vehicle 1-sigma position uncertainty, metres per axis."""
        return np.sqrt(np.maximum(self.cov_trace, 0.0) / 2.0)

    def trusted(self, max_cov_trace: float) -> np.ndarray:
        return self.cov_trace <= max_cov_trace


def error_over_sigma(pos, truth, sigma) -> np.ndarray:
    """The acceptance ratio the plan insists on (§4.6): |err| / sigma.

    Near 1 means the estimator's confidence matches its accuracy. A SMALL ratio
    is not a better result, it is an estimator that is under-claiming; a large
    one is the failure the supervisor would happily certify — confidently wrong.
    That is the whole project's thesis, so it is a function rather than a
    print statement, and the tests assert on it.
    """
    err = np.linalg.norm(np.asarray(pos) - np.asarray(truth), axis=1)
    return err / np.maximum(np.asarray(sigma), 1e-12)


def main() -> None:
    """Offline check: a static 4-vehicle formation, no SITL, no sockets."""
    from sim.range_world import DEFAULT_ANCHORS, RangeWorld

    truth = np.array([[-4.0, -4.0], [4.0, -4.0], [4.0, 4.0], [-4.0, 4.0]])
    world = RangeWorld(seed=11)
    est = LiveEstimator(len(truth), DEFAULT_ANCHORS)

    print("D4.3 live estimator — 4 static vehicles, 4 anchors, cold start")
    print("=" * 78)
    print(f"{'k':>4}{'RMSE':>10}{'sigma':>10}{'ratio':>9}"
          f"{'build':>9}{'update':>9}{'marg':>8}")
    print("-" * 78)

    ratios, totals, rebuilds = [], [], []
    for k in range(200):
        est.step(world.measure(truth))
        r = error_over_sigma(est.pos, truth, est.sigma_m())
        rmse = float(np.sqrt(((est.pos - truth) ** 2).sum(axis=1).mean()))
        ratios.append(r.mean())
        totals.append(est.timing.total_ms)
        if est.timing.rebuild_ms:
            rebuilds.append(est.timing.rebuild_ms)
        if k % 40 == 0 or k == 199:
            print(f"{k:>4}{rmse:>9.3f}m{est.sigma_m().mean():>9.3f}m"
                  f"{r.mean():>9.2f}{est.timing.build_ms:>8.2f}ms"
                  f"{est.timing.update_ms:>8.2f}ms{est.timing.marginal_ms:>7.2f}ms")
    steps = np.array(totals)
    print("-" * 78)
    print(f"  final RMSE            {rmse:.3f} m")
    print(f"  final sigma           {est.sigma_m().mean():.3f} m")
    print(f"  err/sigma (mean)      {np.mean(ratios):.2f}   "
          f"<- 1.0 is calibrated; small is under-confident, large is a lie")
    print(f"  step p50 / p99        {np.percentile(steps, 50):.2f} / "
          f"{np.percentile(steps, 99):.2f} ms  (budget 25 ms end to end)")
    print(f"  fixed-lag rebuilds    {est.rebuilds} "
          f"(median {np.median(rebuilds) if rebuilds else 0:.1f} ms, "
          f"1 per {est.cfg.window - est.cfg.retain} keyframes)")
    print("=" * 78)


if __name__ == "__main__":
    main()
