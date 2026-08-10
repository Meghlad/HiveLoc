"""D1.1 — factors and gauge generators for the anchored planar estimator.

Everything here exists to answer one question numerically: *what does a single
surveyed anchor actually buy you?* So each factor carries its analytic Jacobian
(checked against finite differences in the tests) and the module also supplies
the SE(2) gauge generators, so we can ask whether a given factor set kills them.

STATE LAYOUT. Agent i in [0, M), keyframe t in [0, T), each an SE(2) pose
(x, y, theta). Flattened, agent-major:

    index(i, t) = 3 * (i * T + t)        ->  [x, y, theta]

Heading is in the state even though the plan's study is about *position*
observability. It has to be: without it we cannot write an odometry factor that
is genuinely invariant to global rotation, and a residual that quietly breaks
that invariance would hand us an observable yaw for free and invalidate the
entire null-space result. The gauge argument is only as honest as the odometry
model underneath it.

WHICH FACTORS ARE GAUGE-INVARIANT (the whole point):

  odometry (body-frame)      invariant to global SE(2)   -> carries the gauge
  inter-agent range          invariant to global SE(2)   -> carries the gauge
  inter-agent bearing (body) invariant to global SE(2)   -> carries the gauge
  anchor range               NOT translation-invariant   -> kills tx, ty
  anchor bearing, ANCHOR frame   NOT rotation-invariant  -> kills yaw
  anchor bearing, BODY frame     invariant to rotation
                                 about the anchor        -> kills nothing extra

That last pair is the subtle one and it is why both are implemented. A bearing
is only external information if it is expressed in an externally-known
orientation. The surveyed ground station is bolted down and its heading is
surveyed along with its position, so `AnchorBearingFactor(frame="anchor")` is
real information. An AoA antenna on the *vehicle* measures the anchor in the
vehicle's own drifting body frame; rotate the whole world about the anchor and
that measurement does not change. See `nullspace.py` for the numbers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# --------------------------------------------------------------------------
# State indexing
# --------------------------------------------------------------------------


def state_dim(n_agents: int, n_keyframes: int) -> int:
    return 3 * n_agents * n_keyframes


def index(i: int, t: int, n_keyframes: int) -> int:
    """First scalar index of agent i's pose at keyframe t."""
    return 3 * (i * n_keyframes + t)


def pose(x: np.ndarray, i: int, t: int, n_keyframes: int) -> tuple[np.ndarray, float]:
    k = index(i, t, n_keyframes)
    return x[k:k + 2], float(x[k + 2])


def wrap(a: float) -> float:
    """Angle to (-pi, pi]. Every angular residual goes through this."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def rot(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


def d_rot_t(theta: float) -> np.ndarray:
    """d/dtheta of R(theta)^T."""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[-s, c], [-c, -s]])


def _bearing_grad(d: np.ndarray) -> np.ndarray:
    """d/dd of atan2(d_y, d_x)."""
    n2 = float(d @ d)
    return np.array([-d[1], d[0]]) / n2


# --------------------------------------------------------------------------
# Factors
# --------------------------------------------------------------------------
@dataclass
class Factor:
    """Base: a residual block with an analytic Jacobian and a noise sigma.

    `linearize` returns (r, J) already whitened by sigma, so stacking factors
    and forming H = J^T J is the information matrix with no extra bookkeeping.
    """

    sigma: float = 1.0

    def dim(self) -> int:                                  # pragma: no cover
        raise NotImplementedError

    def residual(self, x: np.ndarray, nk: int) -> np.ndarray:   # pragma: no cover
        raise NotImplementedError

    def jacobian(self, x: np.ndarray, nk: int, n: int) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def linearize(self, x: np.ndarray, nk: int, n: int) -> tuple[np.ndarray, np.ndarray]:
        w = 1.0 / self.sigma
        return w * self.residual(x, nk), w * self.jacobian(x, nk, n)


@dataclass
class OdometryFactor(Factor):
    """Body-frame relative pose between consecutive keyframes of one agent.

    This is the VIO / IMU-preintegration stand-in. Modelled in the *previous
    body frame*, which is what makes it invariant to a global SE(2) shift of
    the whole solution — the property that creates the gauge freedom the anchor
    then has to remove.
    """

    agent: int = 0
    t0: int = 0
    delta_body: np.ndarray = field(default_factory=lambda: np.zeros(2))
    delta_theta: float = 0.0
    sigma_theta: float = 0.01

    def dim(self) -> int:
        return 3

    def residual(self, x, nk):
        p0, th0 = pose(x, self.agent, self.t0, nk)
        p1, th1 = pose(x, self.agent, self.t0 + 1, nk)
        r_pos = rot(th0).T @ (p1 - p0) - self.delta_body
        r_th = wrap(th1 - th0 - self.delta_theta)
        return np.array([r_pos[0], r_pos[1], r_th])

    def jacobian(self, x, nk, n):
        p0, th0 = pose(x, self.agent, self.t0, nk)
        p1, _ = pose(x, self.agent, self.t0 + 1, nk)
        j = np.zeros((3, n))
        a = index(self.agent, self.t0, nk)
        b = index(self.agent, self.t0 + 1, nk)
        rt = rot(th0).T

        j[0:2, a:a + 2] = -rt
        j[0:2, a + 2] = d_rot_t(th0) @ (p1 - p0)
        j[0:2, b:b + 2] = rt
        j[2, a + 2] = -1.0
        j[2, b + 2] = 1.0
        return j

    def linearize(self, x, nk, n):
        # position and heading rows carry different units, so whiten per-row
        r = self.residual(x, nk)
        j = self.jacobian(x, nk, n)
        w = np.array([1.0 / self.sigma, 1.0 / self.sigma, 1.0 / self.sigma_theta])
        return w * r, w[:, None] * j


@dataclass
class InterAgentRangeFactor(Factor):
    """UWB range between two agents at the same keyframe. Gauge-invariant."""

    a: int = 0
    b: int = 1
    t: int = 0
    meas: float = 0.0

    def dim(self) -> int:
        return 1

    def residual(self, x, nk):
        pa, _ = pose(x, self.a, self.t, nk)
        pb, _ = pose(x, self.b, self.t, nk)
        return np.array([np.linalg.norm(pa - pb) - self.meas])

    def jacobian(self, x, nk, n):
        pa, _ = pose(x, self.a, self.t, nk)
        pb, _ = pose(x, self.b, self.t, nk)
        d = pa - pb
        u = d / np.linalg.norm(d)
        j = np.zeros((1, n))
        ia, ib = index(self.a, self.t, nk), index(self.b, self.t, nk)
        j[0, ia:ia + 2] = u
        j[0, ib:ib + 2] = -u
        return j


@dataclass
class InterAgentBearingFactor(Factor):
    """Agent `a` sees agent `b` in a's own body frame. Gauge-invariant."""

    a: int = 0
    b: int = 1
    t: int = 0
    meas: float = 0.0

    def dim(self) -> int:
        return 1

    def residual(self, x, nk):
        pa, tha = pose(x, self.a, self.t, nk)
        pb, _ = pose(x, self.b, self.t, nk)
        d = rot(tha).T @ (pb - pa)
        return np.array([wrap(math.atan2(d[1], d[0]) - self.meas)])

    def jacobian(self, x, nk, n):
        pa, tha = pose(x, self.a, self.t, nk)
        pb, _ = pose(x, self.b, self.t, nk)
        rt = rot(tha).T
        d = rt @ (pb - pa)
        g = _bearing_grad(d)

        j = np.zeros((1, n))
        ia, ib = index(self.a, self.t, nk), index(self.b, self.t, nk)
        j[0, ib:ib + 2] = g @ rt
        j[0, ia:ia + 2] = -(g @ rt)
        j[0, ia + 2] = g @ (d_rot_t(tha) @ (pb - pa))
        return j


@dataclass
class AnchorRangeFactor(Factor):
    """Range from the surveyed anchor to an agent.

    Not translation-invariant: shifting the whole solution changes every anchor
    range. Kills the two translation generators. Leaves rotation *about the
    anchor* untouched, because that motion preserves every distance to A.
    """

    anchor: np.ndarray = field(default_factory=lambda: np.zeros(2))
    agent: int = 0
    t: int = 0
    meas: float = 0.0

    def dim(self) -> int:
        return 1

    def residual(self, x, nk):
        p, _ = pose(x, self.agent, self.t, nk)
        return np.array([np.linalg.norm(p - self.anchor) - self.meas])

    def jacobian(self, x, nk, n):
        p, _ = pose(x, self.agent, self.t, nk)
        d = p - self.anchor
        j = np.zeros((1, n))
        k = index(self.agent, self.t, nk)
        j[0, k:k + 2] = d / np.linalg.norm(d)
        return j


@dataclass
class AnchorBearingFactor(Factor):
    """Bearing between the anchor and an agent. **Frame matters.**

    frame="anchor"  (default)
        The surveyed ground station reports the agent's bearing in its own
        surveyed, non-drifting frame: beta = atan2(p_i - A) in TacFrame.
        Rotating the world about A rotates (p_i - A), so beta changes ->
        this factor kills the yaw generator. Real external information.

    frame="body"
        An AoA antenna on the *vehicle* reports the anchor's bearing in the
        vehicle's own body frame: beta = atan2(R(theta_i)^T (A - p_i)).
        Rotate the world about A and theta_i picks up exactly the rotation the
        vector did, so beta is unchanged -> this factor adds NO yaw
        information. Implemented so the study can demonstrate that rather than
        assert it.
    """

    anchor: np.ndarray = field(default_factory=lambda: np.zeros(2))
    agent: int = 0
    t: int = 0
    meas: float = 0.0
    frame: str = "anchor"

    def __post_init__(self):
        if self.frame not in ("anchor", "body"):
            raise ValueError(f"frame must be 'anchor' or 'body', got {self.frame!r}")

    def dim(self) -> int:
        return 1

    def residual(self, x, nk):
        p, th = pose(x, self.agent, self.t, nk)
        if self.frame == "anchor":
            d = p - self.anchor
        else:
            d = rot(th).T @ (self.anchor - p)
        return np.array([wrap(math.atan2(d[1], d[0]) - self.meas)])

    def jacobian(self, x, nk, n):
        p, th = pose(x, self.agent, self.t, nk)
        j = np.zeros((1, n))
        k = index(self.agent, self.t, nk)

        if self.frame == "anchor":
            d = p - self.anchor
            j[0, k:k + 2] = _bearing_grad(d)          # no dependence on theta
        else:
            rt = rot(th).T
            d = rt @ (self.anchor - p)
            g = _bearing_grad(d)
            j[0, k:k + 2] = -(g @ rt)
            j[0, k + 2] = g @ (d_rot_t(th) @ (self.anchor - p))
        return j


@dataclass
class PriorFactor(Factor):
    """Weak isotropic prior on one pose. Only used to regularise H so a
    covariance can be formed at all; set sigma huge (1e3 m) so it contributes
    no real information and the gauge study still reads the data, not the
    prior. See `nullspace.gauge_covariance`."""

    agent: int = 0
    t: int = 0
    sigma_theta: float = 1e3

    def dim(self) -> int:
        return 3

    def residual(self, x, nk):
        p, th = pose(x, self.agent, self.t, nk)
        return np.array([p[0], p[1], th])

    def jacobian(self, x, nk, n):
        j = np.zeros((3, n))
        k = index(self.agent, self.t, nk)
        j[0, k] = j[1, k + 1] = j[2, k + 2] = 1.0
        return j

    def linearize(self, x, nk, n):
        r = self.residual(x, nk)
        j = self.jacobian(x, nk, n)
        w = np.array([1.0 / self.sigma, 1.0 / self.sigma, 1.0 / self.sigma_theta])
        return w * r, w[:, None] * j


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------


def linearize_all(factors, x, n_agents, n_keyframes):
    """Stack every factor into one whitened (r, J)."""
    n = state_dim(n_agents, n_keyframes)
    rs, js = [], []
    for f in factors:
        r, j = f.linearize(x, n_keyframes, n)
        rs.append(np.atleast_1d(r))
        js.append(np.atleast_2d(j))
    return np.concatenate(rs), np.vstack(js)


def information_matrix(factors, x, n_agents, n_keyframes) -> np.ndarray:
    """H = J^T J with J already whitened. Symmetric PSD by construction."""
    _, j = linearize_all(factors, x, n_agents, n_keyframes)
    h = j.T @ j
    return 0.5 * (h + h.T)          # kill float asymmetry so eigvalsh is clean


# --------------------------------------------------------------------------
# Gauge generators
# --------------------------------------------------------------------------


def gauge_generators(x: np.ndarray, n_agents: int, n_keyframes: int,
                     centre: np.ndarray | None = None) -> np.ndarray:
    """The three infinitesimal global SE(2) motions, as rows of a (3, N) array.

    A gauge-invariant factor set has all three in its null space; each row is a
    direction along which the solution can slide with zero cost.

      row 0  global +x translation
      row 1  global +y translation
      row 2  global rotation about `centre` (default: the origin)

    The choice of `centre` only mixes row 2 with rows 0-1 — the *span* is the
    same 3-D space either way. It is exposed because when a factor set leaves
    exactly one direction free, that direction is rotation about a specific
    point, and naming that point is the physical result: with a single range-only
    anchor it is the anchor itself.
    """
    c = np.zeros(2) if centre is None else np.asarray(centre, dtype=float)
    n = state_dim(n_agents, n_keyframes)
    g = np.zeros((3, n))

    for i in range(n_agents):
        for t in range(n_keyframes):
            k = index(i, t, n_keyframes)
            p, _ = pose(x, i, t, n_keyframes)
            g[0, k] = 1.0                              # translate +x
            g[1, k + 1] = 1.0                          # translate +y
            d = p - c
            g[2, k] = -d[1]                            # rotate about c
            g[2, k + 1] = d[0]
            g[2, k + 2] = 1.0                          # heading turns with it
    return g


def normalize_rows(m: np.ndarray) -> np.ndarray:
    return m / np.linalg.norm(m, axis=1, keepdims=True)
