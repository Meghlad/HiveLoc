"""D3 — event-triggered dispatch to a standoff station.

Dispatch is REACTIVE. When the external go-signal arrives, the agent that
receives it breaks off and flies an individual guided approach to a standoff
station near X_tac. There is no impact-time consensus, no simultaneity coupling,
and no homing onto the target point: the agent converges to a station on a
perimeter around X, at an approach geometry chosen for the task.

Explicitly NOT here, and not derived anywhere in this repo: terminal-homing-to-
point laws, PN with t_go bias, salvo timing, or any construction whose objective
is r_i -> 0 for several vehicles onto one coordinate. The station is a viewing
or working position at a standoff distance, the vehicle stops there, and a
second dispatch (if one ever happens) gets its own station and its own timing.

WHY DUBINS. The plan asks for a "fixed curvature-bounded path". Dubins paths are
that, exactly and provably — the shortest path between two poses subject to a
minimum turning radius. Anything smoother-looking (a spline through the two
poses) would have curvature we would then have to check and might have to
reject. Here the bound is structural.

Each of the six Dubins words is computed in closed form and then VERIFIED by
forward-integrating it and comparing the endpoint against the requested pose.
Words that do not reconstruct are discarded. That check costs microseconds and
means a sign error in one of the six formulas becomes "that word is never
chosen" instead of "the aircraft flies somewhere else".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from hive.supervisor_gate import StreamLimits, SupervisorConfig, check_stream

TWO_PI = 2.0 * math.pi


def mod2pi(a: float) -> float:
    return a % TWO_PI


def wrap(a: float) -> float:
    return (a + math.pi) % TWO_PI - math.pi


# --------------------------------------------------------------------------
# 3.1 Standoff station placement
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class TaskGeometry:
    """Standoff radius and approach bearing for one task type.

    The plan lists these as an open question, and they stay one: there is no
    derivation here, because the right numbers come from the sensor's working
    distance, the rotor wash, and the local rules of engagement — not from
    geometry. What this table does is make the choice explicit and per-task
    instead of a magic constant buried in the guidance loop.

    `bearing_deg` is measured in TacFrame, from +x, and says which side of the
    target the vehicle should sit on. `approach_deg` is the heading the vehicle
    should arrive with; None means "face the target", which is what an
    inspection wants and a delivery does not.
    """

    name: str
    standoff_m: float
    bearing_deg: float = 180.0
    approach_deg: float | None = None


TASKS: dict[str, TaskGeometry] = {
    # far enough to keep the whole subject in frame, facing it
    "inspection": TaskGeometry("inspection", standoff_m=12.0, bearing_deg=200.0),
    # close, facing the target, so the arm/probe reaches on a straight run-in
    "sampling": TaskGeometry("sampling", standoff_m=4.0, bearing_deg=170.0),
    # offset and arriving crosswind-ish, so the release is not over the target
    "delivery": TaskGeometry("delivery", standoff_m=6.0, bearing_deg=140.0,
                             approach_deg=90.0),
    "probe": TaskGeometry("probe", standoff_m=5.0, bearing_deg=225.0),
}


def standoff_station(x_tac, task: str | TaskGeometry) -> tuple[np.ndarray, float]:
    """Return (station, approach_heading) for a target in TacFrame.

    s_i = X_tac + d_s * [cos(theta), sin(theta)]   — a viewing point, not X.
    """
    g = TASKS[task] if isinstance(task, str) else task
    x = np.asarray(x_tac, dtype=float)[:2]
    th = math.radians(g.bearing_deg)
    station = x + g.standoff_m * np.array([math.cos(th), math.sin(th)])

    if g.approach_deg is None:
        # face the target from wherever the station ended up
        d = x - station
        heading = math.atan2(d[1], d[0])
    else:
        heading = math.radians(g.approach_deg)
    return station, wrap(heading)


def stations_for(x_tac, task: str | TaskGeometry, n: int, spread_deg: float = 45.0):
    """`n` distinct stations spread around the perimeter.

    For the plan's "a second signal arrives, a second agent dispatches" case.
    They are separate positions with separate approaches — spreading them is a
    collision and sensor-occlusion measure, NOT a coordinated arrival pattern.
    Nothing here couples their timing.
    """
    g = TASKS[task] if isinstance(task, str) else task
    if n == 1:
        return [standoff_station(x_tac, g)]
    offs = np.linspace(-spread_deg / 2, spread_deg / 2, n)
    out = []
    for o in offs:
        out.append(standoff_station(
            x_tac, TaskGeometry(g.name, g.standoff_m, g.bearing_deg + o,
                                g.approach_deg)))
    return out


# --------------------------------------------------------------------------
# Dubins path
# --------------------------------------------------------------------------
_WORDS = ("LSL", "RSR", "LSR", "RSL", "RLR", "LRL")


def _segment_end(pose, kind: str, length: float, radius: float):
    """Integrate one Dubins segment in closed form."""
    x, y, th = pose
    if kind == "S":
        return (x + length * math.cos(th), y + length * math.sin(th), th)
    turn = length / radius
    if kind == "L":
        nth = th + turn
        return (x + radius * (math.sin(nth) - math.sin(th)),
                y - radius * (math.cos(nth) - math.cos(th)), nth)
    nth = th - turn
    return (x - radius * (math.sin(nth) - math.sin(th)),
            y + radius * (math.cos(nth) - math.cos(th)), nth)


def _candidates(alpha: float, beta: float, d: float):
    """The six closed-form Dubins words, as (name, (t, p, q)) in units of R."""
    sa, ca = math.sin(alpha), math.cos(alpha)
    sb, cb = math.sin(beta), math.cos(beta)
    c_ab = math.cos(alpha - beta)
    out = []

    # LSL
    tmp = math.atan2(cb - ca, d + sa - sb)
    p_sq = 2 + d * d - 2 * c_ab + 2 * d * (sa - sb)
    if p_sq >= 0:
        out.append(("LSL", (mod2pi(-alpha + tmp), math.sqrt(p_sq),
                            mod2pi(beta - tmp))))
    # RSR
    tmp = math.atan2(ca - cb, d - sa + sb)
    p_sq = 2 + d * d - 2 * c_ab + 2 * d * (sb - sa)
    if p_sq >= 0:
        out.append(("RSR", (mod2pi(alpha - tmp), math.sqrt(p_sq),
                            mod2pi(-beta + tmp))))
    # LSR
    p_sq = -2 + d * d + 2 * c_ab + 2 * d * (sa + sb)
    if p_sq >= 0:
        p = math.sqrt(p_sq)
        tmp = math.atan2(-ca - cb, d + sa + sb) - math.atan2(-2.0, p)
        out.append(("LSR", (mod2pi(-alpha + tmp), p, mod2pi(-mod2pi(beta) + tmp))))
    # RSL
    p_sq = -2 + d * d + 2 * c_ab - 2 * d * (sa + sb)
    if p_sq >= 0:
        p = math.sqrt(p_sq)
        tmp = math.atan2(ca + cb, d - sa - sb) - math.atan2(2.0, p)
        out.append(("RSL", (mod2pi(alpha - tmp), p, mod2pi(beta - tmp))))
    # RLR
    tmp = (6.0 - d * d + 2 * c_ab + 2 * d * (sa - sb)) / 8.0
    if abs(tmp) <= 1.0:
        p = mod2pi(TWO_PI - math.acos(tmp))
        t = mod2pi(alpha - math.atan2(ca - cb, d - sa + sb) + p / 2.0)
        out.append(("RLR", (t, p, mod2pi(alpha - beta - t + p))))
    # LRL — the q term here is the one the endpoint verification caught. It was
    # written as `beta - alpha + 2p`, which reconstructs to the wrong pose, so
    # LRL was silently rejected and never selected across 3000 random pose
    # pairs while its mirror image RLR was chosen 142 times. That asymmetry is
    # what a correct implementation cannot produce.
    tmp = (6.0 - d * d + 2 * c_ab + 2 * d * (sb - sa)) / 8.0
    if abs(tmp) <= 1.0:
        p = mod2pi(TWO_PI - math.acos(tmp))
        t = mod2pi(-alpha + math.atan2(-ca + cb, d + sa - sb) + p / 2.0)
        out.append(("LRL", (t, p, mod2pi(mod2pi(beta) - alpha - t + p))))
    return out


@dataclass
class DubinsPath:
    """A verified curvature-bounded path from one pose to another."""

    start: tuple[float, float, float]
    end: tuple[float, float, float]
    radius: float
    word: str
    lengths: tuple[float, float, float]     # metres, in order

    @property
    def length(self) -> float:
        return float(sum(self.lengths))

    def sample(self, gamma: float) -> tuple[np.ndarray, float]:
        """Position and tangent heading at path parameter gamma in [0, 1]."""
        s = float(np.clip(gamma, 0.0, 1.0)) * self.length
        pose = self.start
        for kind, seg in zip(self.word, self.lengths):
            if s <= seg or seg <= 0:
                pose = _segment_end(pose, kind, min(s, seg), self.radius)
                s -= min(s, seg)
                if s <= 1e-12:
                    break
            else:
                pose = _segment_end(pose, kind, seg, self.radius)
                s -= seg
        return np.array([pose[0], pose[1]]), wrap(pose[2])

    def curvature(self, gamma: float) -> float:
        """Signed path curvature at gamma: +1/R on a left arc, -1/R on a right
        arc, 0 on the straight. Piecewise constant, which is exactly what makes
        a Dubins path cheap to feed forward."""
        s = float(np.clip(gamma, 0.0, 1.0)) * self.length
        for kind, seg in zip(self.word, self.lengths):
            if s <= seg or seg <= 0:
                return 0.0 if kind == "S" else \
                    (1.0 if kind == "L" else -1.0) / self.radius
            s -= seg
        kind = self.word[-1]
        return 0.0 if kind == "S" else (1.0 if kind == "L" else -1.0) / self.radius

    def polyline(self, n: int = 200) -> np.ndarray:
        return np.array([self.sample(g)[0] for g in np.linspace(0, 1, n)])


def walk_waypoints(paths, step_m: float) -> np.ndarray:
    """Resample Dubins paths into one (K, A, 2) ladder of gated waypoints.

    D3's `follow()` flies a path with its own integrator at `v_cruise`. The live
    loop cannot: it emits position setpoints at `plan_hz` and the supervisor
    rejects any tick that moves further than `StreamLimits.v_max`. So the curve
    has to be handed over as a SEQUENCE of goals, each within one tick's budget,
    rather than as a single distant goal — which is what makes the flown ground
    track an arc instead of the straight line a lone endpoint would produce.

    Every path is resampled to the SAME K. `sample()` is arc-length
    parameterised, so uniform gamma is uniform distance and K is chosen from the
    longest path; a shorter path simply takes smaller steps. Equal K is what
    keeps the coalition in lockstep — `step_toward` only advances a leg once
    EVERY vehicle has arrived, so unequal ladders would make the shorter path
    idle at its last rung waiting for the longer one.
    """
    if step_m <= 0:
        raise ValueError(f"step_m must be positive, got {step_m}")
    paths = list(paths)
    if not paths:
        raise ValueError("walk_waypoints needs at least one path")
    longest = max(p.length for p in paths)
    k = max(int(np.ceil(longest / step_m)) + 1, 2)
    gammas = np.linspace(0.0, 1.0, k)
    # (A, K, 2) -> (K, A, 2): the loop indexes by rung, not by agent.
    per_agent = np.array([[p.sample(g)[0] for g in gammas] for p in paths])
    return np.swapaxes(per_agent, 0, 1)


def dubins(start, end, radius: float, tol: float = 1e-6) -> DubinsPath:
    """Shortest curvature-bounded path between two poses.

    Every candidate word is reconstructed by forward integration and rejected if
    its endpoint does not match the requested pose. A closed-form Dubins
    implementation is a thicket of sign conventions; this way a mistake in one
    word costs a slightly longer path instead of a wrong trajectory.
    """
    x0, y0, th0 = float(start[0]), float(start[1]), float(start[2])
    x1, y1, th1 = float(end[0]), float(end[1]), float(end[2])

    dx, dy = x1 - x0, y1 - y0
    dist = math.hypot(dx, dy)
    if radius <= 0:
        raise ValueError("turning radius must be positive")

    course = math.atan2(dy, dx)
    d = dist / radius
    alpha = mod2pi(th0 - course)
    beta = mod2pi(th1 - course)

    best = None
    for word, segs in _candidates(alpha, beta, d):
        if any(s < -1e-9 or not math.isfinite(s) for s in segs):
            continue
        lengths = tuple(max(s, 0.0) * radius for s in segs)
        cand = DubinsPath((x0, y0, th0), (x1, y1, th1), radius, word, lengths)

        # verification: does this word actually land on the requested pose?
        pose = cand.start
        for kind, seg in zip(word, lengths):
            pose = _segment_end(pose, kind, seg, radius)
        if (math.hypot(pose[0] - x1, pose[1] - y1) > max(tol, tol * dist)
                or abs(wrap(pose[2] - th1)) > 1e-6):
            continue

        if best is None or cand.length < best.length:
            best = cand

    if best is None:                      # pragma: no cover - defensive
        raise RuntimeError(
            f"no Dubins word reconstructed for {start} -> {end} at R={radius}")
    return best


# --------------------------------------------------------------------------
# 3.2 Vector-field path following
# --------------------------------------------------------------------------
@dataclass
class GuidanceLimits:
    """Saturations. `v_min = 0` is the multirotor case, which is what this repo
    flies (ArduCopter) — the vehicle can stop on station. A fixed wing with
    v_min > 0 cannot, and would need a terminal holding orbit instead of an
    arrival; that variant is not implemented here."""

    v_min: float = 0.0
    v_max: float = 6.0
    v_cruise: float = 5.0
    omega_max: float = 1.2              # rad/s
    a_max: float = 3.0                  # m/s^2
    dt: float = 0.1


@dataclass
class GuidanceGains:
    k_e: float = 0.55                   # cross-track aggressiveness
    k_psi: float = 2.2                  # heading loop
    capture_m: float = 5.0              # switch to station capture inside this
    arrive_radius: float = 0.35         # "on station"


def _path_frame(path: DubinsPath, n: int = 600):
    """Precomputed samples of the path, for the reported cross-track error."""
    g = np.linspace(0.0, 1.0, n)
    pts = np.empty((n, 2))
    tan = np.empty(n)
    for i, gi in enumerate(g):
        p, th = path.sample(float(gi))
        pts[i] = p
        tan[i] = th
    return pts, tan


def cross_track_to_path(pts, tan, p) -> float:
    """Signed perpendicular distance from `p` to the NEAREST point on the path.

    This is what gets reported, and it is deliberately a different computation
    from the one the controller uses. The controller measures error against the
    reference at its own tracked gamma, which is correct for control and useless
    for reporting: at the follow/capture handover the controller changes which
    line it measures against, and the reported error jumped 1.3 m while the
    vehicle was in fact within 4 cm of the path the whole time. A metric that
    steps when nothing physical steps will eventually be read as a real
    excursion by someone, and one well-defined quantity beats two.
    """
    d = np.asarray(p, dtype=float) - pts
    i = int(np.argmin((d * d).sum(axis=1)))
    return float(-math.sin(tan[i]) * d[i, 0] + math.cos(tan[i]) * d[i, 1])


@dataclass
class ApproachResult:
    pos: np.ndarray                     # (T, 2)
    heading: np.ndarray                 # (T,)
    speed: np.ndarray                   # (T,)
    cross_track: np.ndarray             # (T,) signed, metres
    gamma: np.ndarray                   # (T,)
    arrived: bool
    arrive_index: int | None
    path: DubinsPath

    @property
    def final_error(self) -> float:
        return float(np.linalg.norm(self.pos[-1] - np.array(self.path.end[:2])))


def follow(path: DubinsPath, start_pose, station_heading: float,
           limits: GuidanceLimits | None = None,
           gains: GuidanceGains | None = None,
           max_steps: int = 4000) -> ApproachResult:
    """Vector-field path following, then station capture.

        psi_des = psi_tan(gamma) - arctan(k_e * e / v)      [follow phase]
        gamma_dot = v * cos(psi - psi_tan) / L

    TWO DEVIATIONS FROM THE PLAN'S EXPRESSIONS, BOTH DELIBERATE.

    1. `gamma_dot` carries a cos(psi - psi_tan) factor the plan's `v / L` does
       not. With pure `v / L` the reference advances at full speed while the
       vehicle is still turning to catch it, so early cross-track error makes
       the reference run away. The two agree exactly once aligned, which is the
       regime the plan's expression describes.

    2. There is a terminal CAPTURE phase, and without it this function is
       wrong. A vector field regulates cross-track error and nothing else. Run
       to gamma = 1 and the field happily steers the vehicle along the tangent
       line *through* the station and out the other side — measured, not
       hypothesised: the first version of this flew 1955 m past the station at
       full cruise, reporting a cross-track error of 0.000 m the entire way,
       because it was perfectly on a line that no longer had a destination on
       it. "e -> 0" is not the same claim as "arrives", and only one of them is
       what dispatch needs.

       So inside `capture_m` of the end, guidance switches to regulating
       position directly to the station under a sqrt(2*a*d) braking profile.
       The REPORTED cross-track error is measured against the nearest point on
       the path throughout (see `cross_track_to_path`), so it stays one
       continuous quantity across the switch.

    Single agent. No coupling term, no other vehicle's state in this function's
    signature at all — that is the design, not an omission.
    """
    limits = limits or GuidanceLimits()
    gains = gains or GuidanceGains()
    goal = np.array(path.end[:2], dtype=float)

    x, y, psi = float(start_pose[0]), float(start_pose[1]), float(start_pose[2])
    v = min(limits.v_cruise, limits.v_max)
    gamma = 0.0
    ref_pts, ref_tan = _path_frame(path)

    pos, heads, speeds, errs, gammas = [], [], [], [], []
    arrive_index = None

    for step in range(max_steps):
        p = np.array([x, y])
        remaining = (1.0 - gamma) * path.length
        to_goal = float(np.linalg.norm(p - goal))
        capturing = gamma >= 1.0 - 1e-9 or remaining <= gains.capture_m

        if capturing:
            # regulate straight to the station
            psi_tan = station_heading
            psi_des = (math.atan2(goal[1] - y, goal[0] - x)
                       if to_goal > 1e-6 else station_heading)
            brake = math.sqrt(max(2.0 * limits.a_max * to_goal, 0.0))
            kappa = 0.0
        else:
            p_ref, psi_tan = path.sample(gamma)
            d = p - p_ref
            # control error: against the reference at the TRACKED gamma
            e_ctrl = -math.sin(psi_tan) * d[0] + math.cos(psi_tan) * d[1]
            psi_des = psi_tan - math.atan2(gains.k_e * e_ctrl, max(v, 0.5))
            brake = math.sqrt(max(2.0 * limits.a_max * remaining, 0.0))
            kappa = path.curvature(gamma)

        # reported error: against the nearest point on the path, in both phases
        e = cross_track_to_path(ref_pts, ref_tan, p)

        v_want = float(np.clip(min(limits.v_cruise, brake),
                               limits.v_min, limits.v_max))
        v += float(np.clip(v_want - v, -limits.a_max * limits.dt,
                           limits.a_max * limits.dt))

        # feedforward + feedback. The reference heading ROTATES on every arc, so
        # a pure proportional loop is asked to track a ramp and necessarily lags
        # — which shows up as the vehicle riding wide through every turn. The
        # v*kappa term supplies the turn rate the arc requires; the k_psi term
        # is then only correcting error, which is what it is good at.
        omega = float(np.clip(v * kappa + gains.k_psi * wrap(psi_des - psi),
                              -limits.omega_max, limits.omega_max))

        pos.append([x, y])
        heads.append(psi)
        speeds.append(v)
        errs.append(e)
        gammas.append(gamma)

        if to_goal <= gains.arrive_radius:
            arrive_index = step
            break

        # Exact unicycle step under constant (v, omega) over the interval — the
        # arc, not its chord. Forward Euler here spirals off a curve at
        # dt = 0.1 s and 1 rad/s, and that shows up as a metre of "cross-track
        # error" that is an artefact of the integrator rather than anything the
        # guidance law did. Constant (v, omega) between control updates is also
        # what the vehicle physically does, so this is the more faithful model.
        if abs(omega) > 1e-9:
            psi_next = psi + omega * limits.dt
            x += (v / omega) * (math.sin(psi_next) - math.sin(psi))
            y -= (v / omega) * (math.cos(psi_next) - math.cos(psi))
            psi = wrap(psi_next)
        else:
            x += v * math.cos(psi) * limits.dt
            y += v * math.sin(psi) * limits.dt
        if path.length > 1e-9 and not capturing:
            gamma = min(1.0, gamma + v * math.cos(wrap(psi - psi_tan))
                        / path.length * limits.dt)

    return ApproachResult(
        pos=np.array(pos), heading=np.array(heads), speed=np.array(speeds),
        cross_track=np.array(errs), gamma=np.array(gammas),
        arrived=arrive_index is not None, arrive_index=arrive_index, path=path,
    )


# --------------------------------------------------------------------------
# 3.3 The dispatch, end to end
# --------------------------------------------------------------------------
@dataclass
class Dispatch:
    """One agent, one go-signal, one station."""

    agent: int
    station: np.ndarray
    approach_heading: float
    path: DubinsPath
    result: ApproachResult
    task: str


def min_turn_radius(limits: GuidanceLimits) -> float:
    """The tightest turn the vehicle can actually hold at cruise: v / omega_max.

    Planning a Dubins path tighter than this produces a reference the aircraft
    cannot track — it flies the outside of every turn and the follower reports a
    cross-track error it can never null, which looks like a badly tuned gain and
    is in fact a kinematically impossible request. Caught at construction here
    rather than diagnosed later.
    """
    return limits.v_cruise / max(limits.omega_max, 1e-9)


def dispatch_on_signal(agent: int, pose, x_tac, task: str = "inspection",
                       turn_radius: float | None = None,
                       limits: GuidanceLimits | None = None,
                       gains: GuidanceGains | None = None) -> Dispatch:
    """Edge-triggered: the agent that receives the go-signal breaks off now.

    Nothing in this call refers to any other agent's state or timing. A second
    signal to a second agent produces a second, independent Dispatch.

    `turn_radius=None` sizes the path to the vehicle, with 15% margin.
    """
    limits = limits or GuidanceLimits()
    r_min = min_turn_radius(limits)
    if turn_radius is None:
        turn_radius = 1.15 * r_min
    elif turn_radius < r_min - 1e-9:
        raise ValueError(
            f"turn_radius {turn_radius:.2f} m is tighter than the vehicle can "
            f"hold at cruise (v_cruise/omega_max = {r_min:.2f} m)")

    station, heading = standoff_station(x_tac, task)
    path = dubins(pose, (station[0], station[1], heading), turn_radius)
    res = follow(path, pose, heading, limits, gains)
    return Dispatch(agent=agent, station=station, approach_heading=heading,
                    path=path, result=res, task=task)


def setpoint_stream(result: ApproachResult, decimate: int = 1) -> np.ndarray:
    """(H, 1, 2) — the approach as a supervisor-checkable setpoint stream."""
    return result.pos[::decimate][:, None, :]


def validate_approach(result: ApproachResult, cfg: SupervisorConfig,
                      limits: StreamLimits, static=None):
    """Run the emitted approach through the same gate everything else uses."""
    return check_stream(setpoint_stream(result), cfg, limits, static=static)


# --------------------------------------------------------------------------
# Demo
# --------------------------------------------------------------------------
def main() -> None:
    x_tac = np.array([16.0, 3.0])
    # on the loiter ring, nose along the tangent and pointing AWAY from the
    # target: the honest case, where the vehicle has to turn around first and
    # the curvature bound actually binds
    start = (-2.0, 7.8, math.radians(155.0))

    print(f"\nEvent-triggered dispatch  —  target X_tac = "
          f"{x_tac.tolist()}, agent at {start[:2]}")
    print("=" * 92)
    print(f"{'task':<12}{'station':<22}{'d_s':>6}{'path':>7}{'len':>9}"
          f"{'|e| final':>11}{'arrive':>9}{'t':>8}")
    print("-" * 92)

    lim = GuidanceLimits()
    for task in TASKS:
        d = dispatch_on_signal(0, start, x_tac, task)
        r = d.result
        t_s = (r.arrive_index or len(r.pos)) * lim.dt
        print(f"{task:<12}"
              f"({d.station[0]:6.2f}, {d.station[1]:6.2f})      "
              f"{TASKS[task].standoff_m:>5.1f}{d.path.word:>7}"
              f"{d.path.length:>8.1f}m{abs(r.cross_track[-1]):>10.3f}m"
              f"{('yes' if r.arrived else 'NO'):>9}{t_s:>7.1f}s")
    print("=" * 92)

    d = dispatch_on_signal(0, start, x_tac, "inspection")
    r = d.result
    print(f"inspection run: |e| peak {np.abs(r.cross_track).max():.2f} m -> "
          f"final {abs(r.cross_track[-1]):.4f} m; "
          f"stops {r.final_error:.3f} m from the station, "
          f"{np.linalg.norm(d.station - x_tac):.1f} m from the target itself "
          f"(never homes onto it).")

    from hive.plots import plot_standoff_approach
    print(f"\nfigure -> {plot_standoff_approach(x_tac, start)}")


if __name__ == "__main__":
    main()
