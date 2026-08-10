"""D2.3 — the guarded mission FSM: ACCEPT as the steady state.

    ANCHOR_INIT -> LOITER_MESH -> TASK_INGEST -> AUCTION -> RECONFIG
                                                               |
                                                     (external go-signal)
                                                               v
                                                           DISPATCH -> APPROACH
                                                               |
                                                          STANDOFF_TASK -> REJOIN

THE DESIGN COMMITMENT. The supervisor is never routed around, and it is also
never *argued with*. Each transition compiles a full setpoint stream over
horizon H, checks that stream against the same invariants the Rust supervisor
enforces, and only then commits. If the guard fails, the FSM does not transition
— it stays where it is and re-plans. So a REJECT arriving from the real
supervisor is not an expected event to be handled gracefully; it is a bug
report, and `MissionLog.rejections` existing at all is the alarm.

That is why RECONFIG is rate-limited rather than solved in one jump. The
one-jump answer is a Lloyd step straight to the new centroids, which is correct
in the limit and violates v_max on the very first tick. The FSM instead walks
toward the Lloyd target at a bounded slew and re-checks clearance every tick, so
"the plan is safe" is a property of every intermediate state, not just the
endpoint. A plan that is only valid at its destination is not a plan, it is a
wish.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

import numpy as np

from hive.cbba import (
    BidWeights,
    SwarmState,
    bid_scores,
    comms_adjacency,
    consensus_elect,
    partition_control_graph,
)
from hive.supervisor_gate import (
    Assignment,
    EstimateSnapshot,
    Plan,
    StreamLimits,
    SupervisorConfig,
    check_stream,
    clearance,
    validate,
)


class State(enum.Enum):
    ANCHOR_INIT = "ANCHOR_INIT"
    LOITER_MESH = "LOITER_MESH"
    TASK_INGEST = "TASK_INGEST"
    AUCTION = "AUCTION"
    RECONFIG = "RECONFIG"
    DISPATCH = "DISPATCH"


@dataclass
class MissionConfig:
    n_slots: int = 2                  # coalition size for the task
    r_comm: float = 11.0
    horizon: int = 40                 # ticks of forward simulation per guard
    limits: StreamLimits = field(default_factory=StreamLimits)
    supervisor: SupervisorConfig = field(default_factory=SupervisorConfig)
    weights: BidWeights = field(default_factory=BidWeights)
    lloyd_iters: int = 8
    tick_ms: int = 100


@dataclass
class MissionLog:
    """Everything the run wants to be able to prove afterwards."""

    transitions: list[tuple[str, str, str]] = field(default_factory=list)
    decisions: list[dict] = field(default_factory=list)
    rejections: list[dict] = field(default_factory=list)
    # (plan, estimate) as submitted, so a test can replay them through the real
    # Rust supervisor rather than trusting the Python mirror end-to-end
    submitted: list[tuple] = field(default_factory=list)
    guard_failures: list[dict] = field(default_factory=list)
    min_clearance: float = float("inf")
    max_step: float = 0.0
    ticks: int = 0

    def record(self, frm: State, to: State, why: str):
        self.transitions.append((frm.value, to.value, why))


# --------------------------------------------------------------------------
# Loiter geometry
# --------------------------------------------------------------------------
def lloyd_targets(pos, region, iters: int = 8) -> np.ndarray:
    """Lloyd's algorithm on the coverage samples: each agent moves to the
    centroid of its Voronoi cell, repeat. This is the *destination* of a
    RECONFIG, never the trajectory — see `slew_toward`."""
    p = np.array(pos, dtype=float)
    r = np.asarray(region, dtype=float)
    if len(p) == 0:
        return p
    for _ in range(iters):
        d2 = ((r[:, None, :] - p[None, :, :]) ** 2).sum(axis=2)
        owner = d2.argmin(axis=1)
        for i in range(len(p)):
            cell = r[owner == i]
            if len(cell):
                p[i] = cell.mean(axis=0)
    return p


def slew_toward(start, goal, limits: StreamLimits, horizon: int,
                static=None, d_clear_pad: float = 1.06) -> np.ndarray:
    """Compile a rate-limited stream from `start` to `goal`.

    Returns (H, K, 2). Every tick moves each agent at most `v_max * dt` toward
    its goal, so the velocity invariant holds by construction rather than by
    checking. Clearance cannot be guaranteed by construction — two agents can be
    ordered to swap sides — so it is *checked*, and a nudge is applied when a
    tick would close below the floor.

    `static` is the rest of the swarm, holding position. Moving agents are
    pushed away from them; they never move in response, because they are not
    part of this stream and commanding them here would be exactly the kind of
    silent cross-block coupling the adjacency cut exists to prevent.

    `d_clear_pad` aims the nudge slightly above the floor. Nudging to exactly
    the floor leaves the stream sitting on the boundary, where the next tick's
    rounding decides whether the guard passes — the same reason the supervisor
    treats a waypoint *on* the geofence as outside it.

    The nudge is deliberately simple (push apart along the separation axis, then
    re-clamp the step). Anything cleverer belongs in a real motion planner; what
    matters here is that the compiler either emits a stream that passes every
    gate or admits it could not.
    """
    s = np.array(start, dtype=float)
    g = np.array(goal, dtype=float)
    held = (np.zeros((0, 2)) if static is None
            else np.asarray(static, dtype=float).reshape(-1, 2))
    step_max = limits.v_max * limits.dt
    floor = limits.d_clear * d_clear_pad

    out = np.empty((horizon, len(s), 2))
    cur = s.copy()
    for t in range(horizon):
        d = g - cur
        n = np.linalg.norm(d, axis=1, keepdims=True)
        scale = np.minimum(1.0, step_max / np.maximum(n, 1e-12))
        nxt = cur + d * scale

        # separation nudge, then re-clamp so the velocity gate still holds
        for _ in range(8):
            if clearance(np.vstack([nxt, held])) >= floor:
                break
            for i in range(len(nxt)):
                for j in range(i + 1, len(nxt)):       # mover vs mover: both give
                    diff = nxt[i] - nxt[j]
                    dist = float(np.linalg.norm(diff))
                    if dist < floor:
                        push = (diff / max(dist, 1e-9)) * (floor - dist) * 0.55
                        nxt[i] += push
                        nxt[j] -= push
                for hpt in held:                       # mover vs holder: mover gives
                    diff = nxt[i] - hpt
                    dist = float(np.linalg.norm(diff))
                    if dist < floor:
                        nxt[i] += (diff / max(dist, 1e-9)) * (floor - dist) * 1.05
            move = nxt - cur
            mn = np.linalg.norm(move, axis=1, keepdims=True)
            nxt = cur + move * np.minimum(1.0, step_max / np.maximum(mn, 1e-12))

        cur = nxt
        out[t] = cur
    return out


# --------------------------------------------------------------------------
# The FSM
# --------------------------------------------------------------------------
class MissionFSM:
    """One mission, driven by explicit events, with every transition guarded."""

    def __init__(self, swarm: SwarmState, cfg: MissionConfig | None = None,
                 now_ms: int = 1_000_000):
        self.swarm = swarm
        self.cfg = cfg or MissionConfig()
        self.state = State.ANCHOR_INIT
        self.log = MissionLog()
        self.now_ms = now_ms

        self.pos = np.array(swarm.pos, dtype=float)
        self.target = None
        self.active: list[int] = []
        self.loiter: list[int] = list(range(len(self.pos)))
        self.partition = None
        self.plan_seq = 0

    # -- plumbing ---------------------------------------------------------
    def _estimate(self) -> EstimateSnapshot:
        cov = (self.swarm.cov_trace if self.swarm.cov_trace is not None
               else np.full(len(self.pos), 0.001))
        return EstimateSnapshot(frame_index=self.log.ticks,
                                stamp_unix_ms=self.now_ms,
                                pos=[tuple(p) for p in self.pos],
                                cov_trace=[float(c) for c in cov])

    def _submit(self, waypoints, vehicles) -> bool:
        """Hand a single tick's setpoints to the supervisor and record what it
        said. Every accepted tick is one more piece of evidence that the guard
        upstream was telling the truth.

        Every TRUSTED agent is assigned, every tick — movers get their new
        setpoint, the rest are commanded to hold where they are. Assigning the
        holders is not padding: the supervisor's spacing gate only compares
        waypoints *within one plan*, so a plan containing only the movers would
        leave mover-vs-holder separation unchecked by the one component allowed
        to say no.

        Untrusted agents are the exception, and they are excluded for the
        opposite reason. The supervisor refuses to command a vehicle whose
        covariance is too high, and it is right to: you must not steer a drone
        you cannot localise. So they are left out of the plan entirely — but
        they are still passed to the horizon guard as static obstacles, because
        not being commandable does not make an aircraft stop occupying space.
        (Their clearance margin should really be inflated by their position
        uncertainty; here it is not, which is a known simplification rather than
        an oversight.)
        """
        self.plan_seq += 1
        moving = {int(v): (float(w[0]), float(w[1]))
                  for v, w in zip(vehicles, waypoints)}
        assignments = [
            Assignment(i, moving.get(i, (float(self.pos[i][0]),
                                         float(self.pos[i][1]))))
            for i in range(len(self.pos))
            if i in moving or self._trusted(i)
        ]
        plan = Plan(
            plan_id=f"m{self.plan_seq:04d}",
            issued_unix_ms=self.now_ms,
            assignments=assignments,
        )
        est = self._estimate()
        d = validate(plan, est, self.cfg.supervisor, self.now_ms)
        self.log.submitted.append((plan, est, self.now_ms))
        self.log.decisions.append({"plan_id": plan.plan_id,
                                   "accepted": d.accepted,
                                   "violations": d.kinds()})
        if not d.accepted:
            self.log.rejections.append({"plan_id": plan.plan_id,
                                        "state": self.state.value,
                                        "violations": d.violations})
        return d.accepted

    def _fly(self, stream, vehicles) -> bool:
        """Execute a pre-validated stream, one tick at a time."""
        ok = True
        for t in range(len(stream)):
            self.now_ms += self.cfg.tick_ms
            self.log.ticks += 1
            ok &= self._submit(stream[t], vehicles)
            for idx, v in enumerate(vehicles):
                self.pos[v] = stream[t, idx]
            self.log.min_clearance = min(self.log.min_clearance,
                                         clearance(self.pos))
        return ok

    def _trusted(self, i: int) -> bool:
        """Would the supervisor's covariance gate let this vehicle be commanded?

        Asked before commanding rather than after being rejected — the FSM's
        whole promise is that a REJECT never happens.
        """
        if self.swarm.cov_trace is None:
            return True
        return bool(self.swarm.cov_trace[i] <= self.cfg.supervisor.max_cov_trace)

    def _held(self, vehicles) -> np.ndarray:
        """Positions of every agent NOT in this stream — trusted or not. They
        hold; the movers must clear them."""
        moving = set(int(v) for v in vehicles)
        idx = [i for i in range(len(self.pos)) if i not in moving]
        return self.pos[idx] if idx else np.zeros((0, 2))

    def _guard(self, stream, vehicles, name: str) -> bool:
        """The horizon check. Nothing moves until this returns True."""
        v = check_stream(stream, self.cfg.supervisor, self.cfg.limits, vehicles,
                         static=self._held(vehicles))
        self.log.min_clearance = min(self.log.min_clearance, v.min_clearance)
        self.log.max_step = max(self.log.max_step, v.max_step)
        if not v.ok:
            self.log.guard_failures.append({"guard": name,
                                            "kinds": v.kinds(),
                                            "n": len(v.violations)})
        return v.ok

    # -- transitions ------------------------------------------------------
    def anchor_init(self) -> bool:
        """The frame exists and every agent is inside the fence. Nothing moves."""
        assert self.state is State.ANCHOR_INIT
        inside = check_stream(self.pos[None, :, :], self.cfg.supervisor,
                              self.cfg.limits)
        if not inside.ok:
            self.log.guard_failures.append({"guard": "ANCHOR_INIT",
                                            "kinds": inside.kinds(),
                                            "n": len(inside.violations)})
            return False
        self.log.record(self.state, State.LOITER_MESH, "frame surveyed, mesh inside fence")
        self.state = State.LOITER_MESH
        return True

    def ingest_task(self, x_tac) -> bool:
        """Target arrives in TacFrame (already projected by frames.py).

        Zero setpoint change, so no gate can fire — the plan's own note. The
        guard is still run rather than skipped, because "obviously safe" is how
        unguarded transitions get written.
        """
        assert self.state is State.LOITER_MESH
        self.target = np.asarray(x_tac, dtype=float)
        stream = np.repeat(self.pos[None, :, :], 2, axis=0)
        if not self._guard(stream, list(range(len(self.pos))), "TASK_INGEST"):
            return False
        self.log.record(self.state, State.TASK_INGEST, "target projected into TacFrame")
        self.state = State.TASK_INGEST
        return True

    def run_auction(self) -> bool:
        """Elect the coalition. Still no motion."""
        assert self.state is State.TASK_INGEST
        adj = comms_adjacency(self.pos, self.cfg.r_comm)
        scores = bid_scores(
            SwarmState(self.pos, self.swarm.battery, self.swarm.sensor,
                       self.swarm.region, self.swarm.cov_trace,
                       self.swarm.max_cov_trace),
            self.target, self.cfg.weights)

        res = consensus_elect(scores, adj, self.cfg.n_slots)
        if not res.agreed():
            self.log.guard_failures.append({"guard": "AUCTION",
                                            "kinds": ["ConsensusNotAgreed"],
                                            "n": 1})
            return False
        if any(not np.isfinite(scores[a]) for a in res.winners):
            self.log.guard_failures.append({"guard": "AUCTION",
                                            "kinds": ["UntrustedWinner"], "n": 1})
            return False

        self.active = sorted(res.winners)
        self.partition = partition_control_graph(adj, self.active)
        self.loiter = self.partition.loiter
        self.log.record(self.state, State.AUCTION,
                        f"coalition {self.active} elected in {res.rounds} rounds")
        self.state = State.AUCTION
        return True

    def reconfigure(self) -> bool:
        """Close the coverage hole the coalition is about to leave behind.

        The residual set re-solves its loiter over M−N agents. Rate-limited: the
        stream walks toward the Lloyd target instead of jumping to it, and every
        intermediate tick is checked, because that is where a jump would break.
        """
        assert self.state is State.AUCTION

        # only agents the supervisor would accept a command for can be moved;
        # the rest stay put and are avoided
        movers = [i for i in self.loiter if self._trusted(i)]
        if not movers:
            self.log.record(self.state, State.RECONFIG,
                            "no commandable residual agents to re-solve")
            self.state = State.RECONFIG
            return True

        start = self.pos[movers]
        goal = lloyd_targets(start, self.swarm.region, self.cfg.lloyd_iters)
        stream = slew_toward(start, goal, self.cfg.limits, self.cfg.horizon,
                             static=self._held(movers))

        if not self._guard(stream, movers, "RECONFIG"):
            return False
        self._fly(stream, movers)
        skipped = len(self.loiter) - len(movers)
        note = f"loiter re-solved over {len(movers)} agents"
        if skipped:
            note += f" ({skipped} held: estimator does not trust them)"
        self.log.record(self.state, State.RECONFIG, note)
        self.state = State.RECONFIG
        return True

    def dispatch(self, stations) -> bool:
        """Commit the coalition to its standoff stations.

        The full approach stream is validated over the whole horizon BEFORE any
        motion, which is the plan's `RECONFIG -> DISPATCH` guard verbatim.
        """
        assert self.state is State.RECONFIG
        start = self.pos[self.active]
        stream = slew_toward(start, np.asarray(stations, dtype=float),
                             self.cfg.limits, self.cfg.horizon,
                             static=self._held(self.active))

        if not self._guard(stream, self.active, "DISPATCH"):
            return False
        self._fly(stream, self.active)
        self.log.record(self.state, State.DISPATCH, "approach stream pre-validated")
        self.state = State.DISPATCH
        return True

    # -- the whole sequence ----------------------------------------------
    def run(self, x_tac, stations) -> MissionLog:
        steps = [
            ("ANCHOR_INIT", lambda: self.anchor_init()),
            ("TASK_INGEST", lambda: self.ingest_task(x_tac)),
            ("AUCTION", lambda: self.run_auction()),
            ("RECONFIG", lambda: self.reconfigure()),
            ("DISPATCH", lambda: self.dispatch(stations)),
        ]
        for name, step in steps:
            if not step():
                self.log.guard_failures.append({"guard": name,
                                                "kinds": ["HALTED"], "n": 1})
                break
        return self.log


# --------------------------------------------------------------------------
# Demo / artifact
# --------------------------------------------------------------------------
def default_scenario(m: int = 12, radius: float = 8.0, seed: int = 3):
    """A 12-agent loiter mesh, a fence with room to manoeuvre, a target outside
    the ring. Geofence and spacing are in metres here, not the Brain's
    normalised units, so the supervisor config is scaled to match."""
    rng = np.random.default_rng(seed)
    a = 2 * np.pi * np.arange(m) / m
    pos = np.stack([radius * np.cos(a), radius * np.sin(a)], axis=1)

    ang = np.linspace(0, 2 * np.pi, 180, endpoint=False)
    rad = radius * np.sqrt(np.linspace(0.55, 1.0, 5))
    aa, rr = np.meshgrid(ang, rad)
    region = np.stack([(rr * np.cos(aa)).ravel(), (rr * np.sin(aa)).ravel()],
                      axis=1)

    swarm = SwarmState(pos=pos, battery=rng.uniform(0.65, 1.0, m),
                       sensor=rng.uniform(0.4, 1.0, m), region=region,
                       cov_trace=np.full(m, 0.001))

    fence = [(-22.0, -22.0), (22.0, -22.0), (22.0, 22.0), (-22.0, 22.0)]
    cfg = MissionConfig(
        n_slots=2, r_comm=11.0, horizon=40,
        limits=StreamLimits(v_max=0.9, d_clear=1.2, dt=1.0),
        supervisor=SupervisorConfig(geofence=fence, min_spacing_m=1.2,
                                    max_cov_trace=0.004,
                                    max_plan_age_ms=5_000,
                                    max_estimate_age_ms=1_000),
    )
    return swarm, cfg


def main() -> None:
    swarm, cfg = default_scenario()
    target = np.array([16.0, 3.0])
    stations = np.array([[13.2, 1.0], [13.2, 5.4]])

    fsm = MissionFSM(swarm, cfg)
    log = fsm.run(target, stations)

    print(f"\nM->N handover  —  {len(swarm.pos)} agents, "
          f"{cfg.n_slots}-slot coalition")
    print("=" * 84)
    for frm, to, why in log.transitions:
        print(f"  {frm:<14} -> {to:<14}  {why}")
    print("-" * 84)
    accepted = sum(1 for d in log.decisions if d["accepted"])
    print(f"  final state          {fsm.state.value}")
    print(f"  coalition            {fsm.active}   loiter {fsm.loiter}")
    print(f"  setpoint ticks       {log.ticks}")
    print(f"  supervisor ACCEPT    {accepted}/{len(log.decisions)}")
    print(f"  supervisor REJECT    {len(log.rejections)}")
    print(f"  guard failures       {len(log.guard_failures)}")
    print(f"  min clearance        {log.min_clearance:.3f} m  "
          f"(floor {cfg.limits.d_clear} m)")
    print(f"  max per-tick step    {log.max_step:.3f} m  "
          f"(limit {cfg.limits.v_max * cfg.limits.dt} m)")
    print("=" * 84)
    if log.rejections:
        print("REJECTIONS PRESENT — the guard let something through. "
              "That is a bug, not an event.")
        for r in log.rejections[:5]:
            print(f"  {r['state']}: {r['violations']}")
    else:
        print("Zero rejections. ACCEPT was the steady state for every one of "
              f"the {len(log.decisions)} plans submitted — which is the claim "
              "Domain 2 had to earn.")


if __name__ == "__main__":
    main()
