"""D4.5 — modes to Plan to gate to setpoints, and the loop that runs them.

This is where Domains 2 and 3 meet the live vehicles. It emits the existing
`Plan` (`plan_id`, `issued_unix_ms`, `min_spacing_m`, `assignments`), hands it to
the Rust supervisor, and transmits `SET_POSITION_TARGET_LOCAL_NED` only for a
plan the gate accepted.

    hold        fixed formation (diamond / square) about the anchor centroid
    translate   move the formation centroid, staying inside the anchor footprint
    dispatch    Domain 3's standoff approach for one elected agent

WHO IS ALLOWED TO EMIT. The Rust gate is invoked as a subprocess per plan and
decides; this process transmits. §4.7 defines it that way ("supervisor a
callable Rust gate ... setpoints back over MAVLink"), and the property that has
to survive is not which process owns the socket — it is that no setpoint ever
leaves without an ACCEPT. So that is enforced structurally rather than by
convention: `emit()` takes a `Decision`, and refuses to transmit if the decision
is a reject or if its `plan_id` does not match the plan in hand. Passing the
wrong decision object is a `GateViolation`, not a silent send.

Using the binary's own `--emit` would have been the alternative, and it does not
work here for a concrete reason: it transmits over UDP to a port the vehicle
must be listening on, and ArduPilot SITL has no `udpin` serial device — only
`tcp`, `tcpclient` and `udpclient`, none of which accept an unsolicited datagram
from an ephemeral port. Reaching it would need a relay process per vehicle. The
gate is called instead, which is what the restructured plan asks for anyway.

TWO GATES, AND THE FIRST ONE IS THE POINT. Every mode compiles a full setpoint
stream and runs it through `hive.supervisor_gate.check_stream` over the whole
horizon BEFORE anything is submitted. The Rust gate is then expected to say yes
every single time, and `rejections` existing at all is a bug report rather than
an event — the same commitment `hive/mission_fsm.py` makes offline, now against
an autopilot that will actually move.

START ORDER IS LOAD-BEARING (the plan's §4.6 and its operational lessons):

    SITL up -> truth flowing -> estimator converges AT REST -> params + origin
    -> arm -> takeoff -> command

Streaming setpoints at a vehicle that never armed prints a cheerful log and
silently invalidates every "after-flight" number in it. `require_armed` makes
that a refusal instead of a misleading success.
"""

from __future__ import annotations

import itertools
import json
import math
import pathlib
import subprocess
import time
from dataclasses import dataclass, field

import numpy as np

from hive import supervisor_io
from hive.cbba import (BidWeights, ElectionResult, SwarmState, bid_scores,
                       comms_adjacency, consensus_elect)
from hive.standoff import (TASKS, GuidanceLimits, dispatch_on_signal, dubins,
                           stations_for, walk_waypoints)
from hive.supervisor_gate import (
    Assignment,
    Decision,
    EstimateSnapshot,
    Plan,
    StreamLimits,
    SupervisorConfig,
    check_stream,
)

REPO = pathlib.Path(__file__).resolve().parents[2]
SUPERVISOR_BIN = REPO / "brain/rust/target/release/swarm-supervisor"
SUPERVISOR_CFG = pathlib.Path(__file__).resolve().parent / "config/supervisor.json"

# ArduCopter custom flight modes (MAV_CMD_DO_SET_MODE, custom_mode field).
COPTER_MODE_STABILIZE = 0
COPTER_MODE_GUIDED = 4
COPTER_MODE_LAND = 9

# Position-only type mask: ignore velocity, acceleration, yaw and yaw rate.
TYPE_MASK_POSITION_ONLY = 0b110111111000

# Bid weights for the LIVE auction. D2.1's defaults are tuned for the offline
# scenario's scale and go nearly flat at this one: `w_d / (1 + dist)` spans only
# 0.046 -> 0.030 across targets 20.6 to 32.0 m away, while battery and sensor
# (both 1.0 for every vehicle here) add a constant 0.65. Total bid spread is 2%
# over a 55% spread in distance, so the position-sensitive coverage term decides
# the winner and the election becomes a coin flip: 5 cm of estimator noise was
# measured electing 54 DISTINCT coalitions out of 56 possible at n=8, most of
# which then have to cross the loiter mesh and are refused for clearance.
#
# Raising w_d restores what `BidWeights` says it means ("distance dominates by
# design") at this scale, and dropping w_r stops a near-degenerate coverage
# penalty from casting the deciding vote. Measured over 200 runs at 0.3 m noise:
#
#     n=8  slots=3    167/200 refused  ->   12/200
#     n=12 slots=4    193/200 refused  ->    5/200
#     n=6  slots=3    128/200 refused  ->   34/200
#
# `hive/cbba.py` is deliberately NOT changed: its defaults back the published
# Domain 2 result, and weights are a parameter precisely so a different scenario
# can carry different ones.
LIVE_BID_WEIGHTS = BidWeights(w_d=25.0, w_r=0.05)


class GateViolation(RuntimeError):
    """Raised when something tries to emit without a matching ACCEPT."""


# --------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------
class SupervisorGate:
    """The real Rust binary, called once per plan.

    One-shot by design: read a plan, validate, exit. A gate with memory is a
    gate that can be argued into a decision, so every plan is judged from
    scratch. The cost is process spawn — measured at 3-6 ms on the M1 — which is
    why the plan rate is deliberately lower than the estimator rate. ArduPilot
    holds the last position target indefinitely; it does not need 20 Hz of them.
    """

    def __init__(self, binary: pathlib.Path | None = None,
                 config: pathlib.Path | None = None,
                 shared_dir: pathlib.Path | None = None) -> None:
        self.binary = pathlib.Path(binary or SUPERVISOR_BIN)
        self.config = pathlib.Path(config or SUPERVISOR_CFG)
        self.shared = pathlib.Path(shared_dir or supervisor_io.HIVE_DIR)
        self.shared.mkdir(parents=True, exist_ok=True)
        self.accepted = 0
        self.rejected = 0
        self.call_ms: list[float] = []

    @property
    def available(self) -> bool:
        return self.binary.exists()

    def judge(self, plan: Plan, est: EstimateSnapshot,
              now_ms: int | None = None) -> Decision:
        """Run the gate. Returns the Rust binary's own Decision.

        The plan and estimate go through `hive.supervisor_io`, so they get the
        ENU -> (North, East) swap and the atomic write, exactly as the offline
        stack does. Reimplementing the write here would be a second place for
        the frame swap to be wrong.
        """
        if not self.available:
            raise FileNotFoundError(
                f"supervisor binary not built at {self.binary} — "
                f"cargo build --release --manifest-path brain/rust/Cargo.toml")

        supervisor_io.publish_estimate(est)
        supervisor_io.publish_plan(plan)

        cmd = [str(self.binary),
               "--plan", str(supervisor_io.PLAN_PATH),
               "--estimate", str(supervisor_io.ESTIMATE_PATH),
               "--config", str(self.config)]
        if now_ms is not None:
            cmd += ["--now-ms", str(int(now_ms))]

        t0 = time.perf_counter()
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        self.call_ms.append((time.perf_counter() - t0) * 1e3)

        text = out.stdout
        try:
            payload = json.loads(text[text.index("{"):text.rindex("}") + 1])
        except (ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"supervisor produced no parsable decision "
                f"(rc={out.returncode}): {out.stderr.strip()[:400]}") from exc

        decision = Decision(plan_id=payload["plan_id"],
                            accepted=bool(payload["accepted"]),
                            violations=payload.get("violations", []))
        if decision.accepted:
            self.accepted += 1
        else:
            self.rejected += 1
        return decision


# --------------------------------------------------------------------------
# Vehicle command surface
# --------------------------------------------------------------------------
class VehicleCommander:
    """Mode, arm, takeoff, land and setpoints for one vehicle.

    TRANSMITS on the connection the ground-truth bridge owns, but never
    RECEIVES on it. Reception goes through the bridge's mailboxes, because
    pymavlink's `recv_match(type=...)` discards non-matching messages rather
    than requeueing them — two readers on one socket silently eat each other's
    traffic. See `GroundTruthBridge.pump`.
    """

    def __init__(self, index: int, bridge) -> None:
        self.index = index
        self.bridge = bridge
        self.link = bridge.link(index)
        self.last_status: str = ""

    # -- telemetry (from the bridge's mailboxes, never from the socket) ----
    @property
    def armed(self) -> bool:
        return self.bridge.armed[self.index]

    @property
    def mode(self) -> int | None:
        return self.bridge.mode[self.index]

    def drain_status(self) -> list[str]:
        out = self.bridge.drain_status(self.index)
        if out:
            self.last_status = out[-1]
        return out

    def heartbeat_state(self):
        return self.armed, self.mode

    # -- commands ---------------------------------------------------------
    def _command(self, command: int, *params) -> None:
        args = list(params) + [0.0] * (7 - len(params))
        self.link.mav.command_long_send(
            self.link.target_system, self.link.target_component,
            command, 0, *[float(a) for a in args])

    def set_mode(self, custom_mode: int) -> None:
        from pymavlink import mavutil
        self._command(mavutil.mavlink.MAV_CMD_DO_SET_MODE,
                      mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                      custom_mode)

    def arm(self, force: bool = False) -> None:
        from pymavlink import mavutil
        self._command(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                      1, 21196 if force else 0)

    def disarm(self, force: bool = False) -> None:
        from pymavlink import mavutil
        self._command(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                      0, 21196 if force else 0)

    def takeoff(self, alt_m: float) -> None:
        from pymavlink import mavutil
        self._command(mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
                      0, 0, 0, 0, 0, 0, alt_m)

    def land(self) -> None:
        self.set_mode(COPTER_MODE_LAND)

    def setpoint_ned(self, north: float, east: float, down: float) -> None:
        from pymavlink import mavutil
        self.link.mav.set_position_target_local_ned_send(
            0, self.link.target_system, self.link.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED, TYPE_MASK_POSITION_ONLY,
            float(north), float(east), float(down),
            0, 0, 0, 0, 0, 0, 0, 0)

    def wait_armed(self, timeout_s: float = 30.0) -> bool:
        """Only useful if something else is pumping the bridge."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            self.bridge.pump()
            self.drain_status()
            if self.armed:
                return True
            time.sleep(0.1)
        return False


# --------------------------------------------------------------------------
# Formation geometry
# --------------------------------------------------------------------------
def formation(n: int, shape: str = "square", spacing: float = 6.0,
              centre=(0.0, 0.0)) -> np.ndarray:
    """Station positions in TacFrame metres.

    `spacing` is the circumradius, not the edge length, because the supervisor's
    spacing invariant is about pairwise distance and a circumradius makes the
    worst pair obvious: for a square it is `spacing * sqrt(2)` on the diagonal
    and `spacing * sqrt(2)` on the edge, i.e. uniform.
    """
    c = np.asarray(centre, dtype=float)
    if n == 1:
        return c.reshape(1, 2)
    if shape == "line":
        offs = (np.arange(n) - (n - 1) / 2.0) * spacing
        return np.stack([c[0] + offs, np.full(n, c[1])], axis=1)
    a = 2.0 * np.pi * np.arange(n) / n
    if shape == "diamond":
        a = a + np.pi / 4.0
    return np.stack([c[0] + spacing * np.cos(a),
                     c[1] + spacing * np.sin(a)], axis=1)


def assign_stations(starts, stations) -> list[int]:
    """Which station each agent takes, minimising TOTAL transit distance.

    Not a tidiness measure — a collision one. `walk_waypoints` gives every path
    the same rung count, so all movers fly equal FRACTIONS of their own arc in
    lockstep. Two agents whose paths cross therefore reach the crossing point on
    the same rung, and the supervisor refuses the stream for `SpacingTooClose`.
    Assigning by agent index makes that likely as soon as the coalition's
    angular order differs from the perimeter's.

    A minimum-total-distance assignment cannot contain a crossing pair: if two
    straight legs cross, swapping their endpoints strictly shortens the total by
    the triangle inequality, so the optimum has no crossings. Dubins arcs are
    not straight, so this removes the cause rather than proving the absence —
    which is why `dispatch_coalition` still checks the finished ladder.

    Exact by enumeration. A coalition is a handful of agents, and a permutation
    search over that beats adding a solver dependency to a codebase that keeps
    its graph code numpy-only on purpose.
    """
    starts = np.asarray(starts, dtype=float)
    stations = np.asarray(stations, dtype=float)
    k = len(stations)
    cost = np.linalg.norm(starts[:, None, :] - stations[None, :, :], axis=2)
    if k > 8:
        # 9! is 362 880 and climbing; greedy nearest-first instead. No coalition
        # this stack elects is anywhere near here, but a silent 40 s stall while
        # a factorial runs would be a worse failure than a suboptimal pairing.
        order, taken = [], set()
        for i in range(k):
            j = min((j for j in range(k) if j not in taken),
                    key=lambda j: cost[i, j])
            order.append(j)
            taken.add(j)
        return order
    return list(min(itertools.permutations(range(k)),
                    key=lambda p: cost[range(k), p].sum()))


@dataclass
class OrchestratorConfig:
    alt_m: float = 2.5
    shape: str = "square"
    spacing_m: float = 6.0
    horizon: int = 30                  # ticks of forward simulation per guard
    limits: StreamLimits = field(
        default_factory=lambda: StreamLimits(v_max=0.6, d_clear=2.0, dt=1.0))
    arrive_tol_m: float = 0.8
    plan_hz: float = 4.0


@dataclass
class OrchestratorLog:
    plans: int = 0
    accepted: int = 0
    rejections: list[dict] = field(default_factory=list)
    guard_failures: list[dict] = field(default_factory=list)
    emitted_setpoints: int = 0
    modes: list[str] = field(default_factory=list)


class Orchestrator:
    """Compile a mode into a gated, transmitted setpoint stream."""

    def __init__(self, commanders, cfg: OrchestratorConfig | None = None,
                 gate: SupervisorGate | None = None,
                 supervisor_cfg: SupervisorConfig | None = None) -> None:
        self.commanders = list(commanders)
        self.n = len(self.commanders)
        self.cfg = cfg or OrchestratorConfig()
        self.gate = gate or SupervisorGate()
        self.supervisor_cfg = supervisor_cfg or self._load_supervisor_cfg()
        self.log = OrchestratorLog()
        self.plan_seq = 0
        self.require_armed = True

    @staticmethod
    def _load_supervisor_cfg() -> SupervisorConfig:
        """Read the SAME json the Rust gate reads.

        Two configs that drift apart is the failure this avoids: the Python
        pre-check would certify a stream against a geofence the Rust gate has
        never heard of, and the first thing anyone would see is an inexplicable
        REJECT on a plan the planner was certain about.
        """
        raw = json.loads(SUPERVISOR_CFG.read_text())
        return SupervisorConfig(
            geofence=[tuple(p) for p in raw["geofence"]],
            min_spacing_m=float(raw["min_spacing_m"]),
            max_cov_trace=float(raw["max_cov_trace"]),
            max_plan_age_ms=int(raw["max_plan_age_ms"]),
            max_estimate_age_ms=int(raw["max_estimate_age_ms"]),
        )

    # -- stream compilation ------------------------------------------------
    def _slew(self, start, goal) -> np.ndarray:
        """Rate-limited stream from `start` to `goal`, (H, K, 2).

        Reuses `hive.mission_fsm.slew_toward` — the same compiler the offline
        M->N handover used, including its separation nudge. Nothing about being
        live changes what a safe stream is.
        """
        from hive.mission_fsm import slew_toward
        return slew_toward(np.asarray(start, float), np.asarray(goal, float),
                           self.cfg.limits, self.cfg.horizon)

    def guard(self, stream, vehicles, name: str) -> bool:
        v = check_stream(stream, self.supervisor_cfg, self.cfg.limits, vehicles)
        if not v.ok:
            self.log.guard_failures.append(
                {"guard": name, "kinds": v.kinds(), "n": len(v.violations),
                 "min_clearance": v.min_clearance, "max_step": v.max_step})
        return v.ok

    # -- modes -------------------------------------------------------------
    def hold(self, centre=(0.0, 0.0)) -> np.ndarray:
        return formation(self.n, self.cfg.shape, self.cfg.spacing_m, centre)

    def translate(self, current, centre_goal, range_world=None) -> np.ndarray:
        """Move the formation centroid to `centre_goal`.

        S2's honest constraint, checked here rather than discovered in a plot:
        bulk translation of the whole formation is observable ONLY through
        changing anchor ranges. Leave the anchor footprint and the swarm's
        absolute position stops being pinned by anything — the estimate will
        still look confident, which is the exact failure this project exists to
        refuse. If a `range_world` is supplied, a goal outside its footprint is
        rejected before it becomes a plan.
        """
        goal = self.hold(centre_goal)
        if range_world is not None:
            reach = range_world.anchor_reach(goal)
            outside = np.flatnonzero(reach > range_world.cfg.r_anchor)
            if len(outside):
                raise ValueError(
                    f"translate target puts vehicles {outside.tolist()} outside "
                    f"the anchor footprint (nearest anchor "
                    f"{reach.max():.1f} m > r_anchor "
                    f"{range_world.cfg.r_anchor:.1f} m). Bulk translation is "
                    f"unobservable out there — this is S3's flying anchor, "
                    f"not S2.")
        return goal

    def square(self, side: float = 5.0, centre=(0.0, 0.0)) -> list[np.ndarray]:
        """S0's "small commanded square", as a sequence of formation goals.

        The single most informative thing one vehicle can be asked to do. A hold
        proves the estimator does not drift; a square proves it survives MOTION,
        which is the exact axis the old stack failed on — calibrated at rest
        (ratio 0.78), 147x overconfident the moment anything moved. Four legs
        also exercise all four anchors asymmetrically rather than sitting at the
        symmetric centre where errors cancel.
        """
        c = np.asarray(centre, dtype=float)
        h = side / 2.0
        corners = [c + np.array(o) for o in
                   ((+h, +h), (-h, +h), (-h, -h), (+h, -h), (+h, +h))]
        return [np.array([corner + p for p in
                          (formation(self.n, self.cfg.shape,
                                     self.cfg.spacing_m) if self.n > 1
                           else np.zeros((1, 2)))])
                for corner in corners]

    def dispatch(self, agent: int, current, x_tac, task: str = "inspection"):
        """Domain 3's standoff approach for one elected agent.

        The vehicle flies to a station NEAR the coordinate, never onto it. The
        heading is whatever the task geometry asks for; ArduCopter is commanded
        in position only, so the Dubins path is used for its length and its
        curvature bound and the yaw is left to the autopilot.
        """
        p = np.asarray(current, dtype=float)[agent]
        d = dispatch_on_signal(agent, (p[0], p[1], 0.0), x_tac, task,
                               limits=GuidanceLimits(v_max=self.cfg.limits.v_max
                                                     / max(self.cfg.limits.dt, 1e-9)))
        return d.station, d

    def elect(self, current, x_tac, n_slots: int = 2, r_comm: float = 30.0,
              cov_trace=None, weights: BidWeights | None = None
              ) -> ElectionResult:
        """D2.1's auction, run live for the first time.

        `max_cov_trace` is taken from the SAME supervisor config the gate reads
        rather than `SwarmState`'s stricter default. The two thresholds mean the
        same thing — "the estimator does not trust this agent" — and if the
        auction used a tighter bar than the gate it would refuse to elect agents
        the gate would happily have accepted, which reads as a broken auction
        rather than as a deliberately different constant.
        """
        pos = np.asarray(current, dtype=float)
        n = len(pos)
        cov = (np.zeros(n) if cov_trace is None
               else np.asarray(cov_trace, dtype=float))
        state = SwarmState(
            pos=pos,
            battery=np.ones(n),
            sensor=np.ones(n),
            region=self.hold(),                 # the loiter mesh it must cover
            cov_trace=cov,
            max_cov_trace=self.supervisor_cfg.max_cov_trace)
        scores = bid_scores(state, x_tac, weights or LIVE_BID_WEIGHTS)
        return consensus_elect(scores, comms_adjacency(pos, r_comm), n_slots)

    def dispatch_coalition(self, agents, current, x_tac,
                           task: str = "inspection", spread_deg: float = 45.0):
        """D3 for a coalition: one station each, one Dubins path each.

        Returns (goals, stations, paths). `goals` is the (K, N, 2) ladder the
        loop walks: the elected agents step along their own curve while everyone
        else holds their loiter slot, so the guard sees the approach and the
        static mesh in the same stream — which is the case that actually needs
        checking, and the one `-m dispatch` never exercised.
        """
        pos = np.asarray(current, dtype=float)
        agents = list(agents)
        x = np.asarray(x_tac, dtype=float)[:2]
        placed = stations_for(x_tac, task, len(agents), spread_deg=spread_deg)
        limits = GuidanceLimits(
            v_max=self.cfg.limits.v_max / max(self.cfg.limits.dt, 1e-9))
        d_s = TASKS[task].standoff_m if isinstance(task, str) else task.standoff_m

        # An agent already INSIDE the standoff perimeter cannot fly a standoff
        # approach — there is no path out of a circle you start within that
        # never enters it. Refusing here rather than flying it is the same
        # discipline `translate` applies to the anchor footprint: the failure is
        # in the scenario, and flying it anyway would quietly retire D3's one
        # load-bearing claim ("converges to a station, never to the target").
        inside = [int(a) for a in agents
                  if np.linalg.norm(pos[a] - x) < d_s]
        if inside:
            raise ValueError(
                f"vehicles {inside} start inside the {d_s:.0f} m standoff "
                f"perimeter of {x.tolist()} — no approach to a station on that "
                f"perimeter can avoid entering it. Move the target further from "
                f"the loiter mesh, or choose a task with a smaller standoff.")

        # Pair agents to stations before any path is solved — see
        # `assign_stations`. Index order is what made three-agent coalitions
        # cross in mid-transit and stall against the guard.
        placed = [placed[j] for j in
                  assign_stations(pos[agents], [s for s, _ in placed])]

        stations, paths = [], []
        for agent, (station, heading) in zip(agents, placed):
            p = pos[agent]
            d = dispatch_on_signal(agent, (p[0], p[1], 0.0), x_tac, task,
                                   limits=limits)
            # dispatch_on_signal places the agent at the task's single canonical
            # station; for a coalition each agent gets its own perimeter slot,
            # so the path is re-solved to THIS agent's station and heading.
            #
            # The start heading is the bearing to the station, NOT the 0.0 that
            # `dispatch` hardcodes. ArduCopter is commanded in position only and
            # its true yaw is never fed back, so the value is free — but it is
            # not harmless: an arbitrary heading over a hop shorter than 2R
            # forces Dubins into a loop, and that loop can swing through the
            # standoff perimeter. Pointing at the station costs nothing and
            # removes the whole failure mode.
            bearing = math.atan2(station[1] - p[1], station[0] - p[0])
            path = dubins((p[0], p[1], bearing),
                          (float(station[0]), float(station[1]), float(heading)),
                          d.path.radius)
            stations.append(np.asarray(station, dtype=float))
            paths.append(path)

        rungs = walk_waypoints(paths, self.cfg.limits.v_max * self.cfg.limits.dt)

        # Checked on the RUNGS, not on the analytic path: the rungs are what is
        # actually flown, and a curve that respects the perimeter can still be
        # sampled into one that cuts a chord across it.
        breach = np.linalg.norm(rungs - x, axis=2).min()
        if breach < d_s - self.cfg.limits.v_max:
            raise ValueError(
                f"the approach enters the {d_s:.0f} m standoff perimeter "
                f"(closest {breach:.2f} m). Dispatch converges to a station, "
                f"never to the target — this stream would break that.")

        held = self.hold()
        goals = np.repeat(held[None, :, :], len(rungs), axis=0)
        for slot, agent in enumerate(agents):
            goals[:, agent, :] = rungs[:, slot, :]

        # The ladder is checked HERE, on the ground, against every vehicle —
        # movers and holders alike. Without this the failure surfaces as
        # `step_toward` returning False forever: the guard refuses each rung,
        # the leg never advances, and the coalition simply stops in mid-air part
        # way to the perimeter with nothing printed. A stall that looks like a
        # flown mission is the worst outcome available here, so it is converted
        # into a refusal that names the pair and the distance.
        d = np.linalg.norm(goals[:, :, None, :] - goals[:, None, :, :], axis=3)
        d[:, np.arange(self.n), np.arange(self.n)] = np.inf
        if d.min() < self.cfg.limits.d_clear:
            k, i, j = np.unravel_index(d.argmin(), d.shape)
            role = lambda v: "mover" if v in agents else "holder"
            raise ValueError(
                f"the coalition's own streams conflict: v{i} ({role(i)}) and "
                f"v{j} ({role(j)}) come within {d.min():.2f} m at rung {k} of "
                f"{len(goals) - 1}, inside the {self.cfg.limits.d_clear:.1f} m "
                f"clearance floor. Widen --spread, elect fewer agents, or move "
                f"the target further out.")
        return goals, stations, paths

    # -- plan / gate / emit -------------------------------------------------
    def build_plan(self, waypoints, vehicles, now_ms: int) -> Plan:
        self.plan_seq += 1
        return Plan(
            plan_id=f"d4-{self.plan_seq:05d}",
            issued_unix_ms=now_ms,
            assignments=[Assignment(int(v), (float(w[0]), float(w[1])))
                         for v, w in zip(vehicles, waypoints)],
            min_spacing_m=self.supervisor_cfg.min_spacing_m,
        )

    def submit(self, waypoints, vehicles, est: EstimateSnapshot,
               mode: str) -> tuple[Plan, Decision]:
        """Gate one tick's setpoints. Never transmits."""
        now_ms = supervisor_io.now_ms()
        plan = self.build_plan(waypoints, vehicles, now_ms)
        decision = self.gate.judge(plan, est)
        self.log.plans += 1
        self.log.modes.append(mode)
        if decision.accepted:
            self.log.accepted += 1
        else:
            self.log.rejections.append(
                {"plan_id": plan.plan_id, "mode": mode,
                 "violations": decision.violations})
        return plan, decision

    def emit(self, plan: Plan, decision: Decision) -> int:
        """Transmit an ACCEPTED plan. Refuses anything else.

        This is the invariant, and it is a raise rather than a log line because
        the one thing that must never be recoverable-by-accident is a setpoint
        leaving on a plan the gate did not pass.
        """
        if not decision.accepted:
            raise GateViolation(
                f"plan {plan.plan_id} was REJECTED "
                f"({[v.get('kind') for v in decision.violations]}) — "
                f"no setpoint may be emitted")
        if decision.plan_id != plan.plan_id:
            raise GateViolation(
                f"decision is for plan {decision.plan_id!r}, not {plan.plan_id!r} "
                f"— a stale ACCEPT is not an ACCEPT")

        sent = 0
        for a in plan.assignments:
            cmd = self.commanders[a.vehicle]
            if self.require_armed and not cmd.armed:
                # Streaming at a vehicle that never armed prints `done` and
                # silently invalidates every measurement taken afterwards.
                continue
            east, north = a.waypoint_ne          # Plan carries TacFrame ENU
            cmd.setpoint_ned(north=north, east=east, down=-self.cfg.alt_m)
            sent += 1
        self.log.emitted_setpoints += sent
        return sent

    def step_toward(self, current, goal, est: EstimateSnapshot,
                    mode: str, movers=None) -> tuple[np.ndarray, bool]:
        """One planning tick: compile, guard, gate, emit, report the next state.

        Returns (next commanded waypoints, arrived). The stream is compiled over
        the whole horizon and guarded over the whole horizon; only its FIRST
        tick is commanded, because the vehicles fly the physics between ticks
        and the next tick re-plans from where they actually are. Guarding the
        horizon and commanding the head is what makes "pre-validated over
        horizon H" mean something when the world can push back.

        `movers` names the vehicles whose arrival decides progress; the default
        is all of them, which is right when all of them are going somewhere. In
        a standoff dispatch most of the fleet is HOLDING, and a holder is never
        exactly on its slot — it is station-keeping against wind and estimator
        noise. Letting those vehicles vote on arrival couples the coalition's
        progress to the worst station-keeping error in the fleet, and at six
        vehicles on a loaded machine that error exceeds `arrive_tol_m` often
        enough that the ladder simply stops. Every vehicle is still PLANNED,
        GUARDED and COMMANDED every tick — only the arrival test narrows.
        """
        cur = np.asarray(current, dtype=float)
        goal = np.asarray(goal, dtype=float)
        vehicles = list(range(self.n))
        watch = vehicles if movers is None else list(movers)

        if np.all(np.linalg.norm(goal[watch] - cur[watch], axis=1)
                  <= self.cfg.arrive_tol_m):
            return goal, True

        stream = self._slew(cur, goal)
        if not self.guard(stream, vehicles, mode):
            return cur, False

        waypoints = stream[0]
        plan, decision = self.submit(waypoints, vehicles, est, mode)
        if decision.accepted:
            self.emit(plan, decision)
        return waypoints, False

    def report(self) -> dict:
        gate_ms = np.array(self.gate.call_ms) if self.gate.call_ms else np.zeros(1)
        return {
            "plans": self.log.plans,
            "accepted": self.log.accepted,
            "rejected": len(self.log.rejections),
            "guard_failures": len(self.log.guard_failures),
            "setpoints": self.log.emitted_setpoints,
            "gate_call_p50_ms": float(np.percentile(gate_ms, 50)),
            "gate_call_p99_ms": float(np.percentile(gate_ms, 99)),
        }


# --------------------------------------------------------------------------
# The closed loop
# --------------------------------------------------------------------------
@dataclass
class LoopResult:
    ticks: int = 0
    rmse_m: list[float] = field(default_factory=list)
    sigma_m: list[float] = field(default_factory=list)
    ratio: list[float] = field(default_factory=list)
    est_latency_ms: list[float] = field(default_factory=list)
    setpoint_latency_ms: list[float] = field(default_factory=list)
    alt_agl_m: list[float] = field(default_factory=list)
    legs_flown: int = 0
    orchestrator: dict = field(default_factory=dict)
    fanout: dict = field(default_factory=dict)
    truth_sources: dict = field(default_factory=dict)
    coalition: list[int] = field(default_factory=list)
    stations: list = field(default_factory=list)
    station_error_m: list[float] = field(default_factory=list)
    target: list = field(default_factory=list)
    tracks: dict = field(default_factory=dict)      # agent -> flown TRUTH xy
    planned: list = field(default_factory=list)     # the commanded rung ladder
    # None when the run was unsigned. Populated from bridge.signing_report()
    # at the end, so the counters cover the whole flight rather than the
    # handshake — a badsig count that only appears mid-flight is the
    # interesting one.
    signing: dict | None = None
    # None when the run was unguarded. H1/H3 inbound-defence counters: range
    # links refused before the factor graph, and estimates refused before the
    # fan-out. Both should be ZERO on a clean run — a non-zero count is either
    # an attack or the estimator misbehaving, and the whole point of the gate
    # is that it does not need to know which.
    guards: dict | None = None

    # Keyframes the cold start is allowed before it is scored. The estimator
    # multilaterates its first fix from anchor ranges alone and then converges,
    # so keyframe 0 is legitimately several sigma out. Reported separately as
    # `cold_start_peak_ratio` rather than dropped, because "we ignored the first
    # twenty" and "we hid the worst number" have to be distinguishable.
    warmup: int = 20

    def summary(self) -> dict:
        def pct(a, q):
            return float(np.percentile(a, q)) if len(a) else float("nan")

        def steady(a):
            return a[self.warmup:] if len(a) > self.warmup else a

        rmse, sigma, ratio = (steady(self.rmse_m), steady(self.sigma_m),
                              steady(self.ratio))
        cold = self.ratio[:self.warmup]
        nan = float("nan")
        return {
            "ticks": self.ticks,
            "warmup": self.warmup,
            "rmse_mean_m": float(np.mean(rmse)) if rmse else nan,
            "rmse_max_m": float(np.max(rmse)) if rmse else nan,
            "sigma_mean_m": float(np.mean(sigma)) if sigma else nan,
            "ratio_mean": float(np.mean(ratio)) if ratio else nan,
            "ratio_max": float(np.max(ratio)) if ratio else nan,
            "cold_start_peak_ratio": float(np.max(cold)) if cold else nan,
            "est_latency_p50_ms": pct(self.est_latency_ms, 50),
            "est_latency_p99_ms": pct(self.est_latency_ms, 99),
            "setpoint_latency_p50_ms": pct(self.setpoint_latency_ms, 50),
            "setpoint_latency_p99_ms": pct(self.setpoint_latency_ms, 99),
        }


def _tick(bridge, world, estimator, fanout, result, guards=None,
          dt_s: float = 0.1) -> np.ndarray:
    """One estimator tick. Returns ground truth for scoring.

    The order here IS the loop: pump the true positions, generate this tick's
    ranges from them, estimate, publish. `t_range_in` is stamped before
    `measure()` because that is when the radio would have produced the reading;
    everything after it is our latency, and D4.7's number is only honest if the
    clock starts at the measurement rather than at the convenient point.

    `guards` (COMMS_HARDENING_PLAN.md H1/H3) inserts the two inbound checks at
    the only two places they can work: range filtering BEFORE the factor graph,
    because a fused measurement's influence cannot be withdrawn afterwards, and
    estimate plausibility BEFORE the fan-out, because that is the last moment a
    position can be refused before an autopilot starts flying on it. Both sit
    inside the latency window on purpose — security that is not measured against
    the 25 ms budget is security whose cost is unknown.
    """
    bridge.pump()
    truth = bridge.truths()
    t_range_in = time.monotonic()
    frame = world.measure(truth)
    if guards is not None:
        frame = guards.filter_ranges(frame, dt_s)
    estimator.step(frame)

    trusted = None
    if guards is not None:
        stamp_ms = time.time() * 1e3
        trusted = guards.judge(estimator.pos, estimator.cov_trace,
                               stamp_unix_ms=stamp_ms,
                               now_unix_ms=stamp_ms).trusted

    fanout.publish(estimator.pos, estimator.cov_trace, trusted=trusted,
                   origin_mono=t_range_in)
    result.est_latency_ms.append((time.monotonic() - t_range_in) * 1e3)
    return truth


def _score(estimator, truth, result) -> None:
    """Error, sigma and the ratio that is the actual acceptance criterion."""
    from sim.live_estimator import error_over_sigma
    live = np.all(np.isfinite(truth), axis=1)
    if not live.any():
        return
    err = np.linalg.norm(estimator.pos[live] - truth[live], axis=1)
    sigma = estimator.sigma_m()[live]
    result.rmse_m.append(float(np.sqrt((err ** 2).mean())))
    result.sigma_m.append(float(sigma.mean()))
    result.ratio.append(float(error_over_sigma(
        estimator.pos[live], truth[live], sigma).mean()))


def _open_keystore(path, say):
    """Open (or create) the per-vehicle signing keystore for this fleet.

    Deliberately fails rather than inventing a passphrase — see
    `security/keystore.py`. The message names the environment variable because
    the alternative is a run that dies at step 1 with `ValueError` and no hint.
    """
    from security.keystore import Keystore

    try:
        ks = Keystore.open_or_create(path)
    except ValueError as exc:
        raise SystemExit(
            f"signing: {exc}\n"
            f"  export HIVE_KEYSTORE_PASSPHRASE=... before running with --sign, "
            f"and use the SAME one every time: the keys are inside that file "
            f"and the vehicles already hold copies in FRAM.") from None
    say(f"      keystore {ks.path} "
        f"({len(ks.vehicle_ids())} key(s) on file)")
    return ks


def _verify_unsigned_is_refused(bridge, n: int, say) -> None:
    """Confirm the vehicle actually drops an unsigned command. Fail if not.

    Signing that the far end does not enforce is the worst outcome available:
    the counters look healthy, every frame we send is authenticated, and the
    attacker simply omits the signature. So this is a hard stop, not a warning
    — COMMS_HARDENING_PLAN.md §0's third property is that an attack is *seen*,
    and a check that shrugs sees nothing.

    Runs here and nowhere else because `probe_unsigned_rejected` uses
    `recv_match`, which discards non-matching traffic, and it toggles
    `sign_outgoing` on a codec that is not thread-safe. Both are only safe
    before the estimator tick and the fanout thread exist.
    """
    from security.enable_signing import probe_unsigned_rejected

    open_links = []
    for i in range(n):
        verdict = probe_unsigned_rejected(bridge.link(i))
        if verdict["rejected"]:
            say(f"      vehicle {i}: unsigned command refused — link authenticated")
        else:
            say(f"      vehicle {i}: UNSIGNED COMMAND ACCEPTED")
            open_links.append((i, verdict["detail"]))

    if open_links:
        detail = "\n".join(f"  vehicle {i}: {d}" for i, d in open_links)
        raise SystemExit(
            "signing is provisioned but the vehicle still acts on unsigned "
            f"commands:\n{detail}\n"
            "  Relaunch the fleet with sim/run_fleet.sh, which puts the loop on "
            "--serial2. Pass --no-probe-unsigned to run anyway, knowing the "
            "link is authenticated in one direction only.")


def run_loop(n: int = 1, mode: str = "hold", seconds: float = 40.0,
             settle_s: float = 12.0, alt_m: float = 2.5,
             spacing_m: float = 6.0, translate_to=(8.0, 0.0),
             target=(16.0, 3.0), task: str = "inspection",
             square_side: float = 5.0, n_slots: int = 2,
             spread_deg: float = 45.0, markers: bool = True,
             estimator_hz: float = 10.0, fly: bool = True,
             sign: bool = False, keystore_path=None,
             probe_unsigned: bool = True, guard: bool = False,
             verbose: bool = True) -> LoopResult:
    """S0-S3: the range-only closed loop against live ArduCopter SITL.

    Start order is the plan's and is not negotiable:
      SITL up -> truth flowing -> params -> external nav priming + origin
      -> estimator converges AT REST -> arm -> takeoff -> command -> land.

    `sign` inserts COMMS_HARDENING_PLAN.md stage H0 into step 1: a per-vehicle
    MAVLink 2 signing key is generated (or reused), pushed with `SETUP_SIGNING`,
    and every subsequent frame this process sends is authenticated. It is opt-in
    rather than default because the estimator is the thing under repair right
    now and an unsigned run must stay reproducible; there is no argument for
    leaving it off once that work lands.
    """
    from sim.external_nav_fanout import ExternalNavFanout, FanoutConfig
    from sim.ground_truth_bridge import GroundTruthBridge, default_frame
    from sim.live_estimator import LiveEstimator
    from sim.range_world import DEFAULT_ANCHORS, RangeWorld

    frame_tac = default_frame()
    bridge = GroundTruthBridge(frame_tac, n)
    result = LoopResult()

    def say(*a):
        if verbose:
            print(*a, flush=True)

    say(f"\nD4 closed loop — {n} vehicle(s), mode {mode!r}, "
        f"alt {alt_m:.1f} m, no GPS, no vision")
    say("=" * 84)

    say("[1/7] connecting to SITL and verifying the ground-truth source")
    keystore = _open_keystore(keystore_path, say) if sign else None
    bridge.connect(verbose=verbose, keystore=keystore)
    if not bridge.wait_for_truth(timeout_s=30.0):
        raise SystemExit(
            "no ground truth — is SIMSTATE streaming?"
            + ("\nSigning is ON: if the fleet was healthy before --sign, the "
               "likely cause is a key the vehicle does not share. Check "
               "sim/logs/vehicle_*.log for a SETUP_SIGNING complaint, or "
               "relaunch with --wipe (run_fleet.sh already does) to clear the "
               "key it kept in FRAM from a previous session." if sign else ""))
    if keystore is not None and probe_unsigned:
        _verify_unsigned_is_refused(bridge, n, say)
    report = bridge.verify_sources()
    result.truth_sources = report["per_vehicle"]
    for i, row in report["per_vehicle"].items():
        say(f"      vehicle {i}: ground truth from {row['source']}")

    commanders = [VehicleCommander(i, bridge) for i in range(n)]
    world = RangeWorld(DEFAULT_ANCHORS, seed=17)
    estimator = LiveEstimator(n, DEFAULT_ANCHORS)

    # H1/H3 inbound defences. Off unless asked for, so an unguarded run stays
    # bit-for-bit the run that produced the published S0 numbers.
    from security.guards import LoopGuards
    guards = LoopGuards.from_run(guard, n, DEFAULT_ANCHORS, keystore=keystore)
    if guards is not None:
        say(f"      guards ON: range-mesh integrity (A3) + estimate "
            f"plausibility (A2)"
            + ("" if keystore is not None else
               ", audit log DISABLED (needs --sign for the chain key)"))
    dt_s = 1.0 / estimator_hz

    fanout = ExternalNavFanout(
        [bridge.link(i) for i in range(n)],
        FanoutConfig(hz=20.0, alt_m=alt_m,
                     origin_lat_deg=frame_tac.anchor_lat_deg,
                     origin_lon_deg=frame_tac.anchor_lon_deg,
                     origin_alt_m=frame_tac.anchor_alt_m))

    say("[2/7] setting EKF3 external-nav parameters (GPS_TYPE=0)")
    fanout.configure()

    say("[3/7] priming the external-nav stream, then SET_GPS_GLOBAL_ORIGIN")
    fanout.prime()          # starts the sender thread, then sets the origin

    period = 1.0 / estimator_hz
    try:
        say(f"[4/7] converging the estimator AT REST for {settle_s:.0f} s")
        t_end = time.monotonic() + settle_s
        while time.monotonic() < t_end:
            truth = _tick(bridge, world, estimator, fanout, result, guards, dt_s)
            _score(estimator, truth, result)
            time.sleep(period)
        rest = {"rmse": result.rmse_m[-1], "sigma": result.sigma_m[-1],
                "ratio": result.ratio[-1]}
        say(f"      at rest: {rest['rmse']:.3f} m error / "
            f"{rest['sigma']:.3f} m sigma = ratio {rest['ratio']:.2f}")
        if not fly:
            say("      --no-fly: stopping before arm")
            result.ticks = len(result.rmse_m)
            result.fanout = fanout.report()
            return result

        say("[5/7] GUIDED, arm, take off")
        for c in commanders:
            c.drain_status()
            c.set_mode(COPTER_MODE_GUIDED)
        time.sleep(1.0)
        for c in commanders:
            c.arm()
        armed = []
        t_end = time.monotonic() + 30.0
        retry_at = time.monotonic() + 5.0
        while time.monotonic() < t_end and len(armed) < n:
            _tick(bridge, world, estimator, fanout, result, guards, dt_s)
            for c in commanders:
                for line in c.drain_status():
                    say(f"      v{c.index}: {line}")
            armed = [c.index for c in commanders if c.armed]
            if time.monotonic() >= retry_at and len(armed) < n:
                # Re-issue rather than wait it out. Arming is refused until the
                # EKF has settled on the external-nav position, and how long
                # that takes depends on the estimator, so a single attempt at a
                # fixed moment is a race. Repeating is harmless: ArduPilot acks
                # each request independently.
                for c in commanders:
                    if not c.armed:
                        c.set_mode(COPTER_MODE_GUIDED)
                        c.arm()
                retry_at = time.monotonic() + 5.0
            time.sleep(period)
        if len(armed) < n:
            raise SystemExit(
                f"only vehicles {armed} armed — read the STATUSTEXT above; "
                f"streaming setpoints at unarmed vehicles would produce "
                f"numbers that mean nothing")
        for c in commanders:
            c.takeoff(alt_m)
        say(f"      armed {armed}, climbing to {alt_m:.1f} m")

        t_end = time.monotonic() + 10.0
        while time.monotonic() < t_end:
            truth = _tick(bridge, world, estimator, fanout, result, guards, dt_s)
            _score(estimator, truth, result)
            time.sleep(period)

        say(f"[6/7] flying mode {mode!r} for {seconds:.0f} s")
        orch = Orchestrator(commanders,
                            OrchestratorConfig(alt_m=alt_m, spacing_m=spacing_m))
        coalition: list[int] = []
        stations: list = []
        if mode == "hold":
            goals = [orch.hold()]
        elif mode == "square":
            goals = orch.square(side=square_side)
        elif mode == "translate":
            goals = [orch.translate(estimator.pos, translate_to, world)]
        elif mode == "dispatch":
            station, _ = orch.dispatch(0, estimator.pos, target, task)
            goal = orch.hold()
            goal[0] = station
            goals = [goal]
        elif mode == "standoff":
            election = orch.elect(estimator.pos, target, n_slots=n_slots,
                                  cov_trace=estimator.cov_trace)
            coalition = sorted(election.winners)
            result.coalition = coalition
            say(f"      AUCTION: coalition {coalition} elected in "
                f"{election.rounds} round(s), converged={election.converged}")
            goals, stations, paths = orch.dispatch_coalition(
                coalition, estimator.pos, target, task, spread_deg=spread_deg)
            result.stations = [s.tolist() for s in stations]
            result.target = list(map(float, np.asarray(target, float)[:2]))
            result.planned = goals[:, coalition, :].tolist()
            for agent, st, pth in zip(coalition, stations, paths):
                say(f"      v{agent} -> station ({st[0]:.2f}, {st[1]:.2f})  "
                    f"{np.linalg.norm(np.asarray(target, float) - st):.1f} m "
                    f"from target, Dubins {pth.word} {pth.length:.1f} m")
            say(f"      {len(goals)} rung(s) at <= "
                f"{orch.cfg.limits.v_max:.2f} m per tick")
            if markers:
                from sim import qgc_markers
                marks = qgc_markers.markers_for_dispatch(
                    target, stations, coalition, alt_m=alt_m)
                qgc_markers.publish(bridge, frame_tac, marks, range(n),
                                    send_lock=fanout._send_lock,
                                    verbose=verbose)
        else:
            raise ValueError(f"unknown mode {mode!r}")
        say(f"      {len(goals)} waypoint(s); first "
            f"{np.round(goals[0], 2).tolist()}")

        leg = 0
        stall_warned = False
        last_advance = time.monotonic()
        plan_period = 1.0 / orch.cfg.plan_hz
        next_plan = time.monotonic()
        t_end = time.monotonic() + seconds
        while time.monotonic() < t_end:
            truth = _tick(bridge, world, estimator, fanout, result, guards, dt_s)
            _score(estimator, truth, result)
            result.alt_agl_m.append(float(np.nanmean(bridge.altitudes_agl())))
            if coalition:
                # Scored against TRUTH, not against the estimate. "The vehicle
                # thinks it is on station" is the claim this project refuses to
                # accept on its own.
                result.station_error_m.append(float(np.mean([
                    np.linalg.norm(truth[a] - stations[k])
                    for k, a in enumerate(coalition)])))
                for a in coalition:
                    result.tracks.setdefault(a, []).append(
                        [float(truth[a][0]), float(truth[a][1])])
            now = time.monotonic()
            if now >= next_plan:
                snap = EstimateSnapshot(
                    frame_index=estimator.t,
                    stamp_unix_ms=supervisor_io.now_ms(),
                    pos=[(float(p[0]), float(p[1])) for p in estimator.pos],
                    cov_trace=[float(c) for c in estimator.cov_trace])
                t0 = time.monotonic()
                _, arrived = orch.step_toward(estimator.pos, goals[leg], snap,
                                              mode,
                                              movers=coalition or None)
                result.setpoint_latency_ms.append((time.monotonic() - t0) * 1e3)
                # A ladder that stops advancing is a stalled mission, not a
                # flown one. It reads as success in every summary line, so it
                # has to announce itself while there is still time to watch it.
                if leg + 1 < len(goals) and not stall_warned \
                        and time.monotonic() - last_advance > 10.0:
                    stall_warned = True
                    gf = orch.log.guard_failures
                    why = (f"guard refusing: {gf[-1]['kinds']} "
                           f"min_clearance {gf[-1]['min_clearance']:.2f} m"
                           if gf else
                           "no guard failure — vehicles are not reaching the "
                           "rung within arrive_tol_m")
                    say(f"      STALLED at rung {leg}/{len(goals) - 1} for 10 s "
                        f"— {why}")
                if arrived and leg + 1 < len(goals):
                    leg += 1
                    last_advance = time.monotonic()
                    result.legs_flown = leg
                    if coalition:
                        # Report the MOVING vehicles. `goals[leg][0]` is vehicle
                        # 0's slot, and vehicle 0 is usually a holder here — a
                        # log line that never changes reads like a stuck loop.
                        if leg == 1 or leg % 10 == 0 or leg == len(goals) - 1:
                            where = ", ".join(
                                f"v{a} {np.round(goals[leg][a], 1).tolist()}"
                                for a in coalition)
                            say(f"      rung {leg}/{len(goals) - 1}   {where}")
                    else:
                        say(f"      leg {leg}/{len(goals) - 1} -> "
                            f"{np.round(goals[leg][0], 2).tolist()}")
                next_plan = now + plan_period
            time.sleep(period)
        result.orchestrator = orch.report()

        say("[7/7] landing")
        for c in commanders:
            c.land()
        t_end = time.monotonic() + 15.0
        while time.monotonic() < t_end:
            truth = _tick(bridge, world, estimator, fanout, result, guards, dt_s)
            _score(estimator, truth, result)
            time.sleep(period)
        for c in commanders:
            c.disarm()
    finally:
        fanout.stop()
        result.ticks = len(result.rmse_m)
        result.fanout = fanout.report()
        # Read the counters BEFORE closing: they live on the link's codec.
        result.signing = bridge.signing_report()
        if guards is not None:
            guards.audit.checkpoint("run end")
            result.guards = guards.report()
        bridge.close()

    return result


def print_result(res: LoopResult) -> None:
    s = res.summary()
    print("\n" + "=" * 84)
    print("D4 closed-loop result")
    print("-" * 84)
    print(f"  ticks                    {s['ticks']}   "
          f"(first {s['warmup']} excluded: cold start)")
    print(f"  estimator error (mean)   {s['rmse_mean_m']:.3f} m   "
          f"(max {s['rmse_max_m']:.3f} m)")
    print(f"  reported sigma  (mean)   {s['sigma_mean_m']:.3f} m")
    print(f"  ERROR / SIGMA            {s['ratio_mean']:.2f} mean, "
          f"{s['ratio_max']:.2f} max   "
          f"(cold-start peak {s['cold_start_peak_ratio']:.1f})")
    if res.alt_agl_m:
        alt = np.asarray(res.alt_agl_m)
        print(f"  altitude held            {alt.min():.2f}-{alt.max():.2f} m "
              f"AGL over {len(alt)} ticks   <- it actually left the ground")
    if res.legs_flown:
        print(f"  waypoint legs completed  {res.legs_flown}")
    print( "                           <- ~1 is the acceptance bar. Below 1 is "
           "under-confident;")
    print( "                              far above 1 is the confidently-wrong "
           "estimate the")
    print( "                              supervisor would certify, which is the "
           "whole thesis.")
    print("-" * 84)
    print(f"  range-in -> extnav-out   p50 {s['est_latency_p50_ms']:.1f} ms   "
          f"p99 {s['est_latency_p99_ms']:.1f} ms   (target < 25 ms)")
    if res.fanout:
        print(f"  ... to the wire          p50 {res.fanout['air_latency_p50_ms']:.1f} ms  "
              f"p99 {res.fanout['air_latency_p99_ms']:.1f} ms")
        print(f"  extnav stream            {res.fanout['sent']} sends, worst gap "
              f"{res.fanout['max_gap_ms']:.1f} ms "
              f"({'CONTINUOUS' if res.fanout['continuous'] else 'GAPPED'})")
    if res.setpoint_latency_ms:
        print(f"  plan -> gate -> setpoint p50 {s['setpoint_latency_p50_ms']:.1f} ms  "
              f"p99 {s['setpoint_latency_p99_ms']:.1f} ms")
    if res.coalition:
        print("-" * 84)
        print(f"  coalition elected        {res.coalition}   (CBBA auction)")
        for k, a in enumerate(res.coalition):
            st = res.stations[k]
            print(f"    v{a} station           ({st[0]:.2f}, {st[1]:.2f})")
        if res.station_error_m:
            e = np.asarray(res.station_error_m)
            print(f"  distance to station      final {e[-1]:.2f} m, "
                  f"best {e.min():.2f} m   <- against TRUTH, not the estimate")
    if res.orchestrator:
        o = res.orchestrator
        print("-" * 84)
        print(f"  plans submitted          {o['plans']}")
        print(f"  supervisor ACCEPT        {o['accepted']}")
        print(f"  supervisor REJECT        {o['rejected']}   "
              f"<- any non-zero is a bug report, not an event")
        print(f"  horizon guard failures   {o['guard_failures']}")
        print(f"  setpoints emitted        {o['setpoints']}")
        print(f"  gate call                p50 {o['gate_call_p50_ms']:.1f} ms  "
              f"p99 {o['gate_call_p99_ms']:.1f} ms")
    if res.signing is not None:
        print("-" * 84)
        print("  MAVLink 2 signing        ON (per-vehicle keys, reject-unsigned)")
        for i, row in sorted(res.signing["per_vehicle"].items()):
            rej = row.get("rejections", {})
            # goodsig is the one that proves the link worked: it counts frames
            # the VEHICLE signed and we verified. Zero here with a healthy
            # flight means the far end never loaded the key.
            print(f"    vehicle {i}  key {res.signing['fingerprints'][i]}   "
                  f"verified {row['goodsig']}   bad-sig {row['badsig']}   "
                  f"unsigned-refused {rej.get('unsigned', row['unsigned_rejected'])}")
        total_bad = sum(r["badsig"] for r in res.signing["per_vehicle"].values())
        if total_bad:
            print(f"    {total_bad} frame(s) failed verification — §6.1 security "
                  f"event, not noise. Investigate before flying again.")
    if res.guards is not None:
        g = res.guards
        print("-" * 84)
        print("  inbound guards           ON (A3 range mesh, A2 estimate path)")
        print(f"    range links refused    {g['ranges']['dropped']}   "
              f"{g['ranges']['by_kind'] or ''}")
        print(f"    estimates refused      {g['estimates']['total']}   "
              f"{g['estimates']['by_kind'] or ''}")
        if g["audit_seq"]:
            print(f"    audit chain            {g['audit_seq']} record(s), "
                  f"HMAC-chained")
        if g["ranges"]["dropped"] or g["estimates"]["total"]:
            print("    <- non-zero means either an injected measurement or an "
                  "estimator fault.")
            print("       The gate does not distinguish them, because the safe "
                  "response is the same.")
    print("=" * 84)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description="D4.5 — the range-only MAVLink-native closed loop")
    ap.add_argument("-n", "--vehicles", type=int, default=1)
    ap.add_argument("-m", "--mode", default="hold",
                    choices=("hold", "square", "translate", "dispatch",
                             "standoff"))
    ap.add_argument("--square-side", type=float, default=5.0)
    ap.add_argument("--target", type=float, nargs=2, metavar=("X", "Y"),
                    default=(25.0, 8.0),
                    help="the detector's coordinate, TacFrame ENU metres. The "
                         "default sits far enough out that the whole loiter "
                         "mesh starts OUTSIDE the standoff perimeter; D3's own "
                         "figure uses (16, 3) with a single agent that starts "
                         "20.6 m away, which a 6 m mesh does not")
    ap.add_argument("--task", default="inspection",
                    choices=tuple(TASKS), help="standoff geometry")
    ap.add_argument("--slots", type=int, default=2,
                    help="coalition size the auction elects (standoff mode)")
    ap.add_argument("--spread", type=float, default=45.0,
                    help="degrees between coalition stations on the perimeter")
    ap.add_argument("--no-markers", action="store_true",
                    help="skip the QGroundControl marker mission upload")
    ap.add_argument("-s", "--seconds", type=float, default=40.0)
    ap.add_argument("--settle", type=float, default=12.0,
                    help="seconds of AT-REST convergence before arming")
    ap.add_argument("--alt", type=float, default=2.5)
    ap.add_argument("--spacing", type=float, default=6.0)
    ap.add_argument("--hz", type=float, default=10.0, help="estimator rate")
    ap.add_argument("--no-fly", action="store_true",
                    help="S0 de-risk: converge at rest and stop before arming")
    ap.add_argument("--sign", action="store_true",
                    help="COMMS_HARDENING_PLAN.md H0: per-vehicle MAVLink 2 "
                         "signing keys, reject-unsigned. Needs "
                         "$HIVE_KEYSTORE_PASSPHRASE")
    ap.add_argument("--keystore", default=None,
                    help="signing keystore path (default ~/.hive/keys/"
                         "keystore.json, or $HIVE_KEYSTORE)")
    ap.add_argument("--no-probe-unsigned", action="store_true",
                    help="skip the check that the VEHICLE refuses an unsigned "
                         "command. Only meaningful with --sign, and it turns a "
                         "hard stop into a link authenticated one way only")
    ap.add_argument("--guard", action="store_true",
                    help="COMMS_HARDENING_PLAN.md H1/H3: refuse implausible "
                         "ranges (A3) and implausible position estimates (A2). "
                         "Independent of --sign, which authenticates the "
                         "sender but says nothing about whether the payload "
                         "can be true. Combine with --sign for the "
                         "tamper-evident audit chain")
    args = ap.parse_args()

    res = run_loop(n=args.vehicles, mode=args.mode, seconds=args.seconds,
                   settle_s=args.settle, alt_m=args.alt,
                   spacing_m=args.spacing, square_side=args.square_side,
                   target=tuple(args.target), task=args.task,
                   n_slots=args.slots, spread_deg=args.spread,
                   markers=not args.no_markers,
                   estimator_hz=args.hz, fly=not args.no_fly,
                   sign=args.sign, keystore_path=args.keystore,
                   probe_unsigned=not args.no_probe_unsigned,
                   guard=args.guard)
    print_result(res)

    if res.coalition and res.tracks:
        # The flight is the evidence, so it is written down: the raw track next
        # to the figure drawn from it, rather than a figure nobody can re-check.
        blob = {"target": res.target, "stations": res.stations,
                "coalition": res.coalition, "planned": res.planned,
                "tracks": {str(k): v for k, v in res.tracks.items()},
                "station_error_m": res.station_error_m}
        out = pathlib.Path(__file__).resolve().parent / "logs" / "standoff.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(blob))
        from hive.plots import plot_live_standoff
        print(f"  track   -> {out}")
        print(f"  figure  -> {plot_live_standoff(blob, task=args.task)}")


if __name__ == "__main__":
    main()
