"""D4.3 — the safe-hold property, proven without a simulator.

Domain 4's highest-value experiment is the netem sweep: degrade the link 0-30%
and verify the failure mode is *always* "freeze safely" —

    delayed estimate  -> EstimateStale fires
    dropped plan      -> zero setpoints emitted, the vehicle holds

That sweep runs on the Zephyrus. This module runs the same experiment here,
against the same validator, with the network modelled instead of emulated. Two
reasons that is worth doing rather than waiting:

  1. If safe-hold only holds on one particular host, it was never a property of
     the design. Modelling it makes the claim portable and puts it in CI.
  2. It found something. See below.

WHAT IT FOUND: THE POST-BLACKOUT LUNGE.

The plan asserts the failure mode is "vehicle holds (never lunges)". Half of
that is already true and needs no defending — a rejected plan emits nothing, so
during an outage the vehicle sits still. The *lunge* is not about the outage. It
is about the recovery.

While the vehicle is frozen, the planner upstream keeps advancing its stream: it
is compiling tick 40, then 60, then 80, unaware that ticks 41-79 never landed.
When the link comes back, the first plan to arrive is for tick 80 — a waypoint
tens of metres from where the vehicle has been parked. The supervisor checks it
for geofence, spacing, covariance and freshness. It is *fresh* — it was issued a
moment ago. It is inside the fence. Spacing is fine. So it is ACCEPTED, and the
vehicle is commanded to cross the intervening distance in one tick.

Nothing in the current supervisor forbids this, because every gate it has is a
predicate on a single instant. Freshness bounds how old a plan may be; nothing
bounds how far a plan may ask a vehicle to move from where it actually is.

The fix is one more gate, and it is cheap: reject any assignment whose waypoint
is further from that vehicle's estimated position than it could travel since the
last accepted command. `SlewGate` below implements it, and the sweep shows the
difference: without it, commanded jumps grow with packet loss; with it, the
worst commanded jump is flat across the whole 0-30% sweep, which is what
"freeze safely" was supposed to mean.

This is a recommendation for brain/rust/swarm-supervisor/src/lib.rs, not a
change to it — the Rust gate stays authoritative and unmodified until someone
decides to add the gate there.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from hive.supervisor_gate import (
    Assignment,
    EstimateSnapshot,
    Plan,
    SupervisorConfig,
    validate,
)


# --------------------------------------------------------------------------
# The link
# --------------------------------------------------------------------------
@dataclass
class LinkModel:
    """A `tc netem` qdisc, in Python.

    Defaults mirror the plan's example:
        tc qdisc add dev veth-drone7 root netem \\
           delay 50ms 15ms distribution normal loss 8% reorder 2%
    """

    loss: float = 0.08
    delay_ms: float = 50.0
    jitter_ms: float = 15.0
    reorder: float = 0.02
    reorder_shift_ms: float = 40.0

    def deliver(self, send_ms: float, rng: np.random.Generator) -> float | None:
        """Arrival time for a packet sent at `send_ms`, or None if dropped."""
        if rng.random() < self.loss:
            return None
        d = self.delay_ms + rng.normal(0.0, self.jitter_ms)
        if rng.random() < self.reorder:
            d -= self.reorder_shift_ms          # jumps the queue
        return send_ms + max(d, 0.0)


# --------------------------------------------------------------------------
# The extra gate
# --------------------------------------------------------------------------
@dataclass
class SlewGate:
    """Reject a waypoint the vehicle could not reach in ONE tick.

        budget = v_max * dt + tolerance

    The first version of this budgeted `v_max * (time since last command)`, on
    the reasoning that a vehicle held for two seconds has earned a larger step.
    That is exactly backwards and it authorises the precise failure this gate
    exists to stop: the vehicle *held*, so it accumulated no travel, and paying
    it for the wait licenses a 2.7 m lunge as "affordable". The budget is
    per-tick because the command is per-tick.

    The consequence is deliberate and worth stating plainly: after an outage a
    stale stream is rejected forever, because it keeps advancing away from a
    vehicle that never moved. The system then holds indefinitely — safe, and
    stalled. Recovery is not the gate's job; it is the planner's, and it happens
    by re-planning from where the vehicle actually is. `run_link(replan=True)`
    models that, and the FSM already works this way: `MissionFSM` compiles every
    stream from `self.pos`, never from where a previous plan hoped it would be.
    """

    v_max: float = 0.9              # metres per tick interval
    dt: float = 1.0                 # tick length, seconds
    tolerance_m: float = 0.25       # estimator noise allowance
    enabled: bool = True

    @property
    def budget(self) -> float:
        return self.v_max * self.dt + self.tolerance_m

    def check(self, plan: Plan, est: EstimateSnapshot) -> list[dict]:
        if not self.enabled:
            return []
        out = []
        for a in plan.assignments:
            if not (0 <= a.vehicle < len(est.pos)):
                continue
            p = np.asarray(est.pos[a.vehicle], dtype=float)
            d = float(np.linalg.norm(np.asarray(a.waypoint_ne, float) - p))
            if d > self.budget:
                out.append({"kind": "SlewTooLarge", "vehicle": a.vehicle,
                            "dist": d, "budget": self.budget})
        return out


# --------------------------------------------------------------------------
# The experiment
# --------------------------------------------------------------------------
@dataclass
class SafeHoldResult:
    loss: float
    ticks: int
    accepted: int
    held: int
    max_commanded_jump: float
    max_hold_gap_ticks: int
    progress: float = 0.0            # fraction of the way to the goal
    violations: dict[str, int] = field(default_factory=dict)
    lunged: bool = False

    @property
    def emit_rate(self) -> float:
        return self.accepted / max(self.ticks, 1)


def run_link(stream, cfg: SupervisorConfig, link: LinkModel,
             gate: SlewGate | None = None, tick_ms: int = 1000,
             seed: int = 0, est_loss_scale: float = 1.0,
             replan: bool = False, v_max: float = 0.9) -> SafeHoldResult:
    """Push a pre-validated setpoint stream through a degraded link.

    `stream` is (H, K, 2) — the FSM's own output, so the thing under test is the
    transport, not the plan. The vehicle holds its last accepted command; a tick
    that accepts nothing emits nothing and moves nothing.

    `tick_ms` must match the stream's own tick length (StreamLimits.dt), or
    v_max means something different on each side of the link. It defaults to
    1000 ms because the FSM's streams are compiled at dt = 1 s.

    `replan=False` models a planner that keeps advancing its stream while the
    vehicle is frozen — the realistic default, and the one that lunges.
    `replan=True` models a planner that re-anchors to the vehicle's actual
    position after a hold, which is what MissionFSM does.
    """
    s = np.asarray(stream, dtype=float)
    h, k, _ = s.shape
    rng = np.random.default_rng(seed)
    gate = gate or SlewGate(enabled=False)
    goal = s[-1].copy()
    step_max = v_max * (tick_ms / 1000.0)

    est_link = LinkModel(loss=min(1.0, link.loss * est_loss_scale),
                         delay_ms=link.delay_ms, jitter_ms=link.jitter_ms,
                         reorder=link.reorder)

    commanded = s[0].copy()                  # where the vehicles actually are
    last_cmd_ms = {i: 0.0 for i in range(k)}
    inbox_plan: list[tuple[float, int]] = []
    inbox_est: list[tuple[float, int]] = []

    accepted = held = 0
    max_jump = 0.0
    gap = worst_gap = 0
    counts: dict[str, int] = {}

    for t in range(h):
        now = t * tick_ms

        # the planner emits on schedule regardless of what is getting through —
        # this is the crux: upstream has no idea the vehicle is frozen
        a = link.deliver(now, rng)
        if a is not None:
            inbox_plan.append((a, t))
        a = est_link.deliver(now, rng)
        if a is not None:
            inbox_est.append((a, t))

        # freshest arrival so far (reordering means "latest sent", not "last in")
        arrived_p = [(sent, at) for at, sent in inbox_plan if at <= now]
        arrived_e = [(sent, at) for at, sent in inbox_est if at <= now]
        if not arrived_p or not arrived_e:
            held += 1
            gap += 1
            worst_gap = max(worst_gap, gap)
            continue

        p_tick, _ = max(arrived_p)
        e_tick, _ = max(arrived_e)

        if replan:
            # re-anchor to where the vehicle actually is: one bounded step from
            # `commanded` toward the goal, exactly what MissionFSM would emit
            d_goal = goal - commanded
            n = np.linalg.norm(d_goal, axis=1, keepdims=True)
            wp = commanded + d_goal * np.minimum(1.0, step_max / np.maximum(n, 1e-12))
        else:
            wp = s[p_tick]

        plan = Plan(
            plan_id=f"t{t:04d}",
            issued_unix_ms=p_tick * tick_ms,
            assignments=[Assignment(i, tuple(wp[i])) for i in range(k)],
        )
        est = EstimateSnapshot(
            frame_index=e_tick, stamp_unix_ms=e_tick * tick_ms,
            pos=[tuple(p) for p in commanded], cov_trace=[0.001] * k,
        )

        d = validate(plan, est, cfg, now)
        extra = gate.check(plan, est)
        for v in d.violations + extra:
            counts[v["kind"]] = counts.get(v["kind"], 0) + 1

        if d.accepted and not extra:
            new = np.array([a.waypoint_ne for a in plan.assignments], dtype=float)
            max_jump = max(max_jump, float(np.linalg.norm(new - commanded,
                                                          axis=1).max()))
            commanded = new
            for i in range(k):
                last_cmd_ms[i] = now
            accepted += 1
            gap = 0
        else:
            held += 1                        # zero setpoints emitted; hold
            gap += 1
            worst_gap = max(worst_gap, gap)

    start_gap = float(np.linalg.norm(s[-1] - s[0], axis=1).mean())
    left = float(np.linalg.norm(goal - commanded, axis=1).mean())
    return SafeHoldResult(
        loss=link.loss, ticks=h, accepted=accepted, held=held,
        max_commanded_jump=max_jump, max_hold_gap_ticks=worst_gap,
        progress=1.0 - left / max(start_gap, 1e-9),
        violations=counts,
        # a "lunge" is a commanded step the vehicle cannot physically fly
        lunged=max_jump > step_max * 1.5,
    )


def dispatch_stream(horizon: int = 120):
    """The FSM's own dispatch stream, so the sweep degrades real setpoints."""
    from hive.mission_fsm import default_scenario, slew_toward

    swarm, cfg = default_scenario()
    start = swarm.pos[[2, 3]]
    goal = np.array([[13.2, 1.0], [13.2, 5.4]])
    return slew_toward(start, goal, cfg.limits, horizon), cfg


# the three configurations the sweep compares
CONFIGS = {
    "stale_ungated": ("stale stream, no slew gate", False, False),
    "stale_gated": ("stale stream, slew gate", True, False),
    "replan_gated": ("re-plan on hold, slew gate", True, True),
}


def sweep(losses=None, seeds: int = 12, horizon: int = 120):
    """The plan's 0-30% loss sweep, across the three configurations."""
    losses = np.array([0.0, 0.02, 0.05, 0.08, 0.12, 0.18, 0.24, 0.30]) \
        if losses is None else np.asarray(losses, dtype=float)
    stream, cfg = dispatch_stream(horizon)
    tick_ms = int(cfg.limits.dt * 1000)
    out = {"loss": losses, "step": cfg.limits.v_max * cfg.limits.dt}
    out.update({k: [] for k in CONFIGS})

    for loss in losses:
        for key, (_, gated, replan) in CONFIGS.items():
            gate = SlewGate(v_max=cfg.limits.v_max, dt=cfg.limits.dt,
                            enabled=gated)
            runs = [run_link(stream, cfg.supervisor,
                             LinkModel(loss=float(loss)), gate,
                             tick_ms=tick_ms, seed=s, replan=replan,
                             v_max=cfg.limits.v_max)
                    for s in range(seeds)]
            out[key].append({
                "loss": float(loss),
                "emit_rate": float(np.mean([r.emit_rate for r in runs])),
                "max_jump": float(np.max([r.max_commanded_jump for r in runs])),
                "worst_gap": int(np.max([r.max_hold_gap_ticks for r in runs])),
                "progress": float(np.mean([r.progress for r in runs])),
                "lunged": int(sum(r.lunged for r in runs)),
                "runs": len(runs),
            })
    return out


def main() -> None:
    res = sweep()
    step = res["step"]
    n = res["stale_ungated"][0]["runs"]

    print(f"\nnetem loss sweep — safe-hold under degradation "
          f"({n} seeds/point; one tick of legitimate motion = {step:.2f} m)")
    print("=" * 100)
    print(f"{'configuration':<30}{'loss':>7}{'worst hold':>13}"
          f"{'max commanded jump':>21}{'mission progress':>19}{'lunges':>9}")
    print("-" * 100)
    for key, (label, _, _) in CONFIGS.items():
        for r in res[key]:
            flag = "  <-- LUNGE" if r["lunged"] else ""
            print(f"{label:<30}{r['loss']*100:>6.0f}%"
                  f"{r['worst_gap']:>10} ticks{r['max_jump']:>19.2f} m"
                  f"{r['progress']*100:>17.0f}%{r['lunged']:>5}/{r['runs']}{flag}")
        print("-" * 100)
    print("=" * 100)

    u = max(r["max_jump"] for r in res["stale_ungated"])
    g = max(r["max_jump"] for r in res["stale_gated"])
    rp = max(r["max_jump"] for r in res["replan_gated"])
    prog_g = res["stale_gated"][-1]["progress"]
    prog_r = res["replan_gated"][-1]["progress"]

    print(f"""
Reading the three rows:

  stale + no gate    worst commanded jump {u:.1f} m = {u/step:.0f}x a legal tick. Every existing gate
                     passes it: the plan is fresh, in-fence, correctly spaced, from a trusted
                     vehicle. Freshness bounds a plan's AGE; nothing bounds its DISTANCE.

  stale + gate       jump held to {g:.2f} m, but mission progress collapses to {prog_g*100:.0f}% at 30% loss.
                     Safe and stalled: a stream that keeps advancing away from a frozen
                     vehicle is rejected forever, and correctly so.

  re-plan + gate     jump {rp:.2f} m AND {prog_r*100:.0f}% progress at 30% loss. Holding during the outage,
                     resuming at walking pace, finishing the mission. This is the failure
                     mode Domain 4 asked for, and it needs both halves.

Recommendation for brain/rust/swarm-supervisor/src/lib.rs: add a SlewTooLarge
violation. It is ~10 lines and it is the only gate that makes "never lunges" true
rather than merely intended.""")

    from hive.plots import plot_safe_hold
    print(f"\nfigure -> {plot_safe_hold(res)}")


if __name__ == "__main__":
    main()
