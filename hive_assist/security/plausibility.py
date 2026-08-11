"""H4.2 — refuse a position estimate that cannot be true, however well signed.

Stage H1 of `COMMS_HARDENING_PLAN.md`, and per §8 the highest-value item left
after signing, because it is the only one that addresses **A2** — the surface
unique to this architecture.

The argument, restated so this file stands on its own: these vehicles fly on an
externally injected position. An attacker who can put a `VISION_POSITION_ESTIMATE`
on the wire does not need to touch the motors, the mission, or the supervisor.
They redefine where the drone believes it is and let a correctly-functioning
autopilot fly it into the ground. Signing (H0) stops an *outsider* forging that
stream. It does exactly nothing about a stream that is authenticated and wrong —
and §3.1's forensic history is the standing proof that this estimator can be
confidently wrong entirely on its own, with no attacker present at all.

So the same gate closes two doors that look unrelated and are not:

    a forged estimate  ─┐
                        ├─▶ "a position that cannot be true"  ─▶ refuse, hold
    a broken estimator ─┘

That equivalence is the reason this is worth building. A spoof detector that
only caught spoofs would be dead weight most of the time; this one earns its
place on every flight by catching the estimator's own failures, and catches the
attack for free.

---------------------------------------------------------------------------
WHY THIS IS NOT IN THE SUPERVISOR
---------------------------------------------------------------------------

The obvious home is `swarm-supervisor`, and the plan (H4.2) says "extends
swarm-supervisor". It is deliberately not there, for two reasons.

1. **Parity.** `hive/supervisor_gate.py` is a Python mirror of the Rust gate and
   `tests/test_gate_parity.py` diffs the two on randomised cases. Adding a check
   to one side and not the other turns that test red for a reason that is not a
   bug; adding it to both means editing the Rust crate the plan holds fixed.

2. **It guards the wrong direction.** The supervisor gates *outbound* plans —
   "may this setpoint be commanded?". A2 is an *inbound* problem — "may this
   position be believed?". By the time a bad estimate reaches the supervisor it
   has already been fused by EKF3 and is the vehicle's truth; the supervisor's
   covariance check reads the very number the attacker controls. The gate has
   to sit upstream of the fan-out, on the estimate path, which is where this is.

The two compose rather than overlap: this decides what the vehicle is allowed to
*believe*, the supervisor decides what it is allowed to *do*.

---------------------------------------------------------------------------
THE CHECKS, AND WHY THEY ARE IN THIS ORDER
---------------------------------------------------------------------------

Order matters because a violation short-circuits and the first reason reported
should be the most fundamental one. A NaN position is not "a teleport".

  1. `not_finite`   NaN/inf in position or covariance.
  2. `stale`        the estimate is older than the freshness window.
  3. `off_map`      position outside any plausible operating footprint.
  4. `teleport`     moved further since the last tick than `v_max` allows.
  5. `cov_exceeded` reported covariance beyond the trust threshold.
  6. `sigma_lie`    THE ONE THAT MATTERS — see below.

**`sigma_lie` is the project's own thesis as an executable check.** Everything
else on that list catches an estimate that is *obviously* wrong. This one
catches the dangerous case: an estimate that is wrong while reporting that it is
precise, which is exactly what defeats every downstream covariance gate.

The mechanic needs no ground truth, which is what makes it survive contact with
real hardware. Take the vehicle's own two previous accepted positions, predict
the current one under constant velocity, and compare the innovation against the
uncertainty the estimator *claims*:

    p_pred     = 2·p[t-1] − p[t-2]
    innovation = ‖p[t] − p_pred‖
    σ_pred     = √6 · σ            (variance of a 2a−b combination of three
                                    independent draws is 4σ² + σ² + σ²)
    allowance  = ½·a_max·Δt²       (room for a genuine manoeuvre)

    reject if innovation > allowance + ratio · σ_pred

A vehicle reporting σ = 1.7 cm while jumping five metres from where its own
recent motion says it should be is making two claims that cannot both hold. It
does not matter which one is false — an attacker moved it, or the estimator
broke — because the safe response is identical.

---------------------------------------------------------------------------
HYSTERESIS, AND WHY RECOVERY IS SLOW ON PURPOSE
---------------------------------------------------------------------------

A vehicle that trips does not regain trust on the first clean tick. It must
produce `recover_ticks` consecutive plausible estimates first. Without that, an
attacker who straddles a threshold gets the vehicle flapping between trusted and
untrusted, and every other tick of poison lands. Slow recovery costs a little
mission availability during a genuine glitch and removes an entire attack shape,
which is the right trade for something on the safety path.

While untrusted, the vehicle is NOT cut off — `ExternalNavFanout.publish` keeps
re-sending its last good position. Stopping the stream would look like sensor
death and provoke a failsafe; forwarding the bad fix is the thing this file
exists to prevent. Holding is the only option that neither lies nor gaps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .audit_log import NullAuditLog


@dataclass
class PlausibilityConfig:
    """Limits. Defaults are the Domain 4 airframe's, not universal truths."""

    # The vehicle's own kinematic ceiling. A position that outruns this did not
    # come from the vehicle. Generous by design: false positives here cost
    # mission availability, and the check is meant to catch metres, not
    # centimetres.
    v_max_mps: float = 5.0
    a_max_mps2: float = 4.0
    # Absolute slack added to every jump test, covering tick jitter and the
    # discrete-time edge where two ticks arrive nearly together.
    jump_margin_m: float = 0.75
    # Mirrors supervisor.json's max_cov_trace. Kept separate rather than
    # imported so this module has no dependency on the orchestrator's config
    # loading, but a deployment should set them from the same source.
    max_cov_trace: float = 0.02
    max_age_ms: int = 1000
    # How many sigmas of innovation before we call the reported sigma a lie.
    # 6 is deliberately loose: at 3 a hard but legitimate manoeuvre trips it.
    sigma_ratio_max: float = 6.0
    # Radius about the TacFrame origin beyond which a position is absurd. The
    # anchors sit on a 50 m square and r_anchor is 120 m, so anything past a
    # few hundred metres is either a spoof or a frame bug.
    footprint_m: float = 500.0
    # Consecutive clean ticks required to regain trust after a rejection.
    recover_ticks: int = 5
    # A vehicle with no history cannot be motion-checked. Accept the first
    # estimate (there is nothing to compare it to) but never more than this
    # many before the motion checks become mandatory.
    warmup_ticks: int = 2


@dataclass
class Violation:
    vehicle: int
    kind: str
    detail: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {"vehicle": int(self.vehicle), "kind": self.kind,
                "detail": self.detail}


@dataclass
class PlausibilityVerdict:
    """One tick's decision for the whole fleet."""

    trusted: np.ndarray                 # bool per vehicle — feed to publish()
    violations: list[Violation] = field(default_factory=list)
    held: np.ndarray | None = None      # untrusted *because* of hysteresis

    @property
    def any_rejected(self) -> bool:
        return bool(self.violations)

    def kinds(self) -> list[str]:
        return [v.kind for v in self.violations]

    def to_json(self) -> dict:
        return {"trusted": [bool(t) for t in self.trusted],
                "violations": [v.to_json() for v in self.violations]}


class EstimateGate:
    """Stateful per-vehicle plausibility gate on the estimate stream.

    Stateful because the interesting checks are all differential — a single
    position in isolation is almost never implausible. Keeps the last two
    *accepted* positions per vehicle, so a rejected estimate never becomes the
    baseline that makes the next spoof look reasonable. That detail is the
    difference between a gate and a ratchet: without it, an attacker walks the
    vehicle anywhere in small legal steps.
    """

    def __init__(self, n: int, cfg: PlausibilityConfig | None = None,
                 audit=None) -> None:
        self.n = int(n)
        self.cfg = cfg or PlausibilityConfig()
        self.audit = audit if audit is not None else NullAuditLog()

        self._p1: list[np.ndarray | None] = [None] * self.n
        self._p2: list[np.ndarray | None] = [None] * self.n
        self._t1: list[float | None] = [None] * self.n
        self._t2: list[float | None] = [None] * self.n
        self._seen = np.zeros(self.n, dtype=int)
        self._penalty = np.zeros(self.n, dtype=int)
        self.rejections = np.zeros(self.n, dtype=int)
        self.by_kind: dict[str, int] = {}

    # -- the checks ---------------------------------------------------------
    def _check_one(self, i: int, p: np.ndarray, trace: float,
                   stamp_ms: float, age_ms: float) -> Violation | None:
        cfg = self.cfg

        if not (np.all(np.isfinite(p)) and math.isfinite(trace)):
            return Violation(i, "not_finite",
                             {"pos": [float(x) for x in p],
                              "cov_trace": float(trace)
                              if math.isfinite(trace) else None})

        if age_ms > cfg.max_age_ms:
            return Violation(i, "stale", {"age_ms": float(age_ms),
                                          "max_ms": cfg.max_age_ms})

        radius = float(np.linalg.norm(p))
        if radius > cfg.footprint_m:
            return Violation(i, "off_map", {"radius_m": radius,
                                            "max_m": cfg.footprint_m})

        p1, t1 = self._p1[i], self._t1[i]
        if p1 is not None and t1 is not None:
            dt = max((stamp_ms - t1) / 1000.0, 1e-3)
            step = float(np.linalg.norm(p - p1))
            budget = cfg.v_max_mps * dt + cfg.jump_margin_m
            if step > budget:
                return Violation(i, "teleport",
                                 {"step_m": step, "budget_m": budget,
                                  "dt_s": dt,
                                  "implied_v_mps": step / dt})

        if trace > cfg.max_cov_trace:
            return Violation(i, "cov_exceeded",
                             {"trace": float(trace),
                              "max": cfg.max_cov_trace})

        # -- the one that matters ------------------------------------------
        p2, t2 = self._p2[i], self._t2[i]
        if (p1 is not None and p2 is not None
                and t1 is not None and t2 is not None
                and self._seen[i] >= cfg.warmup_ticks):
            dt = max((stamp_ms - t1) / 1000.0, 1e-3)
            dt_prev = max((t1 - t2) / 1000.0, 1e-3)
            # Rescale the previous step to this interval before extrapolating,
            # or irregular tick spacing alone produces a false innovation.
            velocity = (p1 - p2) / dt_prev
            pred = p1 + velocity * dt
            innovation = float(np.linalg.norm(p - pred))

            sigma = math.sqrt(max(trace, 0.0) / 2.0)      # trace -> per-axis
            sigma_pred = math.sqrt(6.0) * sigma
            allowance = 0.5 * cfg.a_max_mps2 * dt * dt
            bound = allowance + cfg.sigma_ratio_max * sigma_pred

            if innovation > bound:
                return Violation(i, "sigma_lie", {
                    "innovation_m": innovation,
                    "bound_m": bound,
                    "reported_sigma_m": sigma,
                    "manoeuvre_allowance_m": allowance,
                    "ratio": innovation / max(sigma_pred, 1e-9),
                })
        return None

    # -- the tick -----------------------------------------------------------
    def check(self, pos, cov_trace, stamp_unix_ms: float,
              now_unix_ms: float | None = None) -> PlausibilityVerdict:
        """Judge one estimate snapshot. Returns the mask to hand `publish()`."""
        cfg = self.cfg
        p_all = np.asarray(pos, dtype=float).reshape(self.n, 2)
        c_all = np.asarray(cov_trace, dtype=float).reshape(self.n)
        now = float(stamp_unix_ms if now_unix_ms is None else now_unix_ms)
        age = now - float(stamp_unix_ms)

        trusted = np.ones(self.n, dtype=bool)
        held = np.zeros(self.n, dtype=bool)
        violations: list[Violation] = []

        for i in range(self.n):
            v = self._check_one(i, p_all[i], float(c_all[i]),
                                float(stamp_unix_ms), age)
            if v is not None:
                trusted[i] = False
                violations.append(v)
                self.rejections[i] += 1
                self.by_kind[v.kind] = self.by_kind.get(v.kind, 0) + 1
                self._penalty[i] = cfg.recover_ticks
                # `detail` is nested rather than splatted: its keys come from
                # whichever check fired and must never be able to collide with
                # this call's own arguments.
                self.audit.append("estimate_rejected", vehicle=i,
                                  violation=v.kind, detail=v.detail,
                                  stamp_unix_ms=int(stamp_unix_ms))
                # History is NOT updated on rejection — see the class docstring.
                continue

            if self._penalty[i] > 0:
                # Plausible, but still serving a penalty from an earlier trip.
                self._penalty[i] -= 1
                trusted[i] = False
                held[i] = True

            # Accepted as *history* either way: it passed every check, so it is
            # a sound baseline even while the vehicle is not yet trusted for
            # transmission. Otherwise the penalty window would itself destroy
            # the history the motion checks need to clear it.
            self._p2[i], self._t2[i] = self._p1[i], self._t1[i]
            self._p1[i], self._t1[i] = p_all[i].copy(), float(stamp_unix_ms)
            self._seen[i] += 1

        return PlausibilityVerdict(trusted=trusted, violations=violations,
                                   held=held)

    # -- reporting ----------------------------------------------------------
    def report(self) -> dict:
        return {
            "rejections_per_vehicle": [int(x) for x in self.rejections],
            "by_kind": dict(self.by_kind),
            "total": int(self.rejections.sum()),
        }
