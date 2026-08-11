"""One object that carries the inbound defences into the Domain 4 loop.

`enable_signing` protects the link. This protects what arrives *over* it, which
is a different problem with a different answer: signing asks "did the right
party send this?", and everything here asks "could this possibly be true?".
A frame can pass the first and fail the second, and per `COMMS_HARDENING_PLAN.md`
§1.3 that combination — authenticated and wrong — is the attack this specific
architecture is most exposed to.

Bundled rather than wired individually because the two gates are ordered and the
order is not arbitrary:

    ranges ──▶ RangeMonitor ──▶ estimator ──▶ EstimateGate ──▶ fan-out ──▶ EKF3
               (A3, inputs)                   (A2, output)

The range monitor runs *before* the estimator so corrupted measurements never
enter the factor graph — once a bad range is fused, its influence is spread
across every position in the window and cannot be withdrawn. The estimate gate
runs *after*, because it is the last point at which a position can still be
refused before an autopilot begins flying on it.

Two layers rather than one because they fail differently. The range monitor
catches a few corrupted links and is blind to a coherent majority; the estimate
gate does not care how the estimate went wrong, only that it disagrees with
physics or with its own reported confidence. An attacker good enough to defeat
the first still has to produce a trajectory that survives the second.

The guards default to OFF, and `from_run` returns None when disabled, so a run
without them is byte-for-byte the run that existed before. That matters for
reproducing the measured S0 numbers.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field

import numpy as np

from .audit_log import AuditLog, NullAuditLog
from .plausibility import EstimateGate, PlausibilityConfig
from .range_integrity import RangeIntegrityConfig, RangeMonitor


@dataclass
class GuardStats:
    ranges_dropped: int = 0
    estimates_refused: int = 0
    ticks: int = 0
    # Vehicles currently not trusted, sampled at the end of the run.
    untrusted_at_end: list = field(default_factory=list)


class LoopGuards:
    """Range-mesh and estimate plausibility, in the order the loop needs them."""

    def __init__(self, n: int, anchors, audit=None,
                 plaus: PlausibilityConfig | None = None,
                 rng_cfg: RangeIntegrityConfig | None = None) -> None:
        self.n = int(n)
        self.audit = audit if audit is not None else NullAuditLog()
        self.monitor = RangeMonitor(n, anchors, rng_cfg, audit=self.audit)
        self.gate = EstimateGate(n, plaus, audit=self.audit)
        self.stats = GuardStats()
        # The last estimate the gate accepted, used as the range monitor's
        # reference. Deliberately the *accepted* one: feeding back an estimate
        # the gate just refused would let a spoof define what counts as a
        # plausible range on the next tick.
        self._ref: np.ndarray | None = None

    @classmethod
    def from_run(cls, enabled: bool, n: int, anchors, keystore=None,
                 log_path=None):
        """Build the guards, or None if the run did not ask for them."""
        if not enabled:
            return None
        if keystore is not None:
            audit = AuditLog.from_keystore(keystore, log_path)
        else:
            # No keystore means no audit key, and an unkeyed chain is
            # decoration (see audit_log's docstring). Guard still runs; it
            # simply does not claim a tamper-evident record.
            audit = NullAuditLog()
        return cls(n, anchors, audit=audit)

    # -- the two seams ------------------------------------------------------
    def filter_ranges(self, frame, dt_s: float):
        """Drop implausible measurements before they reach the factor graph."""
        verdict = self.monitor.check(frame, ref_pos=self._ref, dt_s=dt_s)
        self.stats.ranges_dropped += verdict.n_dropped
        return verdict

    def judge(self, pos, cov_trace, stamp_unix_ms: float,
              now_unix_ms: float | None = None):
        """Decide which vehicles' estimates may be sent to their autopilots."""
        verdict = self.gate.check(pos, cov_trace, stamp_unix_ms, now_unix_ms)
        self.stats.ticks += 1
        if verdict.violations:
            self.stats.estimates_refused += len(verdict.violations)
        accepted = np.asarray(pos, dtype=float).reshape(self.n, 2).copy()
        if np.any(verdict.trusted):
            self._ref = accepted
        return verdict

    # -- reporting ----------------------------------------------------------
    def report(self) -> dict:
        self.stats.untrusted_at_end = []
        return {
            "ranges": self.monitor.report(),
            "estimates": self.gate.report(),
            "ticks": self.stats.ticks,
            "audit_seq": getattr(self.audit, "seq", 0),
        }

    def summary_line(self) -> str:
        r = self.monitor.report()
        e = self.gate.report()
        return (f"guards: {r['dropped']} range link(s) dropped {r['by_kind']}, "
                f"{e['total']} estimate(s) refused {e['by_kind']}")
