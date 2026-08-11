"""H3.2 — the range mesh is the trust root now, so stop assuming it is honest.

`COMMS_HARDENING_PLAN.md` §4.4 makes a claim that is easy to enjoy and easy to
stop reading too early: *being GPS-denied means GPS spoofing has no purchase.*
True, and it is a real security win. But the sentence does not end there — the
plan's own corollary is that **the anchor and range mesh now carry the trust GPS
would have had.** Removing a trust root does not reduce the number of trust
roots. It moves it.

This file is the part that follows from that. Asset **A3**.

Standard UWB two-way ranging is not distance-authenticated. An attacker with a
compromised or impersonated tag can enlarge or reduce a reported distance, and
the factor graph will dutifully fold that into a position, at the tight sigma
its geometry deserves. The Huber kernels already on the range factors bound the
influence of a *few* bad measurements — that is what they are for — but Huber
is a robustness device against noise, not a detector against an adversary who
can choose which links to corrupt and by how much. It downweights outliers
silently and never tells anyone. Nothing in the loop currently *notices*.

Where 802.15.4z secure ranging (STS) is available in hardware, use it; it
authenticates the timestamp sequence and is a genuinely better answer than
anything here. This module is what you run when it is not available, which is
most of the time, and what you run *alongside* it when it is.

---------------------------------------------------------------------------
THE ASYMMETRY THAT MAKES THIS WORK
---------------------------------------------------------------------------

The single most useful fact about this problem is that the physics of range
error is **one-sided**, and the physics of range *attack* is not.

`sim/range_world.py`'s model — the Brain's day-7 numbers, and they are realistic
— has exactly two error sources beyond thermal noise, and both are positive:

    NLOS bias        exponential, mean 5 cm, ALWAYS LONGER (an obstructed path
                     is a longer path; radio does not take shortcuts)
    multipath spike  uniform +0.20 .. +0.50 m, also always longer

So a measurement that comes back **short** has no benign explanation. Thermal
noise at σ = 1.5 cm accounts for a couple of centimetres and nothing more. A
range reading 40 cm short is not a bad channel — it is a fabricated one.

And the direction matters tactically, not just statistically. Distance
*reduction* is the dangerous attack: it drags the estimated position toward the
attacker's chosen point, which is how you walk a vehicle off a standoff
perimeter or into terrain. Distance *enlargement* mostly degrades geometry and
inflates covariance, which the existing covariance gate already catches.

Hence every residual test here is **asymmetric**: a tight threshold on the short
side, a loose one on the long side. Symmetric gating would either drown in false
positives from ordinary NLOS or leave the short side wide open. Getting this
backwards — which a generic outlier filter does by construction — is worse than
not filtering at all.

---------------------------------------------------------------------------
WHAT IS CHECKED
---------------------------------------------------------------------------

  `rate`        A surveyed anchor does not move. The range to it therefore
                cannot change faster than the vehicle can fly. Catches the
                blunt attack — a link that steps metres between ticks — with no
                position estimate required, so it works from the first frame.

  `residual`    Against the last plausible position (already vetted by
                `plausibility.EstimateGate`), predict what each range should be
                and compare. Asymmetric, as above.

  `geometry`    With enough anchor links, solve for the position they imply and
                look at the leave-one-out residuals. A single corrupted link
                shows up as the one measurement everything else disagrees with,
                and is identified rather than merely detected. This is the only
                check that works at cold start with no prior estimate, which is
                also the moment an attacker most wants to be believed.

  `triangle`    Inter-agent ranges must satisfy the triangle inequality over
                every closed triple. Cheap, needs no positions at all, and
                catches fabricated meshes that are individually plausible but
                jointly impossible.

---------------------------------------------------------------------------
WHAT THIS DELIBERATELY DOES NOT DO
---------------------------------------------------------------------------

It does not *repair*. A suspicious link is dropped for that tick and recorded;
it is not replaced with an estimate of what it "should" have been. Substituting
a modelled value for a measurement is how a filter starts believing its own
predictions, and the whole project's failure history is about estimates that
became confident without becoming correct.

It also cannot defend against an attacker who corrupts a *majority* of a
vehicle's links consistently and coherently. With most measurements lying in
agreement, the minority that tells the truth is what looks like the outlier.
That is the standard limit of consistency-based detection and it is why §4.3
wants STS in hardware rather than statistics in software.

---------------------------------------------------------------------------
THE THREE-ANCHOR BLIND SPOT, MEASURED
---------------------------------------------------------------------------

`geometry` needs redundancy, and three anchor links do not have enough of it.
Measured against this module's own solver (`tests/test_range_integrity.py`
pins both numbers):

    3 anchors, +6 m corruption   -> rms 0.507 m   PASSES the gate, undetected
    3 anchors, +12 m corruption  -> rms 1.676 m   detected
    3 anchors, any corruption    -> every leave-one-out fit is rms 0.0000

That last row is the important one and it is not a tuning problem. Three ranges
constrain two unknowns, so removing any one leaves an exactly-determined system
that fits *perfectly* — every candidate link looks equally guilty, and naming
one would mean naming whichever the loop happened to visit first. Attribution
at three links is impossible in principle, not merely unreliable, which is why
`check` reports `geometry_inconsistent` against the vehicle and drops nothing.

Four links restore both detection and attribution. That is an independent
argument for the four-mast layout in `sim/range_world.py` — Domain 1 chose four
anchors for *observability* (the null-space ladder), and integrity turns out to
want the same number for a completely different reason.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np

from .audit_log import NullAuditLog


@dataclass
class RangeIntegrityConfig:
    """Thresholds. The asymmetry between `short` and `long` is the design."""

    # Vehicle kinematics, for the rate test.
    v_max_mps: float = 5.0
    rate_margin_m: float = 0.50

    # Residual gating, asymmetric on purpose (see the module docstring).
    # Short side: thermal noise only, so a few sigma is the whole budget.
    max_short_m: float = 0.15
    # Long side: must absorb NLOS (exp, mean 5 cm) plus multipath (up to
    # +0.50 m) without crying wolf on every obstructed link.
    max_long_m: float = 0.90

    # Geometry test.
    min_anchor_links: int = 3       # below this the solve is not overdetermined
    geometry_residual_m: float = 1.00
    # A leave-one-out fit must improve by at least this factor before a single
    # link is blamed. Without it, ordinary noise nominates a scapegoat every
    # tick.
    blame_improvement: float = 3.0

    # Triangle inequality slack — accumulated one-sided bias on three links.
    triangle_slack_m: float = 1.50

    # Below this many total links for a vehicle, consistency testing is not
    # meaningful and only the rate test applies.
    min_links_for_consistency: int = 3


@dataclass
class RangeViolation:
    kind: str
    vehicle: int
    peer: int                # anchor index, or the other vehicle
    is_anchor: bool
    measured_m: float
    expected_m: float | None = None
    detail: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {"kind": self.kind, "vehicle": int(self.vehicle),
                "peer": int(self.peer), "is_anchor": bool(self.is_anchor),
                "measured_m": float(self.measured_m),
                "expected_m": (None if self.expected_m is None
                               else float(self.expected_m)),
                "detail": self.detail}


@dataclass
class RangeVerdict:
    """Filtered measurements plus the reasons anything was dropped."""

    inter: list
    anchor: list
    violations: list[RangeViolation] = field(default_factory=list)

    @property
    def n_dropped(self) -> int:
        return len(self.violations)

    def kinds(self) -> list[str]:
        return [v.kind for v in self.violations]


def _multilaterate(anchor_pts: np.ndarray, ranges: np.ndarray,
                   guess: np.ndarray | None = None,
                   iters: int = 20) -> np.ndarray:
    """Least-squares position from anchor ranges. Local, small, dependency-free.

    Gauss-Newton on the range residuals with a Levenberg-style damping term so
    a degenerate geometry (all anchors nearly collinear with the vehicle) does
    not produce a singular normal matrix and a nonsense jump. Deliberately
    duplicated rather than imported from `sim/live_estimator.py`: `security/`
    must not depend on `sim/`, because the security layer has to be usable in
    a deployment where the range simulator does not exist at all.
    """
    p = (np.zeros(2) if guess is None else np.asarray(guess, float).copy())
    for _ in range(iters):
        d = p[None, :] - anchor_pts
        dist = np.linalg.norm(d, axis=1)
        dist = np.maximum(dist, 1e-6)
        r = dist - ranges
        J = d / dist[:, None]
        H = J.T @ J + 1e-6 * np.eye(2)
        try:
            step = np.linalg.solve(H, J.T @ r)
        except np.linalg.LinAlgError:
            break
        p = p - step
        if np.linalg.norm(step) < 1e-9:
            break
    return p


def _rms(anchor_pts: np.ndarray, ranges: np.ndarray, p: np.ndarray) -> float:
    pred = np.linalg.norm(p[None, :] - anchor_pts, axis=1)
    return float(np.sqrt(np.mean((pred - ranges) ** 2)))


class RangeMonitor:
    """Per-tick plausibility of the range mesh, with per-link attribution."""

    def __init__(self, n: int, anchors, cfg: RangeIntegrityConfig | None = None,
                 audit=None) -> None:
        self.n = int(n)
        self.anchors = np.asarray(anchors, dtype=float).reshape(-1, 2)
        self.cfg = cfg or RangeIntegrityConfig()
        self.audit = audit if audit is not None else NullAuditLog()

        self._last_anchor: dict[tuple[int, int], float] = {}
        self._last_inter: dict[tuple[int, int], float] = {}
        self.dropped = 0
        self.by_kind: dict[str, int] = {}

    # -- helpers ------------------------------------------------------------
    def _note(self, v: RangeViolation, out: list[RangeViolation]) -> None:
        out.append(v)
        self.dropped += 1
        self.by_kind[v.kind] = self.by_kind.get(v.kind, 0) + 1
        self.audit.append("range_rejected", link=v.to_json())

    def _asymmetric_bad(self, measured: float, expected: float) -> str | None:
        """None if the error is explainable, else which side it failed on."""
        err = measured - expected
        if err < -self.cfg.max_short_m:
            return "short"
        if err > self.cfg.max_long_m:
            return "long"
        return None

    # -- the tick -----------------------------------------------------------
    def check(self, frame, ref_pos=None, dt_s: float = 0.1) -> RangeVerdict:
        """Filter one `RangeFrame`.

        `frame` needs only `.inter` and `.anchor` (lists of `(i, j, metres)`
        and `(i, anchor_k, metres)`), so any producer works — the simulator
        today, a real UWB driver later.

        `ref_pos` is the last position estimate the plausibility gate accepted,
        or None at cold start. It is used only as a *reference*, never fused,
        and a None simply disables the residual test while leaving rate,
        geometry and triangle active.
        """
        cfg = self.cfg
        violations: list[RangeViolation] = []
        ref = None if ref_pos is None else np.asarray(ref_pos,
                                                      dtype=float).reshape(-1, 2)

        max_step = cfg.v_max_mps * max(dt_s, 1e-3) + cfg.rate_margin_m

        # -- anchor links ---------------------------------------------------
        anchor_ok: list = []
        blamed_anchor: set[tuple[int, int]] = set()

        # geometry first, so a link blamed by the overdetermined solve is
        # removed before the cheaper per-link tests report it a second time.
        per_vehicle: dict[int, list] = {}
        for rec in frame.anchor:
            per_vehicle.setdefault(int(rec[0]), []).append(rec)

        for i, recs in per_vehicle.items():
            if len(recs) < max(cfg.min_anchor_links, cfg.min_links_for_consistency):
                continue
            pts = np.array([self.anchors[int(k)] for _, k, _ in recs])
            rng = np.array([float(m) for _, _, m in recs])
            guess = None if ref is None or i >= len(ref) else ref[i]
            p_all = _multilaterate(pts, rng, guess)
            rms_all = _rms(pts, rng, p_all)
            if rms_all <= cfg.geometry_residual_m:
                continue
            if len(recs) < 4:
                # Detected but not attributable: with 3 links the solve has no
                # redundancy left after removing one, so blaming a specific
                # measurement would be a coin flip. Report the inconsistency
                # against the vehicle and drop nothing.
                self._note(RangeViolation(
                    "geometry_inconsistent", i, -1, True, float(rms_all),
                    None, {"rms_m": rms_all, "n_links": len(recs),
                           "note": "not attributable below 4 links"}),
                    violations)
                continue
            best_idx, best_rms = None, rms_all
            for drop in range(len(recs)):
                keep = [x for x in range(len(recs)) if x != drop]
                p_k = _multilaterate(pts[keep], rng[keep], guess)
                rms_k = _rms(pts[keep], rng[keep], p_k)
                if rms_k < best_rms:
                    best_idx, best_rms = drop, rms_k
            if (best_idx is not None
                    and best_rms * cfg.blame_improvement < rms_all):
                _, k, m = recs[best_idx]
                blamed_anchor.add((i, int(k)))
                self._note(RangeViolation(
                    "geometry", i, int(k), True, float(m),
                    float(np.linalg.norm(
                        _multilaterate(
                            pts[[x for x in range(len(recs)) if x != best_idx]],
                            rng[[x for x in range(len(recs)) if x != best_idx]],
                            guess) - self.anchors[int(k)])),
                    {"rms_with_m": rms_all, "rms_without_m": best_rms}),
                    violations)

        for rec in frame.anchor:
            i, k, m = int(rec[0]), int(rec[1]), float(rec[2])
            if (i, k) in blamed_anchor:
                continue
            key = (i, k)

            prev = self._last_anchor.get(key)
            if prev is not None and abs(m - prev) > max_step:
                self._note(RangeViolation(
                    "rate", i, k, True, m, prev,
                    {"delta_m": m - prev, "max_step_m": max_step}),
                    violations)
                continue

            if ref is not None and i < len(ref):
                expected = float(np.linalg.norm(ref[i] - self.anchors[k]))
                side = self._asymmetric_bad(m, expected)
                if side is not None:
                    self._note(RangeViolation(
                        "residual", i, k, True, m, expected,
                        {"side": side, "error_m": m - expected,
                         "limit_m": (cfg.max_short_m if side == "short"
                                     else cfg.max_long_m)}),
                        violations)
                    continue

            self._last_anchor[key] = m
            anchor_ok.append(rec)

        # -- inter-agent links ----------------------------------------------
        inter_ok: list = []
        measured: dict[tuple[int, int], float] = {}
        for rec in frame.inter:
            i, j, m = int(rec[0]), int(rec[1]), float(rec[2])
            measured[(min(i, j), max(i, j))] = m

        bad_triples: set[tuple[int, int]] = set()
        keys = sorted(measured)
        nodes = sorted({x for pair in keys for x in pair})
        for a, b, c in itertools.combinations(nodes, 3):
            ab = measured.get((a, b))
            ac = measured.get((a, c))
            bc = measured.get((b, c))
            if ab is None or ac is None or bc is None:
                continue
            longest, s1, s2, pair = max(
                ((ab, ac, bc, (a, b)), (ac, ab, bc, (a, c)),
                 (bc, ab, ac, (b, c))), key=lambda t: t[0])
            if longest > s1 + s2 + cfg.triangle_slack_m:
                bad_triples.add(pair)
                self._note(RangeViolation(
                    "triangle", pair[0], pair[1], False, float(longest),
                    float(s1 + s2),
                    {"triple": [a, b, c], "excess_m":
                     longest - (s1 + s2)}), violations)

        for rec in frame.inter:
            i, j, m = int(rec[0]), int(rec[1]), float(rec[2])
            key = (min(i, j), max(i, j))
            if key in bad_triples:
                continue

            prev = self._last_inter.get(key)
            # Two moving vehicles, so the closing speed budget is doubled.
            if prev is not None and abs(m - prev) > 2.0 * max_step:
                self._note(RangeViolation(
                    "rate", i, j, False, m, prev,
                    {"delta_m": m - prev, "max_step_m": 2.0 * max_step}),
                    violations)
                continue

            if ref is not None and i < len(ref) and j < len(ref):
                expected = float(np.linalg.norm(ref[i] - ref[j]))
                side = self._asymmetric_bad(m, expected)
                if side is not None:
                    self._note(RangeViolation(
                        "residual", i, j, False, m, expected,
                        {"side": side, "error_m": m - expected,
                         "limit_m": (cfg.max_short_m if side == "short"
                                     else cfg.max_long_m)}),
                        violations)
                    continue

            self._last_inter[key] = m
            inter_ok.append(rec)

        return RangeVerdict(inter=inter_ok, anchor=anchor_ok,
                            violations=violations)

    # -- reporting ----------------------------------------------------------
    def report(self) -> dict:
        return {"dropped": int(self.dropped), "by_kind": dict(self.by_kind)}
