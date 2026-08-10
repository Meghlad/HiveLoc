"""D4.2 — the simulated UWB radio: true positions in, noisy ranges out.

This is the ONE simulated component in the Domain 4 loop, and it is simulated
for the same reason SITL fakes a GPS: there is no radio on the desk. Everything
else in the loop is real — a real autopilot, a real EKF3, real rotor dynamics,
a real gate. Swapping this module for a driver that reads a DW1000 is a
hardware step, not an architecture change, which is exactly why the interface it
presents (a per-tick bundle of `(i, j, metres)` tuples) is the interface a real
UWB stack presents and not something more convenient.

REUSED, NOT REINVENTED. The noise model is the Brain's, from
`src/estimation/day7_realistic_robust.py` and `src/flight/close_the_loop.py`:

    sigma_uwb    0.015 m   line-of-sight thermal noise
    p_nlos       0.15      chance the path is obstructed
    nlos_scale   0.05 m    mean of the NLOS bias, drawn from an exponential
    p_outlier    0.03      multipath spike, uniform 0.2-0.5 m
    p_dropout    0.10      link simply not reported this tick

Two properties of that model matter downstream and neither is arbitrary. NLOS
error is drawn from an EXPONENTIAL and added, never subtracted: an obstructed
radio path is longer than the straight line, never shorter, so the error is
one-sided. A symmetric noise model would let the estimator average NLOS away,
which real UWB does not permit and which would make the Huber kernel in
`live_estimator` look like decoration instead of load-bearing. And dropout is
per-link per-tick, so the graph's connectivity changes every tick — that is the
condition the safety prior in the estimator exists to survive.

ANCHOR RANGE IS NOT INTER-AGENT RANGE. `r_anchor` is deliberately much larger
than `r_comm`. A ground anchor is mains-powered infrastructure on a mast with a
real antenna; a drone-borne tag is battery-powered and shadowed by its own
airframe. Giving the anchor the drones' range budget would understate the
anchored configuration for no physical reason. (The plan states this as
verified; it is a modelling choice, and it is the reason S2's "stay inside the
footprint" constraint is a footprint at all.)

NaN IS A DROPPED LINK, NOT A ZERO. A vehicle the bridge has not heard from has
NaN truth. Every pair involving it is skipped. Coercing NaN to 0 would put a
phantom vehicle on the anchor and feed the estimator ranges to it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class UwbModel:
    """The radio. Defaults are the Brain's day-7 numbers, unchanged."""

    sigma_m: float = 0.015
    p_nlos: float = 0.15
    nlos_scale_m: float = 0.05
    p_outlier: float = 0.03
    outlier_lo_m: float = 0.20
    outlier_hi_m: float = 0.50
    p_dropout: float = 0.10

    def sample(self, true_d: float, rng: np.random.Generator) -> float | None:
        """One range reading, or None if the link dropped this tick."""
        if rng.random() < self.p_dropout:
            return None
        err = rng.normal(0.0, self.sigma_m)
        if rng.random() < self.p_nlos:
            err += rng.exponential(self.nlos_scale_m)          # always LONGER
        if rng.random() < self.p_outlier:
            err += rng.uniform(self.outlier_lo_m, self.outlier_hi_m)
        return float(true_d + err)


@dataclass
class RangeConfig:
    r_comm: float = 30.0            # inter-agent ranging radius (m)
    r_anchor: float = 120.0         # anchor ranging radius (m) — see docstring
    uwb: UwbModel = field(default_factory=UwbModel)
    anchor_uwb: UwbModel | None = None   # None: anchors use the same radio

    def anchor_model(self) -> UwbModel:
        return self.anchor_uwb if self.anchor_uwb is not None else self.uwb


# A four-anchor square around the operating area, surveyed, in TacFrame metres.
# Four rather than one: §1.3's ladder says range-only to a SINGLE anchor pins the
# swarm radially and leaves it tangentially free. Four anchors at known points
# are four independent radial constraints in different directions, which is the
# "2 surveyed anchors, range only -> dim ker(H) = 0" row of that table with
# margin to spare. It is also what a real site looks like: masts at the corners.
DEFAULT_ANCHORS = np.array([
    [-25.0, -25.0],
    [+25.0, -25.0],
    [+25.0, +25.0],
    [-25.0, +25.0],
], dtype=float)


@dataclass
class RangeFrame:
    """One tick of measurements. Field names match `hive.anchored_isam2`'s
    frame dicts so the offline and live estimators consume the same shape."""

    inter: list[tuple[int, int, float]] = field(default_factory=list)
    anchor: list[tuple[int, int, float]] = field(default_factory=list)
    n_dropped: int = 0
    n_possible: int = 0

    @property
    def link_yield(self) -> float:
        """Fraction of geometrically-possible links that actually reported.

        Worth watching. A yield that falls while the swarm is stationary is a
        model artefact; one that falls as the swarm spreads out is the comms
        radius biting, and that is the number S2's footprint argument lives on.
        """
        if self.n_possible == 0:
            return 0.0
        return 1.0 - self.n_dropped / self.n_possible

    def degree(self, n: int) -> np.ndarray:
        """Per-vehicle count of reporting links, anchors included.

        A vehicle at degree 0 is unobservable this tick and will be carried by
        the motion prior alone; the estimator's covariance should say so, and
        `test_stationary_estimate` checks that it does.
        """
        d = np.zeros(n, dtype=int)
        for i, j, _ in self.inter:
            d[i] += 1
            d[j] += 1
        for i, _, _ in self.anchor:
            d[i] += 1
        return d


class RangeWorld:
    """Pairwise + anchor ranges over N vehicles, given their true positions.

    Deterministic for a given seed and a given sequence of truth arrays, which
    is what makes `tests/test_stationary_estimate.py` a regression rather than a
    coin flip.
    """

    def __init__(self, anchors=None, cfg: RangeConfig | None = None,
                 seed: int = 7) -> None:
        self.anchors = (DEFAULT_ANCHORS.copy() if anchors is None
                        else np.asarray(anchors, dtype=float).reshape(-1, 2))
        self.cfg = cfg or RangeConfig()
        self.rng = np.random.default_rng(seed)

    @property
    def n_anchors(self) -> int:
        return len(self.anchors)

    def measure(self, truth) -> RangeFrame:
        """Ranges for one tick. `truth` is (N, 2) TacFrame metres.

        The name is deliberate: this is `measure()` from the Brain's day5/day6,
        with the day-7 noise model and an anchor radius of its own. Keeping the
        name keeps the lineage legible.
        """
        p = np.asarray(truth, dtype=float).reshape(-1, 2)
        n = len(p)
        live = np.all(np.isfinite(p), axis=1)
        frame = RangeFrame()
        model = self.cfg.uwb
        anchor_model = self.cfg.anchor_model()

        for i in range(n):
            if not live[i]:
                continue
            for j in range(i + 1, n):
                if not live[j]:
                    continue
                d = float(np.linalg.norm(p[i] - p[j]))
                if d > self.cfg.r_comm:
                    continue
                frame.n_possible += 1
                m = model.sample(d, self.rng)
                if m is None:
                    frame.n_dropped += 1
                else:
                    frame.inter.append((i, j, m))

            for k, a in enumerate(self.anchors):
                d = float(np.linalg.norm(p[i] - a))
                if d > self.cfg.r_anchor:
                    continue
                frame.n_possible += 1
                m = anchor_model.sample(d, self.rng)
                if m is None:
                    frame.n_dropped += 1
                else:
                    frame.anchor.append((i, k, m))

        return frame

    # -- diagnostics ------------------------------------------------------
    def anchor_reach(self, truth) -> np.ndarray:
        """Distance from each vehicle to its NEAREST anchor.

        S2's constraint in one number: bulk translation of the formation is
        observable only through changing anchor ranges, so a vehicle whose
        nearest anchor is beyond `r_anchor` has left the footprint and its
        absolute position is no longer pinned by anything. The orchestrator
        checks this before commanding a translation.
        """
        p = np.asarray(truth, dtype=float).reshape(-1, 2)
        d = np.linalg.norm(p[:, None, :] - self.anchors[None, :, :], axis=2)
        return d.min(axis=1)

    def inside_footprint(self, truth) -> np.ndarray:
        return self.anchor_reach(truth) <= self.cfg.r_anchor


def main() -> None:
    """Offline sanity print — no SITL, no sockets."""
    rng = np.random.default_rng(0)
    n = 4
    truth = np.array([[-4.0, -4.0], [4.0, -4.0], [4.0, 4.0], [-4.0, 4.0]])
    world = RangeWorld(seed=3)

    print(f"D4.2 range world — {n} vehicles, {world.n_anchors} anchors, "
          f"r_comm {world.cfg.r_comm} m, r_anchor {world.cfg.r_anchor} m")
    print("=" * 78)

    yields, errs = [], []
    for t in range(200):
        jitter = truth + rng.normal(0, 0.02, truth.shape)
        f = world.measure(jitter)
        yields.append(f.link_yield)
        for i, j, m in f.inter:
            errs.append(m - float(np.linalg.norm(jitter[i] - jitter[j])))
        for i, k, m in f.anchor:
            errs.append(m - float(np.linalg.norm(jitter[i] - world.anchors[k])))

    e = np.array(errs)
    print(f"  links per tick        {f.n_possible}")
    print(f"  mean link yield       {np.mean(yields):.3f} "
          f"(dropout {world.cfg.uwb.p_dropout:.2f})")
    print(f"  range error mean      {e.mean() * 100:+.2f} cm   "
          f"<- POSITIVE by construction: NLOS is one-sided")
    print(f"  range error median    {np.median(e) * 100:+.2f} cm")
    print(f"  range error sigma     {e.std() * 100:.2f} cm  "
          f"(LOS sigma {world.cfg.uwb.sigma_m * 100:.1f} cm)")
    print(f"  |error| > 10 cm       {(np.abs(e) > 0.10).mean() * 100:.1f} % "
          f"<- what the Huber kernel is for")
    print(f"  anchor reach          {world.anchor_reach(truth).max():.1f} m "
          f"(footprint {world.cfg.r_anchor:.0f} m)")
    print("=" * 78)


if __name__ == "__main__":
    main()
