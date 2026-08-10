"""D2.1 — coverage-aware bidding, distributed election, and the adjacency cut.

One task, N coalition slots. Every agent scores itself, then the swarm agrees on
the top N *without a central computer* — neighbour-to-neighbour only, which is
the same constraint the Brain's day-3 distributed solver worked under.

THE MERGE IS A LATTICE JOIN, WHICH IS WHY IT CONVERGES.
Each agent holds a length-N list of the best (bid, agent) pairs it has heard of.
A round is: merge every neighbour's list into mine, keep the top N. That
operation is idempotent, commutative and associative, so the local views form a
join-semilattice and each round propagates every fact one hop further. After
diam(G) rounds every agent has heard every fact, so all views equal the global
join — the centralised top N. No oscillation is possible, and the bound is
structural rather than empirical. `tests/test_cbba.py` checks it on random
graphs anyway, because a proof about the algorithm is not a proof about the
implementation.

THE COVERAGE PENALTY IS THE INTERESTING TERM.
`w_r * kappa_loiter_i` is what stops the auction from stripping the loiter mesh.
A close, healthy, well-sensored agent is a great candidate right up until you
notice it is the only thing covering the north approach. kappa measures exactly
that: how much worse the residual set's coverage gets if this agent leaves.
The same locational cost drives the Lloyd re-solve during RECONFIG, so the
auction and the reconfiguration agree about what "coverage" means instead of
optimising two different things.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np


# --------------------------------------------------------------------------
# Comms graph
# --------------------------------------------------------------------------
def comms_adjacency(pos, r_comm: float) -> np.ndarray:
    """A_ij(t) = 1[‖p_i − p_j‖ <= R_comm], no self-loops."""
    p = np.asarray(pos, dtype=float)
    d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=2)
    a = (d <= r_comm).astype(int)
    np.fill_diagonal(a, 0)
    return a


def neighbors(adj, i) -> list[int]:
    return np.flatnonzero(adj[i]).tolist()


def is_connected(adj) -> bool:
    n = len(adj)
    if n == 0:
        return True
    seen = {0}
    q = deque([0])
    while q:
        for j in neighbors(adj, q.popleft()):
            if j not in seen:
                seen.add(j)
                q.append(j)
    return len(seen) == n


def diameter(adj) -> int:
    """Longest shortest-path in hops. inf if disconnected."""
    n = len(adj)
    best = 0
    for s in range(n):
        dist = {s: 0}
        q = deque([s])
        while q:
            u = q.popleft()
            for v in neighbors(adj, u):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
        if len(dist) < n:
            return np.inf
        best = max(best, max(dist.values()))
    return best


def laplacian(adj) -> np.ndarray:
    a = np.asarray(adj, dtype=float)
    return np.diag(a.sum(axis=1)) - a


# --------------------------------------------------------------------------
# Coverage — shared by the bid penalty and the RECONFIG re-solve
# --------------------------------------------------------------------------
def locational_cost(pos, region) -> float:
    """Standard Lloyd/locational cost: mean squared distance from each sample
    of the region to its nearest agent. Lower is better coverage.

    `region` is an (S, 2) sample of the area the loiter mesh is meant to watch.
    Sampling rather than integrating keeps this cheap enough to call inside the
    bid loop, and the same samples drive the Lloyd step in mission_fsm.py so the
    two stages cannot disagree about what coverage means.
    """
    p = np.asarray(pos, dtype=float)
    r = np.asarray(region, dtype=float)
    if len(p) == 0:
        return float("inf")
    d2 = ((r[:, None, :] - p[None, :, :]) ** 2).sum(axis=2)
    return float(d2.min(axis=1).mean())


def loiter_criticality(pos, region) -> np.ndarray:
    """kappa_i: fractional coverage degradation if agent i leaves the mesh.

    kappa_i = (cost(without i) − cost(all)) / cost(all)

    0 means redundant — someone else already covers what it covers. Large means
    load-bearing: pull this one and a piece of the area goes dark.
    """
    p = np.asarray(pos, dtype=float)
    base = locational_cost(p, region)
    if not np.isfinite(base) or base <= 0 or len(p) <= 1:
        return np.zeros(len(p))
    out = np.empty(len(p))
    for i in range(len(p)):
        out[i] = (locational_cost(np.delete(p, i, axis=0), region) - base) / base
    return out


# --------------------------------------------------------------------------
# Bidding
# --------------------------------------------------------------------------
@dataclass
class BidWeights:
    """The plan's c_i(X) weights. Distance dominates by design — an agent that
    is already close spends less battery and arrives sooner — but the coverage
    penalty is weighted heavily enough to actually veto a load-bearing agent
    rather than merely nudging the ordering."""

    w_d: float = 1.0        # proximity to the target
    w_b: float = 0.35       # battery
    w_s: float = 0.30       # sensor fitness for the task
    w_r: float = 0.80       # loiter-coverage penalty


@dataclass
class SwarmState:
    pos: np.ndarray
    battery: np.ndarray                     # [0, 1]
    sensor: np.ndarray                      # [0, 1] task fitness
    region: np.ndarray                      # (S, 2) coverage samples
    cov_trace: np.ndarray | None = None     # estimator trust, from Domain 1
    max_cov_trace: float = 0.004

    def __post_init__(self):
        self.pos = np.asarray(self.pos, dtype=float)
        self.battery = np.asarray(self.battery, dtype=float)
        self.sensor = np.asarray(self.sensor, dtype=float)
        self.region = np.asarray(self.region, dtype=float)
        if self.cov_trace is not None:
            self.cov_trace = np.asarray(self.cov_trace, dtype=float)
        n = len(self.pos)
        if not (len(self.battery) == len(self.sensor) == n):
            raise ValueError("pos, battery and sensor must be the same length")


def bid_scores(state: SwarmState, target, w: BidWeights | None = None
               ) -> np.ndarray:
    """c_i(X) for every agent.

    Agents the estimator does not trust bid -inf. This is Domain 1 reaching
    forward into Domain 2: an agent whose covariance would trip the supervisor's
    trust gate must never win the auction, because the plan naming it would be
    rejected on arrival — and the FSM's whole promise is that rejection never
    happens.
    """
    w = w or BidWeights()
    x = np.asarray(target, dtype=float)
    dist = np.linalg.norm(state.pos - x, axis=1)
    kappa = loiter_criticality(state.pos, state.region)

    c = (w.w_d / (1.0 + dist)
         + w.w_b * state.battery
         + w.w_s * state.sensor
         - w.w_r * kappa)

    if state.cov_trace is not None:
        c = np.where(state.cov_trace > state.max_cov_trace, -np.inf, c)
    return c


# --------------------------------------------------------------------------
# Distributed election by max-consensus
# --------------------------------------------------------------------------
def _merge(view_a, view_b, n_slots):
    """Join two winner lists: union, sort by (bid desc, id asc), keep N.

    Tie-break by lowest id, exactly as the plan specifies, so the result is a
    total order and two agents can never settle on different sets with equal
    bids.
    """
    best = {}
    for bid, agent in list(view_a) + list(view_b):
        if agent not in best or bid > best[agent]:
            best[agent] = bid
    ranked = sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))
    return [(bid, agent) for agent, bid in ranked[:n_slots]]


@dataclass
class ElectionResult:
    winners: list[int]
    views: list[list[tuple[float, int]]]
    rounds: int
    converged: bool
    diameter: int | float
    history: list[int] = field(default_factory=list)   # disagreeing agents/round

    def agreed(self) -> bool:
        first = [a for _, a in self.views[0]]
        return all([a for _, a in v] == first for v in self.views)


def consensus_elect(scores, adj, n_slots: int, max_rounds: int | None = None
                    ) -> ElectionResult:
    """Elect the top `n_slots` bidders using neighbour-only communication.

    Returns the round at which every agent's view stopped changing. That is the
    number the plan's `<= diam(G)` claim is about, and the test asserts it.
    """
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    n_slots = min(n_slots, n)
    dia = diameter(adj)
    cap = max_rounds if max_rounds is not None else (
        n + 1 if not np.isfinite(dia) else int(dia) + 2)

    # every agent starts knowing only itself
    views = [[(float(scores[i]), i)] for i in range(n)]
    history = []
    rounds = 0

    for r in range(1, cap + 1):
        new = []
        for i in range(n):
            v = views[i]
            for j in neighbors(adj, i):
                v = _merge(v, views[j], n_slots)
            new.append(v)
        changed = sum(1 for i in range(n) if new[i] != views[i])
        history.append(changed)
        views = new
        if changed == 0:
            rounds = r - 1          # the last round that actually did anything
            break
        rounds = r

    winners = [a for _, a in views[0]]
    return ElectionResult(winners=winners, views=views, rounds=rounds,
                          converged=all(h == 0 for h in history[-1:]),
                          diameter=dia, history=history)


def centralized_top_n(scores, n_slots: int) -> list[int]:
    """What an omniscient auctioneer would pick. The consensus must match it."""
    scores = np.asarray(scores, dtype=float)
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    return order[:min(n_slots, len(scores))]


# --------------------------------------------------------------------------
# 2.1 Adjacency cut — block-diagonalise the CONTROL graph only
# --------------------------------------------------------------------------
@dataclass
class ControlPartition:
    perm: np.ndarray            # P: the permutation putting S first
    control_adj: np.ndarray     # block-diagonal, in the ORIGINAL agent order
    physical_adj: np.ndarray    # unchanged — the radios do not care about roles
    active: list[int]
    loiter: list[int]

    def blocks(self) -> tuple[np.ndarray, np.ndarray]:
        """(A_sub1, A_sub2) in permuted order — the two Laplacians' sources."""
        a = self.control_adj[np.ix_(self.perm, self.perm)]
        k = len(self.active)
        return a[:k, :k], a[k:, k:]

    def laplacians(self) -> tuple[np.ndarray, np.ndarray]:
        a1, a2 = self.blocks()
        return laplacian(a1), laplacian(a2)


def partition_control_graph(adj, active) -> ControlPartition:
    """Cut the control graph into active sub-team and residual loiter set.

    Physical links persist across the cut — the radios still hear each other and
    the estimator still uses every range it can get, which matters because
    Domain 1's information matrix wants every measurement it can have. Only the
    *control* graph is block-diagonalised, so the two sets coordinate their
    motion independently and neither can drag the other around.
    """
    a = np.asarray(adj, dtype=int)
    n = len(a)
    active = sorted(set(int(i) for i in active))
    if active and (active[0] < 0 or active[-1] >= n):
        raise ValueError("active set contains an unknown agent")
    loiter = [i for i in range(n) if i not in set(active)]

    mask = np.zeros((n, n), dtype=int)
    for group in (active, loiter):
        if group:
            idx = np.array(group)
            mask[np.ix_(idx, idx)] = 1

    return ControlPartition(
        perm=np.array(active + loiter, dtype=int),
        control_adj=a * mask,
        physical_adj=a,
        active=active,
        loiter=loiter,
    )
