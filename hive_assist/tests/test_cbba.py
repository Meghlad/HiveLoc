"""D2.1 acceptance: the distributed election agrees with an omniscient one.

The plan's claim is "converges in <= diam(G) rounds to a conflict-free set". That
is three separate assertions and they are tested separately:

  conflict-free   every agent ends with the SAME winner list
  correct         that list is the centralised top N
  bounded         it took no more than diam(G) rounds to get there

Run on random geometric graphs, because a consensus algorithm that only works on
a complete graph has not been tested at all.
"""

import numpy as np
import pytest

from hive.cbba import (
    BidWeights,
    ElectionResult,
    SwarmState,
    bid_scores,
    centralized_top_n,
    comms_adjacency,
    consensus_elect,
    diameter,
    is_connected,
    laplacian,
    locational_cost,
    loiter_criticality,
    partition_control_graph,
)


def ring_region(radius=8.0, n_ang=180, n_rad=5, seed=0):
    """The annulus the loiter mesh is meant to cover, sampled on a DETERMINISTIC
    polar grid.

    Random sampling was the first thing tried and it was a mistake: with a few
    hundred random samples the per-sector counts are Poisson, so kappa picked up
    ±20% of pure sampling noise and agents that are symmetric by construction
    got visibly different criticalities. A fixed grid makes kappa a property of
    the geometry instead of the seed — which matters in the real system too, not
    just in the tests, since a bid that wobbles with resampling is a bid that
    can flip an election for no reason.
    """
    a = np.linspace(0, 2 * np.pi, n_ang, endpoint=False)
    r = radius * np.sqrt(np.linspace(0.55, 1.0, n_rad))
    aa, rr = np.meshgrid(a, r)
    return np.stack([(rr * np.cos(aa)).ravel(), (rr * np.sin(aa)).ravel()],
                    axis=1)


def ring_swarm(m=10, radius=8.0, seed=1):
    rng = np.random.default_rng(seed)
    a = 2 * np.pi * np.arange(m) / m
    pos = np.stack([radius * np.cos(a), radius * np.sin(a)], axis=1)
    return SwarmState(
        pos=pos,
        battery=rng.uniform(0.6, 1.0, m),
        sensor=rng.uniform(0.3, 1.0, m),
        region=ring_region(radius, seed=seed),
    )


def random_connected_graph(rng, n, lo=0.35, hi=1.4):
    """Random geometric graph, retried until connected."""
    for _ in range(400):
        pos = rng.uniform(0, 1, (n, 2))
        r = rng.uniform(lo, hi)
        adj = comms_adjacency(pos, r)
        if is_connected(adj):
            return pos, adj
    raise RuntimeError("could not sample a connected graph")


# --------------------------------------------------------------------------
# Graph helpers
# --------------------------------------------------------------------------
def test_adjacency_is_symmetric_without_self_loops():
    rng = np.random.default_rng(0)
    _, adj = random_connected_graph(rng, 12)
    assert np.array_equal(adj, adj.T)
    assert np.trace(adj) == 0


def test_diameter_of_a_path_and_a_clique():
    path = np.zeros((5, 5), dtype=int)
    for i in range(4):
        path[i, i + 1] = path[i + 1, i] = 1
    assert diameter(path) == 4

    clique = 1 - np.eye(6, dtype=int)
    assert diameter(clique) == 1


def test_disconnected_graph_has_infinite_diameter():
    adj = np.zeros((4, 4), dtype=int)
    adj[0, 1] = adj[1, 0] = 1
    adj[2, 3] = adj[3, 2] = 1
    assert not is_connected(adj)
    assert not np.isfinite(diameter(adj))


def test_laplacian_rows_sum_to_zero():
    rng = np.random.default_rng(3)
    _, adj = random_connected_graph(rng, 9)
    assert np.allclose(laplacian(adj).sum(axis=1), 0.0)


# --------------------------------------------------------------------------
# The plan's three claims
# --------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(30))
def test_consensus_is_conflict_free_correct_and_bounded(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(4, 14))
    n_slots = int(rng.integers(1, min(4, n) + 1))
    pos, adj = random_connected_graph(rng, n)
    scores = rng.normal(size=n)

    res = consensus_elect(scores, adj, n_slots)

    assert res.agreed(), "agents disagree — the election is not conflict-free"
    assert res.winners == centralized_top_n(scores, n_slots)
    assert res.rounds <= diameter(adj), (
        f"took {res.rounds} rounds, diam(G) = {diameter(adj)}"
    )


@pytest.mark.parametrize("seed", range(10))
def test_consensus_on_a_path_graph_needs_the_full_diameter(seed):
    """A path is the worst case: information crawls one hop per round. If the
    bound held only on dense graphs it would be meaningless."""
    rng = np.random.default_rng(100 + seed)
    n = 8
    adj = np.zeros((n, n), dtype=int)
    for i in range(n - 1):
        adj[i, i + 1] = adj[i + 1, i] = 1
    scores = rng.normal(size=n)

    res = consensus_elect(scores, adj, 3)
    assert res.agreed()
    assert res.winners == centralized_top_n(scores, 3)
    assert res.rounds <= diameter(adj)


def test_ties_break_by_lowest_id():
    """Equal bids must not let two agents settle on different sets."""
    adj = 1 - np.eye(5, dtype=int)
    scores = np.array([1.0, 1.0, 1.0, 0.5, 0.5])
    res = consensus_elect(scores, adj, 2)
    assert res.agreed()
    assert res.winners == [0, 1]


def test_single_agent_and_more_slots_than_agents():
    res = consensus_elect(np.array([0.7]), np.zeros((1, 1), dtype=int), 3)
    assert res.winners == [0] and res.agreed()


def test_disconnected_graph_does_not_hang():
    """Consensus cannot succeed across a partition — it must terminate and the
    disagreement must be visible, not silently papered over."""
    adj = np.zeros((4, 4), dtype=int)
    adj[0, 1] = adj[1, 0] = 1
    adj[2, 3] = adj[3, 2] = 1
    res = consensus_elect(np.array([0.1, 0.9, 0.8, 0.2]), adj, 1)
    assert isinstance(res, ElectionResult)
    assert not res.agreed()


# --------------------------------------------------------------------------
# Coverage-aware bidding
# --------------------------------------------------------------------------
def test_locational_cost_improves_with_more_agents():
    region = ring_region()
    few = np.array([[8.0, 0.0], [-8.0, 0.0]])
    many = ring_swarm(10).pos
    assert locational_cost(many, region) < locational_cost(few, region)


def test_criticality_is_higher_for_a_load_bearing_agent():
    """Two agents clustered together are each redundant; a lone agent covering
    its own sector is not."""
    region = ring_region()
    pos = np.array([[8.0, 0.0], [7.8, 0.6], [-8.0, 0.0]])
    kappa = loiter_criticality(pos, region)
    assert kappa[2] > kappa[0] and kappa[2] > kappa[1]


def test_criticality_is_non_negative():
    kappa = loiter_criticality(ring_swarm(9).pos, ring_region())
    assert (kappa >= -1e-9).all()


def test_closer_agent_outbids_further_one_all_else_equal():
    """Isolate the distance term: equal battery, equal sensor, and w_r = 0 so
    coverage cannot enter. With the penalty ON, a closer agent legitimately can
    lose — that is what the next test is for."""
    m = 8
    a = 2 * np.pi * np.arange(m) / m
    state = SwarmState(
        pos=np.stack([8 * np.cos(a), 8 * np.sin(a)], axis=1),
        battery=np.ones(m), sensor=np.ones(m), region=ring_region(),
    )
    target = np.array([14.0, 0.0])
    c = bid_scores(state, target, BidWeights(w_r=0.0))
    dist = np.linalg.norm(state.pos - target, axis=1)
    assert int(np.argmax(c)) == int(np.argmin(dist))


def test_symmetric_ring_gives_symmetric_criticality():
    """Agents that are geometrically interchangeable must bid interchangeably.
    This is what caught the random-sampling noise in `ring_region`."""
    m = 8
    a = 2 * np.pi * np.arange(m) / m
    pos = np.stack([8 * np.cos(a), 8 * np.sin(a)], axis=1)
    kappa = loiter_criticality(pos, ring_region())
    assert kappa.std() < 0.02 * max(kappa.mean(), 1e-9)


def test_coverage_penalty_can_veto_a_close_agent():
    """The term that earns its keep: an agent right next to the target still
    loses if pulling it is what breaks the mesh."""
    region = ring_region()
    pos = np.array([[8.0, 0.0], [7.6, 0.9], [7.6, -0.9], [-8.0, 0.2]])
    target = np.array([9.5, 0.0])
    state = SwarmState(pos=pos, battery=np.ones(4), sensor=np.ones(4),
                       region=region)

    no_penalty = bid_scores(state, target, BidWeights(w_r=0.0))
    with_penalty = bid_scores(state, target, BidWeights(w_r=6.0))

    assert int(np.argmax(no_penalty)) == 0          # closest wins on distance
    kappa = loiter_criticality(pos, region)
    assert kappa[0] < kappa[3]                       # 0 is redundant, 3 is not
    assert with_penalty[3] < no_penalty[3]           # the lone agent is penalised


def test_untrusted_agents_cannot_win():
    """Domain 1 reaching into Domain 2: an agent whose covariance would trip the
    supervisor's trust gate must never be elected, or the FSM's zero-rejection
    promise is already broken at the auction."""
    m = 6
    a = 2 * np.pi * np.arange(m) / m
    cov = np.full(m, 0.001)
    cov[0] = 0.5                                     # estimator has lost it
    state = SwarmState(
        pos=np.stack([8 * np.cos(a), 8 * np.sin(a)], axis=1),
        battery=np.ones(m), sensor=np.ones(m), region=ring_region(),
        cov_trace=cov,
    )
    c = bid_scores(state, np.array([9.0, 0.0]))
    assert c[0] == -np.inf
    assert 0 not in consensus_elect(c, 1 - np.eye(m, dtype=int), 2).winners


def test_mismatched_state_lengths_rejected():
    with pytest.raises(ValueError):
        SwarmState(pos=np.zeros((3, 2)), battery=np.ones(2), sensor=np.ones(3),
                   region=ring_region())


# --------------------------------------------------------------------------
# 2.1 the adjacency cut
# --------------------------------------------------------------------------
def test_control_graph_is_block_diagonal_but_physical_is_not():
    rng = np.random.default_rng(7)
    pos, adj = random_connected_graph(rng, 10, lo=0.8, hi=1.5)
    active = [0, 3, 5]
    part = partition_control_graph(adj, active)

    # no control edge crosses the cut
    for i in part.active:
        for j in part.loiter:
            assert part.control_adj[i, j] == 0 and part.control_adj[j, i] == 0

    # the radios, however, still hear each other
    assert np.array_equal(part.physical_adj, adj)
    crossing = sum(adj[i, j] for i in active for j in part.loiter)
    assert crossing > 0, "test graph is too sparse to prove anything"
    assert part.control_adj.sum() < adj.sum()


def test_permutation_puts_the_active_set_first():
    rng = np.random.default_rng(2)
    _, adj = random_connected_graph(rng, 9, lo=0.8, hi=1.5)
    part = partition_control_graph(adj, [4, 1])
    assert part.perm[:2].tolist() == [1, 4]
    assert sorted(part.perm.tolist()) == list(range(9))

    a1, a2 = part.blocks()
    assert a1.shape == (2, 2) and a2.shape == (7, 7)


def test_block_laplacians_each_have_a_null_vector():
    """Each block's Laplacian annihilates its own all-ones vector — the two sets
    reach consensus independently, which is the point of cutting."""
    rng = np.random.default_rng(11)
    _, adj = random_connected_graph(rng, 8, lo=0.9, hi=1.5)
    part = partition_control_graph(adj, [0, 1, 2])
    for lap in part.laplacians():
        assert np.allclose(lap @ np.ones(len(lap)), 0.0)


def test_unknown_agent_in_active_set_rejected():
    adj = 1 - np.eye(4, dtype=int)
    with pytest.raises(ValueError):
        partition_control_graph(adj, [0, 9])
