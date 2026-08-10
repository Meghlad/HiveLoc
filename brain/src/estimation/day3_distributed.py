"""
Day 3: DISTRIBUTED localization via consensus ADMM. No central computer.

Each drone holds a private map of itself + its neighbors. Neighbors' maps overlap and
disagree at the "seams". ADMM negotiates the seams away using only neighbor-to-neighbor
messages, and the result matches the centralized solver from Days 1-2.

The loop, per iteration:
  1. LOCAL SOLVE  (no comms)  each node fits its own measurements + a pull toward consensus
  2. AVERAGE      (the comms) each position = average of all neighbors' opinions about it
  3. DUAL UPDATE  (no comms)  each node remembers its running disagreement and leans harder

Needs: pip install cvxpy numpy matplotlib scipy
Run:   python src/estimation/day3_distributed.py
"""

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

rng = np.random.default_rng(1)


# ----------------------------------------------------------------------
# Network (rigid, well-anchored - from Day 2 we know this localizes cleanly)
# ----------------------------------------------------------------------
n = 15
anchors = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], float)
R, sigma = 0.55, 0.02
X_true = rng.uniform(0.1, 0.9, size=(n, 2))

dist_s = {}                          # (i,j) -> noisy sensor-sensor distance
neigh_s = {i: [] for i in range(n)}  # sensor neighbors of i
neigh_a = {i: [] for i in range(n)}  # (anchor_index, noisy distance) for i
for i in range(n):
    for j in range(i + 1, n):
        dd = np.linalg.norm(X_true[i] - X_true[j])
        if dd <= R:
            m = dd + rng.normal(0, sigma)
            dist_s[(i, j)] = m; dist_s[(j, i)] = m
            neigh_s[i].append(j); neigh_s[j].append(i)
    for k in range(len(anchors)):
        dd = np.linalg.norm(X_true[i] - anchors[k])
        if dd <= R:
            neigh_a[i].append((k, dd + rng.normal(0, sigma)))

hold = {i: [i] + neigh_s[i] for i in range(n)}   # position components node i keeps a copy of


# ----------------------------------------------------------------------
# Centralized reference (Day 1 SDP) - the answer the swarm should reach on its own
# ----------------------------------------------------------------------
def solve_sdp():
    d = 2
    Z = cp.Variable((d + n, d + n), symmetric=True)
    cons = [Z >> 0, Z[:d, :d] == np.eye(d)]
    yi = lambda i: d + i
    terms = []
    for (i, j), m in dist_s.items():
        if i < j:
            terms.append(cp.abs(Z[yi(i), yi(i)] + Z[yi(j), yi(j)] - 2 * Z[yi(i), yi(j)] - m**2))
    for i in range(n):
        for (k, m) in neigh_a[i]:
            ak = anchors[k]
            ax = ak[0] * Z[0, yi(i)] + ak[1] * Z[1, yi(i)]
            terms.append(cp.abs(Z[yi(i), yi(i)] - 2 * ax + ak @ ak - m**2))
    cp.Problem(cp.Minimize(cp.sum(terms)), cons).solve(solver=cp.CLARABEL)
    return Z.value[:2, 2:].T

X_central = solve_sdp()
rmse_central = np.sqrt(np.mean(np.sum((X_central - X_true) ** 2, axis=1)))


# ----------------------------------------------------------------------
# Distributed consensus ADMM
# ----------------------------------------------------------------------
def run_admm(rho=2.0, n_iters=60):
    # Global consensus positions Z (the agreed map). Start from a no-info guess: the centroid.
    Z = np.tile([0.5, 0.5], (n, 1)) + rng.normal(0, 0.05, (n, 2))
    # Each node's PRIVATE copies of the positions it holds, and its dual ("grudge") variables.
    Xloc = {i: {p: Z[p].copy() for p in hold[i]} for i in range(n)}
    U    = {i: {p: np.zeros(2) for p in hold[i]} for i in range(n)}

    history_Z, rmse_hist, disagree_hist = [], [], []

    for it in range(n_iters):
        # --- MOVE 1: LOCAL SOLVE (every node, in parallel, no communication) ---
        for i in range(n):
            comps = hold[i]
            idx = {p: c for c, p in enumerate(comps)}
            x0 = np.concatenate([Xloc[i][p] for p in comps])

            def resid(vec, i=i, comps=comps, idx=idx):
                pos = {p: vec[2 * idx[p]:2 * idx[p] + 2] for p in comps}
                r = []
                # fit my own distance measurements (each shared edge split 1/2 with the neighbor)
                for j in neigh_s[i]:
                    r.append(np.sqrt(0.5) * (np.linalg.norm(pos[i] - pos[j]) - dist_s[(i, j)]))
                for (k, m) in neigh_a[i]:                       # anchors are fixed constants
                    r.append(np.linalg.norm(pos[i] - anchors[k]) - m)
                # the leash: stay near the current consensus (minus my accumulated grudge)
                for p in comps:
                    target = Z[p] - U[i][p]
                    r.extend(np.sqrt(rho / 2) * (pos[p] - target))
                return r

            sol = least_squares(resid, x0, method="lm")
            for p in comps:
                Xloc[i][p] = sol.x[2 * idx[p]:2 * idx[p] + 2]

        # --- MOVE 2: AVERAGE (the ONLY communication; neighbors only) ---
        for j in range(n):
            holders = [j] + neigh_s[j]                          # j itself + everyone who sees j
            Z[j] = np.mean([Xloc[i][j] + U[i][j] for i in holders], axis=0)

        # --- MOVE 3: DUAL UPDATE (every node remembers its disagreement) ---
        disagree = 0.0
        for i in range(n):
            for p in hold[i]:
                gap = Xloc[i][p] - Z[p]
                U[i][p] += gap
                disagree += np.linalg.norm(gap)

        history_Z.append(Z.copy())
        rmse_hist.append(np.sqrt(np.mean(np.sum((Z - X_true) ** 2, axis=1))))
        disagree_hist.append(disagree / sum(len(hold[i]) for i in range(n)))

    return history_Z, np.array(rmse_hist), np.array(disagree_hist)


history_Z, rmse_hist, disagree_hist = run_admm()
print(f"centralized SDP RMSE : {rmse_central:.4f} m")
print(f"distributed final RMSE: {rmse_hist[-1]:.4f} m   (should be ~the same - no central computer!)")
print(f"final seam disagreement: {disagree_hist[-1]:.2e}   (-> 0 means consensus reached)")


# ----------------------------------------------------------------------
# Storyboard: watch the swarm self-organize
# ----------------------------------------------------------------------
snaps = [0, 4, 12, len(history_Z) - 1]
fig, axes = plt.subplots(1, 4, figsize=(18, 4.6))
for ax, it in zip(axes, snaps):
    Z = history_Z[it]
    ax.scatter(anchors[:, 0], anchors[:, 1], c="k", marker="^", s=90)
    ax.scatter(X_true[:, 0], X_true[:, 1], c="green", s=30, label="true")
    ax.scatter(Z[:, 0], Z[:, 1], facecolors="none", edgecolors="red", s=70, label="consensus")
    for i in range(n):
        ax.plot([X_true[i, 0], Z[i, 0]], [X_true[i, 1], Z[i, 1]], "r-", alpha=0.4)
    ax.set_title(f"iteration {it}"); ax.axis("equal"); ax.grid(alpha=0.3)
axes[0].legend(loc="upper left")
plt.suptitle("Neighbors-only ADMM self-organizing into the true map")
plt.tight_layout(); plt.savefig("figures/day3_storyboard.png", dpi=120, bbox_inches="tight"); plt.show()

# ----------------------------------------------------------------------
# Convergence curves = the communication-vs-accuracy story
# ----------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
ax[0].plot(rmse_hist, "b-", lw=2, label="distributed (neighbors only)")
ax[0].axhline(rmse_central, color="k", ls="--", label="centralized SDP")
ax[0].set_xlabel("ADMM iteration  (= one neighbor message round each)")
ax[0].set_ylabel("RMSE to truth (m)"); ax[0].set_title("accuracy vs communication")
ax[0].legend(); ax[0].grid(alpha=0.3)

ax[1].semilogy(disagree_hist, "purple", lw=2)
ax[1].set_xlabel("ADMM iteration"); ax[1].set_ylabel("avg seam disagreement (log)")
ax[1].set_title("the seams closing -> consensus"); ax[1].grid(alpha=0.3)
plt.tight_layout(); plt.savefig("figures/day3_convergence.png", dpi=120, bbox_inches="tight"); plt.show()
print("saved day3_storyboard.png and day3_convergence.png")
