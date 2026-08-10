"""
Day 2: WHY the SDP works - alignment, rigidity, and the tightness phase-diagram.

Builds on Day 1. Three parts:
  PART A  Procrustes alignment  - score a *shape* fairly by overlaying it on the truth first.
  PART B  Rigidity check        - is the network a rigid truss or a floppy linkage?
  PART C  Phase diagram         - sweep connectivity x noise, find the cliff where it breaks.

Run:  python src/estimation/day2_rigidity.py
"""

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt


# ======================================================================
# Reusable pieces (these carry forward to Day 3+ - keep them).
# ======================================================================
def generate_network(n, anchors, R, sigma, seed=0):
    """Ground truth + sensing graph + noisy measurements."""
    rng = np.random.default_rng(seed)
    X_true = rng.uniform(0.1, 0.9, size=(n, 2))
    sensor_edges, anchor_edges = [], []
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(X_true[i] - X_true[j])
            if dist <= R:
                sensor_edges.append((i, j, dist + rng.normal(0, sigma)))
        for k in range(len(anchors)):
            dist = np.linalg.norm(X_true[i] - anchors[k])
            if dist <= R:
                anchor_edges.append((i, k, dist + rng.normal(0, sigma)))
    return X_true, sensor_edges, anchor_edges


def solve_sdp(n, anchors, sensor_edges, anchor_edges, d=2):
    """Biswas-Ye full SDP relaxation. Returns (X_est, Z_value)."""
    Z = cp.Variable((d + n, d + n), symmetric=True)
    cons = [Z >> 0, Z[:d, :d] == np.eye(d)]
    yi = lambda i: d + i
    terms = []
    for (i, j, dij) in sensor_edges:
        terms.append(cp.abs(Z[yi(i), yi(i)] + Z[yi(j), yi(j)] - 2 * Z[yi(i), yi(j)] - dij**2))
    for (i, k, dik) in anchor_edges:
        ak = anchors[k]
        ax = ak[0] * Z[0, yi(i)] + ak[1] * Z[1, yi(i)]
        terms.append(cp.abs(Z[yi(i), yi(i)] - 2 * ax + ak @ ak - dik**2))
    cp.Problem(cp.Minimize(cp.sum(terms)), cons).solve(solver=cp.CLARABEL)
    if Z.value is None:
        return None, None
    return Z.value[:d, d:].T, Z.value


# ======================================================================
# PART A - Procrustes alignment
# ======================================================================
def align_procrustes(X_est, X_true):
    """Best rotation+reflection+translation laying X_est onto X_true, then measure leftover.
    We ALLOW reflection on purpose: distances genuinely cannot distinguish a layout from its
    mirror image, so a flipped-but-perfect answer should score as correct."""
    mu_e, mu_t = X_est.mean(0), X_true.mean(0)
    Ec, Tc = X_est - mu_e, X_true - mu_t          # center both clouds
    U, _, Vt = np.linalg.svd(Ec.T @ Tc)           # cross-covariance -> best orthogonal map
    Q = U @ Vt                                    # rotation (or reflection)
    return Ec @ Q + mu_t


def part_A():
    print("\n=== PART A: Procrustes ===")
    n = 15
    anchors = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], float)
    X_true, se, ae = generate_network(n, anchors, R=0.45, sigma=0.02, seed=0)
    X_est, _ = solve_sdp(n, anchors, se, ae)

    # Deliberately rotate+flip+shift the estimate to mimic an "anchor-free" frame mismatch.
    theta = 0.9
    Rrot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    flip = np.array([[1, 0], [0, -1]])            # a mirror flip
    X_scrambled = X_est @ (Rrot @ flip) + np.array([2.0, -1.5])

    raw = np.sqrt(np.mean(np.sum((X_scrambled - X_true) ** 2, axis=1)))
    X_aligned = align_procrustes(X_scrambled, X_true)
    aligned = np.sqrt(np.mean(np.sum((X_aligned - X_true) ** 2, axis=1)))
    print(f"RMSE before alignment: {raw:.3f} m   (looks broken)")
    print(f"RMSE after  alignment: {aligned:.3f} m   (the shape was right all along)")


# ======================================================================
# PART B - Rigidity check
# ======================================================================
def rigidity_rank(X, anchors, sensor_edges, anchor_edges, d=2):
    """Rank of the anchored rigidity matrix. Uniquely localizable (generically, locally)
    iff rank == d*n, i.e. no joint of the sensor network can move while keeping every
    measured distance fixed. Computed on the TRUE layout - rigidity is a property of the
    geometry, used here for ANALYSIS (in the field you wouldn't know X)."""
    n = len(X)
    rows = []
    for (i, j, _) in sensor_edges:
        row = np.zeros(d * n); diff = X[i] - X[j]
        row[d*i:d*i+d] = diff; row[d*j:d*j+d] = -diff
        rows.append(row)
    for (i, k, _) in anchor_edges:
        row = np.zeros(d * n)
        row[d*i:d*i+d] = X[i] - anchors[k]        # anchor can't move -> only sensor i's columns
        rows.append(row)
    if not rows:
        return 0, d * n
    return np.linalg.matrix_rank(np.array(rows)), d * n


def part_B():
    print("\n=== PART B: Rigidity ===")
    n = 15
    anchors = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], float)
    for R in (0.50, 0.30, 0.22):                  # shrink radio range -> remove edges
        X_true, se, ae = generate_network(n, anchors, R=R, sigma=0.0, seed=0)
        rank, full = rigidity_rank(X_true, anchors, se, ae)
        verdict = "RIGID  (unique)" if rank == full else "FLOPPY (a flex exists -> hopeless)"
        print(f"R={R:.2f}: {len(se)+len(ae):3d} edges, rank {rank:2d}/{full} -> {verdict}")


# ======================================================================
# PART C - Tightness phase-diagram
# ======================================================================
def part_C():
    print("\n=== PART C: Phase diagram (this takes ~1 min) ===")
    n = 12
    anchors = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], float)
    Rs = np.linspace(0.25, 0.70, 7)               # connectivity axis
    sigmas = np.linspace(0.0, 0.08, 7)            # noise axis
    K = 6                                          # trials per cell

    rmse_grid = np.zeros((len(sigmas), len(Rs)))
    rigid_grid = np.zeros((len(sigmas), len(Rs)))
    for a, R in enumerate(Rs):
        for b, sig in enumerate(sigmas):
            errs, rigids = [], []
            for t in range(K):
                X_true, se, ae = generate_network(n, anchors, R, sig, seed=100 * a + 10 * b + t)
                rank, full = rigidity_rank(X_true, anchors, se, ae)
                rigids.append(rank == full)
                X_est, _ = solve_sdp(n, anchors, se, ae)
                if X_est is None:
                    errs.append(np.nan); continue
                errs.append(np.sqrt(np.mean(np.sum((X_est - X_true) ** 2, axis=1))))
            rmse_grid[b, a] = np.nanmedian(errs)
            rigid_grid[b, a] = np.mean(rigids)
        print(f"  R={R:.2f} column done")

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ext = [Rs[0], Rs[-1], sigmas[0], sigmas[-1]]
    im0 = ax[0].imshow(rmse_grid, origin="lower", aspect="auto", extent=ext,
                       vmax=0.25, cmap="viridis")
    ax[0].set_title("median localization RMSE (m)\ndark = good, bright = broken")
    ax[0].set_xlabel("connectivity radius R"); ax[0].set_ylabel("noise sigma")
    fig.colorbar(im0, ax=ax[0])

    im1 = ax[1].imshow(rigid_grid, origin="lower", aspect="auto", extent=ext,
                       vmin=0, vmax=1, cmap="RdYlGn")
    ax[1].set_title("fraction of trials that are RIGID\n(red = floppy graph -> geometric failure)")
    ax[1].set_xlabel("connectivity radius R"); ax[1].set_ylabel("noise sigma")
    fig.colorbar(im1, ax=ax[1])

    plt.tight_layout()
    plt.savefig("figures/day2_phase_diagram.png", dpi=130, bbox_inches="tight")
    plt.show()
    print("saved day2_phase_diagram.png")
    print("\nNotice: the RMSE 'cliff' on the LEFT lines up with the rigidity boundary -")
    print("that failure is geometric (too few edges). The gradient going UP is statistical")
    print("(rigid graph, but the measurements themselves are lying). Two different failure modes.")


if __name__ == "__main__":
    part_A()
    part_B()
    part_C()
