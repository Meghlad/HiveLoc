"""
Day 4: ROBUSTNESS (graduated non-convexity) + CERTIFIABILITY (SDP optimality gap).

Story:
  - An adversary spoofs some range measurements (big lies).
  - Plain least-squares gets dragged off course by even one lie.
  - GNC-TLS discovers and rejects the lies WITHOUT being told which they are,
    and WITHOUT a good initial guess of the inlier set.
  - We then CERTIFY the cleaned solution is globally optimal, by squeezing the
    true optimum between an SDP lower bound (dreamer) and our map's cost (realist).

Needs: pip install cvxpy numpy matplotlib scipy
Run:   python src/estimation/day4_robust_certify.py
"""

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

anchors = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], float)
n = 15
R, sigma = 0.55, 0.02


# ----------------------------------------------------------------------
# Build a network, then let an adversary spoof a fraction of sensor edges.
# ----------------------------------------------------------------------
def build(outlier_frac=0.2, seed=0):
    rng = np.random.default_rng(seed)
    X_true = rng.uniform(0.1, 0.9, size=(n, 2))
    sensor_edges, anchor_edges = [], []
    for i in range(n):
        for j in range(i + 1, n):
            dd = np.linalg.norm(X_true[i] - X_true[j])
            if dd <= R:
                sensor_edges.append([i, j, dd + rng.normal(0, sigma)])
        for k in range(len(anchors)):
            dd = np.linalg.norm(X_true[i] - anchors[k])
            if dd <= R:
                anchor_edges.append([i, k, dd + rng.normal(0, sigma)])

    # SPOOF: corrupt a random subset of sensor edges with large lies.
    n_out = int(round(outlier_frac * len(sensor_edges)))
    out_idx = set(rng.choice(len(sensor_edges), size=n_out, replace=False)) if n_out else set()
    for e in out_idx:
        sensor_edges[e][2] += rng.uniform(0.3, 0.6) * rng.choice([-1, 1])
    return X_true, sensor_edges, anchor_edges, out_idx


# Flatten all measurements into one list M so GNC can weight each uniformly.
def make_M(sensor_edges, anchor_edges):
    M = [("s", i, j, d) for (i, j, d) in sensor_edges]
    M += [("a", i, k, d) for (i, k, d) in anchor_edges]
    return M


def residuals(M, X):
    r = []
    for (t, i, j, d) in M:
        other = X[j] if t == "s" else anchors[j]
        r.append(np.linalg.norm(X[i] - other) - d)   # range residual (meters)
    return np.array(r)


# ----------------------------------------------------------------------
# Weighted range least-squares (the inner solver). w=0 edge = ignored.
# ----------------------------------------------------------------------
def weighted_ls(M, w, X_init):
    def res(xflat):
        X = xflat.reshape(n, 2)
        return np.sqrt(w) * residuals(M, X)
    sol = least_squares(res, X_init.flatten(), method="lm")
    return sol.x.reshape(n, 2)


# ----------------------------------------------------------------------
# GNC-TLS: find inliers and solution simultaneously (Yang et al. 2020).
# ----------------------------------------------------------------------
def gnc_tls(M, X_init, c_bar):
    w = np.ones(len(M))
    X = weighted_ls(M, w, X_init)          # initial non-robust fit
    r2 = residuals(M, X) ** 2
    if 2 * r2.max() <= c_bar**2:           # nothing looks like an outlier
        return X, w
    mu = c_bar**2 / (2 * r2.max() - c_bar**2)   # start "blurry" (near-convex)
    for _ in range(100):
        lo, hi = mu / (mu + 1) * c_bar**2, (mu + 1) / mu * c_bar**2
        r = residuals(M, X)
        w_new = np.where(r**2 <= lo, 1.0,
                 np.where(r**2 >= hi, 0.0,
                          c_bar / (np.abs(r) + 1e-12) * np.sqrt(mu * (mu + 1)) - mu))
        w_new = np.clip(w_new, 0, 1)
        X = weighted_ls(M, w_new, X)       # refit with sharpened weights
        if np.max(np.abs(w_new - w)) < 1e-3 and mu > 1:
            w = w_new; break
        w = w_new
        mu *= 1.4                           # "focus" a little more
        if mu > 1e6:
            break
    return X, w


# ----------------------------------------------------------------------
# SDP (Day-1 L1 form) -> returns rounded positions AND the lower bound.
# ----------------------------------------------------------------------
def solve_sdp(sensor_edges, anchor_edges):
    d = 2
    Z = cp.Variable((d + n, d + n), symmetric=True)
    cons = [Z >> 0, Z[:d, :d] == np.eye(d)]
    yi = lambda i: d + i
    terms = []
    for (i, j, m) in sensor_edges:
        terms.append(cp.abs(Z[yi(i), yi(i)] + Z[yi(j), yi(j)] - 2 * Z[yi(i), yi(j)] - m**2))
    for (i, k, m) in anchor_edges:
        ak = anchors[k]
        ax = ak[0] * Z[0, yi(i)] + ak[1] * Z[1, yi(i)]
        terms.append(cp.abs(Z[yi(i), yi(i)] - 2 * ax + ak @ ak - m**2))
    prob = cp.Problem(cp.Minimize(cp.sum(terms)), cons)
    prob.solve(solver=cp.CLARABEL)
    return Z.value[:2, 2:].T, prob.value       # X_round, lower_bound


def true_cost(X, sensor_edges, anchor_edges):
    """The actual (nonconvex) L1 squared-distance cost = the realist's upper bound."""
    g = 0.0
    for (i, j, m) in sensor_edges:
        g += abs(np.sum((X[i] - X[j])**2) - m**2)
    for (i, k, m) in anchor_edges:
        g += abs(np.sum((X[i] - anchors[k])**2) - m**2)
    return g


def rmse(X, X_true):
    return np.sqrt(np.mean(np.sum((X - X_true)**2, axis=1)))


# ======================================================================
# DEMO: one network with 20% spoofed edges
# ======================================================================
X_true, se, ae, out_idx = build(outlier_frac=0.2, seed=3)
M = make_M(se, ae)
c_bar = 3 * sigma                                   # inlier band ~ 3 sigma

X_sdp, lb = solve_sdp(se, ae)                       # SDP init (somewhat robust already)
X_ls = weighted_ls(M, np.ones(len(M)), X_sdp)       # plain L2 (NON-robust)
X_gnc, w = gnc_tls(M, X_sdp, c_bar)                 # GNC-TLS (robust)

print("=== DEMO: 20% of sensor edges spoofed ===")
print(f"plain least-squares RMSE : {rmse(X_ls, X_true):.4f} m   <- dragged by the lies")
print(f"GNC-TLS RMSE             : {rmse(X_gnc, X_true):.4f} m   <- shrugs them off")

# How well did GNC find the spoofed edges? (sensor edges are first len(se) of M)
flagged = {e for e in range(len(se)) if w[e] < 0.5}
tp = len(flagged & out_idx); fp = len(flagged - out_idx); fn = len(out_idx - flagged)
print(f"outlier detection: caught {tp}/{len(out_idx)} spoofs, {fp} false alarms")

# ---- CERTIFY the cleaned solution: solve SDP on GNC-inlier edges, squeeze the optimum ----
inlier_se = [se[e] for e in range(len(se)) if w[e] >= 0.5]
X_cert, lower = solve_sdp(inlier_se, ae)
upper = true_cost(X_cert, inlier_se, ae)
gap = upper - lower
rel = gap / (upper + 1e-12)
print(f"\n=== CERTIFICATE (on GNC-cleaned graph) ===")
print(f"SDP lower bound (dreamer): {lower:.5f}")
print(f"our map's cost  (realist): {upper:.5f}")
verdict = "CERTIFIED GLOBALLY OPTIMAL" if rel < 1e-2 else f"certified within {100*rel:.1f}% of global optimum"
print(f"suboptimality gap: {gap:.2e}  ->  {verdict}")
print(f"certified map RMSE: {rmse(X_cert, X_true):.4f} m")


# ----------------------------------------------------------------------
# FIG 1: did GNC flag the right edges? color edges by weight.
# ----------------------------------------------------------------------
plt.figure(figsize=(6.5, 6.5))
plt.scatter(anchors[:, 0], anchors[:, 1], c="k", marker="^", s=110, zorder=3)
plt.scatter(X_true[:, 0], X_true[:, 1], c="green", s=45, zorder=3)
for e, (i, j, d) in enumerate(se):
    col = plt.cm.RdYlGn(w[e])                       # red=rejected, green=trusted
    style = "--" if e in out_idx else "-"           # dashed = actually spoofed
    lw = 3 if e in out_idx else 1.3
    plt.plot([X_true[i, 0], X_true[j, 0]], [X_true[i, 1], X_true[j, 1]],
             style, color=col, lw=lw, alpha=0.9)
plt.title("GNC edge weights (red=rejected, green=trusted)\ndashed = truly spoofed edge")
plt.axis("equal"); plt.grid(alpha=0.3)
plt.savefig("figures/day4_outlier_detection.png", dpi=130, bbox_inches="tight"); plt.show()


# ----------------------------------------------------------------------
# FIG 2: the headline robustness curve - RMSE vs outlier fraction
# ----------------------------------------------------------------------
print("\n=== SWEEP: RMSE vs outlier fraction (takes ~1 min) ===")
fracs = np.linspace(0, 0.6, 7)
K = 5
ls_curve, gnc_curve = [], []
for f in fracs:
    ls_e, gnc_e = [], []
    for t in range(K):
        Xt, se2, ae2, _ = build(outlier_frac=f, seed=10 * t + 1)
        M2 = make_M(se2, ae2)
        Xs, _ = solve_sdp(se2, ae2)
        ls_e.append(rmse(weighted_ls(M2, np.ones(len(M2)), Xs), Xt))
        Xg, _ = gnc_tls(M2, Xs, c_bar)
        gnc_e.append(rmse(Xg, Xt))
    ls_curve.append(np.median(ls_e)); gnc_curve.append(np.median(gnc_e))
    print(f"  outlier frac {f:.2f} done")

plt.figure(figsize=(7.5, 5))
plt.plot(fracs, ls_curve, "r-o", label="non-robust (least squares)")
plt.plot(fracs, gnc_curve, "g-o", label="robust (GNC-TLS)")
plt.xlabel("fraction of measurements spoofed"); plt.ylabel("median RMSE to truth (m)")
plt.title("Robustness: GNC stays flat while least-squares blows up")
plt.legend(); plt.grid(alpha=0.3)
plt.savefig("figures/day4_robustness_curve.png", dpi=130, bbox_inches="tight"); plt.show()
print("saved day4_outlier_detection.png and day4_robustness_curve.png")
