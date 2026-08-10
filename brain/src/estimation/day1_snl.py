"""
Day 1: Cooperative localization via the Biswas-Ye SDP relaxation.

The story in one file:
  1. GROUND TRUTH   - we (the simulator) place sensors + anchors. The drones never see this.
  2. SENSING GRAPH  - two nodes get an edge only if they're within radio range R.
  3. MEASUREMENTS   - each edge gives a noisy distance. This + anchor positions is ALL the solver gets.
  4. ESTIMATOR      - the SDP. We solve for positions WITHOUT ever using ground truth.
  5. SCORING        - compare estimate to the truth we secretly held back.

Setup:  python3 -m venv swarm && source swarm/bin/activate
        pip install cvxpy numpy matplotlib scipy
Run:    python src/estimation/day1_snl.py
"""

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)

# ----------------------------------------------------------------------
# 1. GROUND TRUTH  (only the simulator knows this)
# ----------------------------------------------------------------------
d = 2                      # working in the plane
n = 15                     # number of UNKNOWN sensors (the drones)
# Anchors at known positions pin the frame so the map can't rotate/slide/flip.
anchors = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])   # known anchor positions
m = len(anchors)

X_true = rng.uniform(0.1, 0.9, size=(n, d))   # true sensor positions (HIDDEN from solver)

R = 0.3        # radio range: nodes closer than R can measure each other
sigma = 0.02    # measurement noise std-dev (meters, on a 1x1 m field)

# ----------------------------------------------------------------------
# 2 + 3. SENSING GRAPH and NOISY MEASUREMENTS
# ----------------------------------------------------------------------
sensor_edges = []   # (i, j, measured_distance)  between two unknown sensors
anchor_edges = []   # (i, k, measured_distance)  between sensor i and anchor k

for i in range(n):
    for j in range(i + 1, n):
        dist = np.linalg.norm(X_true[i] - X_true[j])
        if dist <= R:                                   # in range -> they get an edge
            sensor_edges.append((i, j, dist + rng.normal(0, sigma)))
    for k in range(m):
        dist = np.linalg.norm(X_true[i] - anchors[k])
        if dist <= R:
            anchor_edges.append((i, k, dist + rng.normal(0, sigma)))

print(f"{len(sensor_edges)} sensor-sensor edges, {len(anchor_edges)} sensor-anchor edges")

# ----------------------------------------------------------------------
# 4. ESTIMATOR: the Biswas-Ye SDP relaxation
# ----------------------------------------------------------------------
# THE LIFTING TRICK. Instead of solving for positions X directly (non-convex, full of
# local-minimum traps), we solve for the matrix of all pairwise inner products.
# Build one big PSD matrix:
#       Z = [ I_d    X  ]        X  is d x n  (the positions we want)
#           [ X^T    Y  ]        Y = X^T X    (the Gram matrix of pairwise relationships)
# Forcing Z >= 0 (positive semidefinite) with top-left block = I is the convex relaxation
# of "Y = X^T X". Distances become LINEAR in Z, so the whole thing is a convex SDP.
Z = cp.Variable((d + n, d + n), symmetric=True)
constraints = [Z >> 0, Z[:d, :d] == np.eye(d)]

def yi(i):  # index of sensor i inside the Y block
    return d + i

terms = []
# ||x_i - x_j||^2 = Y_ii + Y_jj - 2 Y_ij    should match measured^2
for (i, j, dij) in sensor_edges:
    expr = Z[yi(i), yi(i)] + Z[yi(j), yi(j)] - 2 * Z[yi(i), yi(j)] - dij**2
    terms.append(cp.abs(expr))
# ||x_i - a_k||^2 = Y_ii - 2 a_k . x_i + a_k.a_k    should match measured^2
for (i, k, dik) in anchor_edges:
    ak = anchors[k]
    ax = ak[0] * Z[0, yi(i)] + ak[1] * Z[1, yi(i)]   # a_k . x_i  (x_i lives in the top rows)
    expr = Z[yi(i), yi(i)] - 2 * ax + ak @ ak - dik**2
    terms.append(cp.abs(expr))

# Minimize total absolute mismatch (L1 is naturally a bit outlier-tolerant).
prob = cp.Problem(cp.Minimize(cp.sum(terms)), constraints)
prob.solve(solver=cp.CLARABEL)   # try cp.SCS if a problem ever gets too big for Clarabel
print(f"solver status: {prob.status}")

X_est = Z.value[:d, d:].T        # pull the X block out -> n x d estimated positions
# Rank check: if Z is (numerically) rank d, the relaxation was TIGHT (a valid flat map).
eigs = np.linalg.eigvalsh(Z.value)
print(f"top eigenvalues of Z: {np.round(eigs[-4:], 4)}  (a big gap after the top {d} => tight)")

# ----------------------------------------------------------------------
# 5. SCORING.  Anchors pin the frame, so we can compare directly (no alignment needed).
#    (With too few/collinear anchors you'd first align via Procrustes - that's Day 2.)
# ----------------------------------------------------------------------
rmse = np.sqrt(np.mean(np.sum((X_est - X_true) ** 2, axis=1)))
print(f"RMSE: {rmse:.4f} m  (noise sigma was {sigma})")

# ----------------------------------------------------------------------
# Plot: truth vs estimate
# ----------------------------------------------------------------------
plt.figure(figsize=(6, 6))
plt.scatter(anchors[:, 0], anchors[:, 1], c="k", marker="^", s=120, label="anchors (known)")
plt.scatter(X_true[:, 0], X_true[:, 1], c="green", s=40, label="true")
plt.scatter(X_est[:, 0], X_est[:, 1], facecolors="none", edgecolors="red", s=80, label="estimated")
for i in range(n):
    plt.plot([X_true[i, 0], X_est[i, 0]], [X_true[i, 1], X_est[i, 1]], "r-", alpha=0.4)
plt.title(f"Biswas-Ye SDP localization  (RMSE={rmse:.3f} m)")
plt.legend(); plt.axis("equal"); plt.grid(alpha=0.3)
plt.savefig("figures/day1_localization.png", dpi=130, bbox_inches="tight")
plt.show()
print("saved day1_localization.png")
