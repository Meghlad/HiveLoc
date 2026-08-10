"""
Day 5: a MOVING swarm. Time-varying graph, re-localize each step, animate it.

The new idea: temporal coherence. Don't re-solve from scratch each frame (amnesia).
Instead PREDICT where each drone should be from its motion, then CORRECT with the
new range measurements. The prediction is a soft anchor that holds the swarm steady
through frames that are momentarily under-constrained.

That predict-correct loop is a baby Kalman filter / factor-graph smoother. The motion
prior term  lambda*||x - x_pred||^2  is an "odometry factor" - on a real drone the IMU
provides exactly this. Same architecture as the flight system.

Needs: pip install cvxpy numpy matplotlib scipy pillow
Run:   python src/estimation/day5_dynamic.py
"""

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.optimize import least_squares

rng = np.random.default_rng(7)

n = 12
anchors = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], float)
R, sigma = 0.55, 0.02
T = 60                                    # number of time steps
lam = 0.5                                 # motion-prior strength (the "odometry factor")


# ----------------------------------------------------------------------
# Ground-truth motion: drones drift with constant velocity, bouncing off the box.
# ----------------------------------------------------------------------
X = rng.uniform(0.15, 0.85, size=(n, 2))
V = rng.uniform(-0.012, 0.012, size=(n, 2))
true_traj = np.zeros((T, n, 2))
for t in range(T):
    true_traj[t] = X
    X = X + V
    for c in range(2):                    # reflect at the walls
        lo, hi = X[:, c] < 0.08, X[:, c] > 0.92
        V[lo | hi, c] *= -1
        X[:, c] = np.clip(X[:, c], 0.08, 0.92)


def measure(Xt):
    """Build this frame's sensing graph + noisy ranges (who sees whom right now)."""
    se, ae = [], []
    for i in range(n):
        for j in range(i + 1, n):
            dd = np.linalg.norm(Xt[i] - Xt[j])
            if dd <= R:
                se.append((i, j, dd + rng.normal(0, sigma)))
        for k in range(len(anchors)):
            dd = np.linalg.norm(Xt[i] - anchors[k])
            if dd <= R:
                ae.append((i, k, dd + rng.normal(0, sigma)))
    return se, ae


# ----------------------------------------------------------------------
# Two estimators: COLD (static SDP every frame) vs WARM (SDP bootstrap + tracking)
# ----------------------------------------------------------------------
def solve_sdp(se, ae):
    d = 2
    Z = cp.Variable((d + n, d + n), symmetric=True)
    cons = [Z >> 0, Z[:d, :d] == np.eye(d)]
    yi = lambda i: d + i
    terms = []
    for (i, j, m) in se:
        terms.append(cp.abs(Z[yi(i), yi(i)] + Z[yi(j), yi(j)] - 2 * Z[yi(i), yi(j)] - m**2))
    for (i, k, m) in ae:
        ak = anchors[k]
        ax = ak[0] * Z[0, yi(i)] + ak[1] * Z[1, yi(i)]
        terms.append(cp.abs(Z[yi(i), yi(i)] - 2 * ax + ak @ ak - m**2))
    if not terms:
        return None
    cp.Problem(cp.Minimize(cp.sum(terms)), cons).solve(solver=cp.CLARABEL)
    return Z.value[:2, 2:].T


def correct(se, ae, X_pred):
    """PREDICT-CORRECT: fit measurements while staying near the motion prediction."""
    def res(xflat):
        Xc = xflat.reshape(n, 2)
        r = [np.linalg.norm(Xc[i] - Xc[j]) - m for (i, j, m) in se]
        r += [np.linalg.norm(Xc[i] - anchors[k]) - m for (i, k, m) in ae]
        for i in range(n):                       # the odometry factor / motion prior
            r.extend(np.sqrt(lam) * (Xc[i] - X_pred[i]))
        return r
    return least_squares(res, X_pred.flatten(), method="lm").x.reshape(n, 2)


frames = [measure(true_traj[t]) for t in range(T)]

cold = np.zeros((T, n, 2))
warm = np.zeros((T, n, 2))
for t in range(T):
    se, ae = frames[t]
    cold[t] = solve_sdp(se, ae)                  # amnesia: from scratch every frame
    if t == 0:
        warm[t] = solve_sdp(se, ae)              # bootstrap the map once with the SDP
    else:
        vel = warm[t - 1] - warm[t - 2] if t >= 2 else np.zeros((n, 2))
        X_pred = warm[t - 1] + vel               # constant-velocity prediction
        warm[t] = correct(se, ae, X_pred)        # correct with this frame's ranges

rmse_cold = np.sqrt(((cold - true_traj) ** 2).sum(2).mean(1))
rmse_warm = np.sqrt(((warm - true_traj) ** 2).sum(2).mean(1))
print(f"mean RMSE  cold (static each frame): {rmse_cold.mean():.4f} m  (jumpy)")
print(f"mean RMSE  warm (predict-correct)  : {rmse_warm.mean():.4f} m  (smooth, robust)")


# ----------------------------------------------------------------------
# RMSE over time: warm tracking survives the under-constrained frames
# ----------------------------------------------------------------------
plt.figure(figsize=(8, 4.2))
plt.plot(rmse_cold, "r-", alpha=0.8, label="static SDP each frame (amnesia)")
plt.plot(rmse_warm, "g-", lw=2, label="predict-correct tracking")
plt.xlabel("time step"); plt.ylabel("RMSE to truth (m)")
plt.title("Temporal coherence: prediction carries the swarm through bad frames")
plt.legend(); plt.grid(alpha=0.3)
plt.savefig("figures/day5_rmse_over_time.png", dpi=130, bbox_inches="tight"); plt.show()


# ----------------------------------------------------------------------
# Animation: the swarm flying, the graph flickering, the estimate tracking
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 6.5))

def draw(t):
    ax.clear()
    Xt, Et = true_traj[t], warm[t]
    se, ae = frames[t]
    for (i, j, _) in se:                          # current sensing graph (flickers as they move)
        ax.plot([Xt[i, 0], Xt[j, 0]], [Xt[i, 1], Xt[j, 1]], color="gray", lw=0.5, alpha=0.4)
    for tt in range(max(0, t - 12), t + 1):       # short trails
        ax.plot(true_traj[tt:tt + 2, :, 0], true_traj[tt:tt + 2, :, 1],
                color="green", alpha=0.15)
    ax.scatter(anchors[:, 0], anchors[:, 1], c="k", marker="^", s=110, zorder=3)
    ax.scatter(Xt[:, 0], Xt[:, 1], c="green", s=40, zorder=3, label="true")
    ax.scatter(Et[:, 0], Et[:, 1], facecolors="none", edgecolors="red", s=80, zorder=3, label="estimate")
    ax.set_xlim(-0.1, 1.1); ax.set_ylim(-0.1, 1.1); ax.set_aspect("equal")
    ax.set_title(f"t={t}   RMSE={rmse_warm[t]:.3f} m"); ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

anim = animation.FuncAnimation(fig, draw, frames=T, interval=120)
try:
    anim.save("figures/day5_swarm.gif", writer=animation.PillowWriter(fps=8))
    print("saved day5_swarm.gif and day5_rmse_over_time.png")
except Exception as e:
    print("GIF save failed (pip install pillow). Static frames still shown.", e)
plt.show()
