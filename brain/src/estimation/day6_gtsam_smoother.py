"""
Step 1 toward flight: a GTSAM batch factor-graph SMOOTHER for the moving swarm.

We run two estimators on the SAME data:
  (A) CAUSAL  - the Day 5 predict-correct tracker (uses data only up to time t)
  (B) SMOOTHER- one big factor graph over ALL times, solved at once (uses future to fix past)

Three factor types wire the graph:
  - anchor-range factor   : drone-at-t  <-> known anchor        (unary)
  - inter-drone range     : drone-at-t  <-> other drone-at-t    (binary, cooperative)
  - constant-velocity     : drone at t-2,t-1,t  (penalize accel; "no teleporting")  (ternary)

We use gtsam.CustomFactor for all three so the code doesn't depend on version-specific
built-in factor names - and so you can see exactly what a factor IS: an error + Jacobian.

Needs: pip install cvxpy numpy matplotlib scipy gtsam
Run:   python src/estimation/day6_gtsam_smoother.py
"""

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import gtsam
from gtsam.symbol_shorthand import X

rng = np.random.default_rng(7)

n = 12
anchors = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], float)
R, sigma = 0.55, 0.02
T = 60
lam = 0.5                       # Day-5 causal motion-prior strength
sigma_motion = 0.04             # smoother's "no teleporting" process noise (accel std)


# ----------------------------------------------------------------------
# Ground-truth motion + per-frame noisy measurements (same world as Day 5)
# ----------------------------------------------------------------------
Xpos = rng.uniform(0.15, 0.85, size=(n, 2))
V = rng.uniform(-0.012, 0.012, size=(n, 2))
true_traj = np.zeros((T, n, 2))
for t in range(T):
    true_traj[t] = Xpos
    Xpos = Xpos + V
    for c in range(2):
        bad = (Xpos[:, c] < 0.08) | (Xpos[:, c] > 0.92)
        V[bad, c] *= -1
        Xpos[:, c] = np.clip(Xpos[:, c], 0.08, 0.92)

def measure(Xt):
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

frames = [measure(true_traj[t]) for t in range(T)]


# ----------------------------------------------------------------------
# (A) CAUSAL baseline: SDP bootstrap at t=0, predict-correct afterward (= Day 5)
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
        ak = anchors[k]; ax = ak[0] * Z[0, yi(i)] + ak[1] * Z[1, yi(i)]
        terms.append(cp.abs(Z[yi(i), yi(i)] - 2 * ax + ak @ ak - m**2))
    cp.Problem(cp.Minimize(cp.sum(terms)), cons).solve(solver=cp.CLARABEL)
    return Z.value[:2, 2:].T

def correct(se, ae, X_pred):
    def res(xflat):
        Xc = xflat.reshape(n, 2)
        r = [np.linalg.norm(Xc[i] - Xc[j]) - m for (i, j, m) in se]
        r += [np.linalg.norm(Xc[i] - anchors[k]) - m for (i, k, m) in ae]
        for i in range(n):
            r.extend(np.sqrt(lam) * (Xc[i] - X_pred[i]))
        return r
    return least_squares(res, X_pred.flatten(), method="lm").x.reshape(n, 2)

causal = np.zeros((T, n, 2))
for t in range(T):
    se, ae = frames[t]
    if t == 0:
        causal[t] = solve_sdp(se, ae)
    else:
        vel = causal[t - 1] - causal[t - 2] if t >= 2 else np.zeros((n, 2))
        causal[t] = correct(se, ae, causal[t - 1] + vel)


# ----------------------------------------------------------------------
# (B) SMOOTHER: build ONE factor graph over all (drone, time) positions
# ----------------------------------------------------------------------
def key(i, t):
    return X(t * n + i)                       # one position variable per drone per time

range_noise  = gtsam.noiseModel.Isotropic.Sigma(1, sigma)          # matches measurement noise
motion_noise = gtsam.noiseModel.Isotropic.Sigma(2, sigma_motion)   # process noise

def anchor_factor(ki, apos, meas):
    def err(this, v, H):
        p = v.atPoint2(ki); diff = p - apos; dist = np.linalg.norm(diff) + 1e-9
        if H is not None:
            H[0] = (diff / dist).reshape(1, 2)
        return np.array([dist - meas])
    return gtsam.CustomFactor(range_noise, [ki], err)

def range_factor(ki, kj, meas):
    def err(this, v, H):
        pi = v.atPoint2(ki); pj = v.atPoint2(kj)
        diff = pi - pj; dist = np.linalg.norm(diff) + 1e-9
        if H is not None:
            u = (diff / dist).reshape(1, 2); H[0] = u; H[1] = -u
        return np.array([dist - meas])
    return gtsam.CustomFactor(range_noise, [ki, kj], err)

def motion_factor(k0, k1, k2):                 # penalize acceleration: p2 - 2 p1 + p0 ~ 0
    def err(this, v, H):
        p0 = v.atPoint2(k0); p1 = v.atPoint2(k1); p2 = v.atPoint2(k2)
        if H is not None:
            H[0] = np.eye(2); H[1] = -2 * np.eye(2); H[2] = np.eye(2)
        return p2 - 2 * p1 + p0
    return gtsam.CustomFactor(motion_noise, [k0, k1, k2], err)

graph = gtsam.NonlinearFactorGraph()
for t in range(T):
    se, ae = frames[t]
    for (i, j, m) in se:
        graph.add(range_factor(key(i, t), key(j, t), m))
    for (i, k, m) in ae:
        graph.add(anchor_factor(key(i, t), anchors[k], m))
for t in range(2, T):
    for i in range(n):
        graph.add(motion_factor(key(i, t - 2), key(i, t - 1), key(i, t)))

# Initialize from the causal pass (a good guess keeps LM in the right basin)
initial = gtsam.Values()
for t in range(T):
    for i in range(n):
        initial.insert(key(i, t), causal[t, i].astype(float))

result = gtsam.LevenbergMarquardtOptimizer(graph, initial).optimize()

smoothed = np.zeros((T, n, 2))
for t in range(T):
    for i in range(n):
        smoothed[t, i] = result.atPoint2(key(i, t))   # if this errors, try result.atVector(...)


# ----------------------------------------------------------------------
# Compare
# ----------------------------------------------------------------------
rmse_causal   = np.sqrt(((causal   - true_traj) ** 2).sum(2).mean(1))
rmse_smoothed = np.sqrt(((smoothed - true_traj) ** 2).sum(2).mean(1))
print(f"mean RMSE  causal (Day 5)      : {rmse_causal.mean():.4f} m")
print(f"mean RMSE  smoother (GTSAM)    : {rmse_smoothed.mean():.4f} m")
print(f"improvement                    : {100*(1 - rmse_smoothed.mean()/rmse_causal.mean()):.1f}%")

worst = int(np.argmax(rmse_causal))    # the frame the causal tracker botched the most
print(f"worst causal frame: t={worst}  (causal {rmse_causal[worst]:.3f} m  ->  smoother {rmse_smoothed[worst]:.3f} m)")

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ax[0].plot(rmse_causal, "r-", alpha=0.85, label="causal (Day 5)")
ax[0].plot(rmse_smoothed, "b-", lw=2, label="batch smoother (GTSAM)")
ax[0].axvline(worst, color="gray", ls=":", label=f"worst causal frame (t={worst})")
ax[0].set_xlabel("time step"); ax[0].set_ylabel("RMSE to truth (m)")
ax[0].set_title("Future fixes the past: smoother stays low at causal's spikes")
ax[0].legend(); ax[0].grid(alpha=0.3)

t = worst
ax[1].scatter(anchors[:, 0], anchors[:, 1], c="k", marker="^", s=110)
ax[1].scatter(true_traj[t, :, 0], true_traj[t, :, 1], c="green", s=45, label="true")
ax[1].scatter(causal[t, :, 0], causal[t, :, 1], facecolors="none", edgecolors="red", s=80, label="causal")
ax[1].scatter(smoothed[t, :, 0], smoothed[t, :, 1], marker="x", c="blue", s=60, label="smoother")
ax[1].set_title(f"the botched frame t={worst}: causal vs smoother"); ax[1].axis("equal")
ax[1].legend(); ax[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figures/day6_smoother_vs_causal.png", dpi=130, bbox_inches="tight"); plt.show()
print("saved day6_smoother_vs_causal.png")
