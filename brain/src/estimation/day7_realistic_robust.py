"""
Step 2 toward flight: REAL dynamics + REAL noise + a ROBUST smoother.

Two changes from the day6 smoother:
  (1) Motion: drones no longer teleport. Each is a point-mass quadrotor under a PID
      position controller tracking a smooth path - so it accelerates, turns, lags, and
      saturates like a real vehicle. This STRESSES the constant-velocity motion factor.
  (2) Ranging: clean Gaussian noise -> a realistic UWB model:
        - positive NLOS bias (through a body, the signal arrives LATE => reads LONGER)
        - occasional multipath OUTLIERS (big spikes)
        - DROPOUTS (links vanish), which thin the graph
      This breaks the plain smoother. We fix it with a GTSAM ROBUST noise model
      (an m-estimator) wrapped around the range factors - the production version of
      Day 4's graduated non-convexity.

NOTE: this uses a self-contained quadrotor model so it always runs on an 8GB M1.
The gym-pybullet-drones / PX4-SITL swap plugs in at the SAME place (it only replaces
the block that produces `true_traj`); the estimator below is unchanged.

Needs: pip install cvxpy numpy matplotlib scipy gtsam
Run:   python src/estimation/day7_realistic_robust.py
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
R = 0.55
T, dt = 80, 0.1

# UWB error model parameters (unit-square field)
sigma_uwb   = 0.015     # base line-of-sight noise
p_nlos      = 0.15      # chance a reading is non-line-of-sight (biased long)
nlos_scale  = 0.05      # mean of the positive NLOS bias
p_outlier   = 0.03      # chance of a multipath spike
p_dropout   = 0.10      # chance a link simply isn't reported

sigma_motion = 0.06     # LOOSER than day6: real motion actually accelerates


# ----------------------------------------------------------------------
# (1) REAL dynamics: PID-controlled point-mass quadrotors tracking smooth paths.
#     <-- this whole block is what gym-pybullet-drones / PX4 SITL would replace -->
# ----------------------------------------------------------------------
freq  = rng.uniform(0.3, 0.8, (n, 2))
phase = rng.uniform(0, 2 * np.pi, (n, 2))
amp   = rng.uniform(0.20, 0.32, (n, 2))
center = np.array([0.5, 0.5])

def target(t):                              # smooth per-drone Lissajous path
    return center + amp * np.sin(freq * (t * dt) + phase)

pos = target(0).copy()
vel = np.zeros((n, 2))
Kp, Kd, v_max, a_max = 6.0, 4.0, 0.6, 3.0
true_traj = np.zeros((T, n, 2))
for t in range(T):
    true_traj[t] = pos
    acc = Kp * (target(t) - pos) - Kd * vel             # PID toward the moving target
    acc = np.clip(acc, -a_max, a_max)
    vel = np.clip(vel + acc * dt, -v_max, v_max)        # momentum + speed limit
    pos = pos + vel * dt


# ----------------------------------------------------------------------
# (2) REAL ranging: the UWB error model
# ----------------------------------------------------------------------
def uwb(true_dist):
    if rng.random() < p_dropout:
        return None                                     # link not reported this frame
    err = rng.normal(0, sigma_uwb)
    if rng.random() < p_nlos:
        err += rng.exponential(nlos_scale)              # NLOS: always biased LONGER
    if rng.random() < p_outlier:
        err += rng.uniform(0.2, 0.5)                    # multipath spike
    return true_dist + err

def make_frame(Xt):
    se, ae = [], []
    for i in range(n):
        for j in range(i + 1, n):
            dd = np.linalg.norm(Xt[i] - Xt[j])
            if dd <= R:
                m = uwb(dd)
                if m is not None:
                    se.append((i, j, m))
        for k in range(len(anchors)):
            dd = np.linalg.norm(Xt[i] - anchors[k])
            if dd <= R:
                m = uwb(dd)
                if m is not None:
                    ae.append((i, k, m))
    return se, ae

frames = [make_frame(true_traj[t]) for t in range(T)]


# ----------------------------------------------------------------------
# Causal pass (SDP bootstrap + predict-correct) -> used to initialize the smoother
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

def correct(se, ae, X_pred, lam=0.5):
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
        vel_est = causal[t - 1] - causal[t - 2] if t >= 2 else np.zeros((n, 2))
        causal[t] = correct(se, ae, causal[t - 1] + vel_est)


# ----------------------------------------------------------------------
# Smoother, parameterized by the range noise model (plain vs robust)
# ----------------------------------------------------------------------
def key(i, t):
    return X(t * n + i)

motion_noise = gtsam.noiseModel.Isotropic.Sigma(2, sigma_motion)

def build_and_solve(range_noise):
    def anchor_factor(ki, apos, meas):
        def err(this, v, H):
            p = v.atPoint2(ki); diff = p - apos; dist = np.linalg.norm(diff) + 1e-9
            if H is not None: H[0] = (diff / dist).reshape(1, 2)
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

    def motion_factor(k0, k1, k2):
        def err(this, v, H):
            p0 = v.atPoint2(k0); p1 = v.atPoint2(k1); p2 = v.atPoint2(k2)
            if H is not None: H[0] = np.eye(2); H[1] = -2 * np.eye(2); H[2] = np.eye(2)
            return p2 - 2 * p1 + p0
        return gtsam.CustomFactor(motion_noise, [k0, k1, k2], err)

    graph = gtsam.NonlinearFactorGraph()
    for t in range(T):
        se, ae = frames[t]
        for (i, j, m) in se: graph.add(range_factor(key(i, t), key(j, t), m))
        for (i, k, m) in ae: graph.add(anchor_factor(key(i, t), anchors[k], m))
    for t in range(2, T):
        for i in range(n): graph.add(motion_factor(key(i, t - 2), key(i, t - 1), key(i, t)))

    initial = gtsam.Values()
    for t in range(T):
        for i in range(n):
            initial.insert(key(i, t), causal[t, i].astype(float))

    result = gtsam.LevenbergMarquardtOptimizer(graph, initial).optimize()
    out = np.zeros((T, n, 2))
    for t in range(T):
        for i in range(n):
            out[t, i] = result.atPoint2(key(i, t))
    return out

base = gtsam.noiseModel.Isotropic.Sigma(1, sigma_uwb)
# Robust = wrap the base noise in an m-estimator. Huber is convex+stable; for harder
# outliers try Tukey/GemanMcClure (redescending, reject harder, need a decent init).
robust = gtsam.noiseModel.Robust.Create(
    gtsam.noiseModel.mEstimator.Huber.Create(1.345), base)

plain_traj  = build_and_solve(base)
robust_traj = build_and_solve(robust)


# ----------------------------------------------------------------------
# Compare on the realistic UWB data
# ----------------------------------------------------------------------
def rmse_t(traj):
    return np.sqrt(((traj - true_traj) ** 2).sum(2).mean(1))

r_plain, r_robust = rmse_t(plain_traj), rmse_t(robust_traj)
print(f"mean RMSE  plain smoother  (Gaussian assumption): {r_plain.mean():.4f} m")
print(f"mean RMSE  robust smoother (Huber m-estimator)  : {r_robust.mean():.4f} m")
print(f"robustness gain                                  : {100*(1 - r_robust.mean()/r_plain.mean()):.1f}%")
worst = int(np.argmax(r_plain))
print(f"worst plain frame t={worst}: plain {r_plain[worst]:.3f} m -> robust {r_robust[worst]:.3f} m")

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ax[0].plot(r_plain, "r-", alpha=0.85, label="plain (assumes clean Gaussian)")
ax[0].plot(r_robust, "b-", lw=2, label="robust (Huber m-estimator)")
ax[0].set_xlabel("time step"); ax[0].set_ylabel("RMSE to truth (m)")
ax[0].set_title("Realistic UWB noise: outliers wreck plain, robust shrugs them off")
ax[0].legend(); ax[0].grid(alpha=0.3)

t = worst
ax[1].scatter(anchors[:, 0], anchors[:, 1], c="k", marker="^", s=110)
ax[1].plot(true_traj[:, :, 0], true_traj[:, :, 1], color="green", alpha=0.15)
ax[1].scatter(true_traj[t, :, 0], true_traj[t, :, 1], c="green", s=45, label="true")
ax[1].scatter(plain_traj[t, :, 0], plain_traj[t, :, 1], facecolors="none", edgecolors="red", s=80, label="plain")
ax[1].scatter(robust_traj[t, :, 0], robust_traj[t, :, 1], marker="x", c="blue", s=60, label="robust")
ax[1].set_title(f"worst plain frame t={worst}"); ax[1].axis("equal")
ax[1].legend(); ax[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figures/day7_robust_smoother.png", dpi=130, bbox_inches="tight"); plt.show()
print("saved day7_robust_smoother.png")
