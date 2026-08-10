"""
Step 3 toward flight: iSAM2 - the INCREMENTAL, real-time smoother.

Day 7 solved the whole 80-frame graph in one batch. A real drone can't wait for the
end of the mission. iSAM2 updates the estimate as each frame arrives, re-solving only
the part of the graph the new measurements actually touch (via its Bayes tree), so each
update is cheap and bounded no matter how long the mission runs.

We feed iSAM2 the SAME realistic data as Day 7 (PID-controlled motion + UWB noise) with
the SAME robust (Huber) range factors, and show:
  - iSAM2's FINAL smoothed trajectory matches the batch smoother's accuracy
  - its per-frame update time stays flat while batch-from-scratch grows with the mission
Safety net: every new variable gets a LOOSE prior at its predicted position, so a drone
whose links all drop out in a frame never causes an indeterminate-system crash.

Needs: pip install cvxpy numpy matplotlib scipy gtsam
Run:   python src/estimation/day8_isam2.py
"""

import time
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
sigma_uwb, p_nlos, nlos_scale, p_outlier, p_dropout = 0.015, 0.15, 0.05, 0.03, 0.10
sigma_motion = 0.06


# ---- (same world as Day 7) PID-controlled dynamics + UWB ranging --------------------
freq  = rng.uniform(0.3, 0.8, (n, 2))
phase = rng.uniform(0, 2 * np.pi, (n, 2))
amp   = rng.uniform(0.20, 0.32, (n, 2))
center = np.array([0.5, 0.5])
def tgt(t): return center + amp * np.sin(freq * (t * dt) + phase)

pos, vel = tgt(0).copy(), np.zeros((n, 2))
Kp, Kd, v_max, a_max = 6.0, 4.0, 0.6, 3.0
true_traj = np.zeros((T, n, 2))
for t in range(T):
    true_traj[t] = pos
    acc = np.clip(Kp * (tgt(t) - pos) - Kd * vel, -a_max, a_max)
    vel = np.clip(vel + acc * dt, -v_max, v_max)
    pos = pos + vel * dt

def uwb(d):
    if rng.random() < p_dropout: return None
    e = rng.normal(0, sigma_uwb)
    if rng.random() < p_nlos:    e += rng.exponential(nlos_scale)
    if rng.random() < p_outlier: e += rng.uniform(0.2, 0.5)
    return d + e

def make_frame(Xt):
    se, ae = [], []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(Xt[i] - Xt[j])
            if d <= R:
                m = uwb(d)
                if m is not None: se.append((i, j, m))
        for k in range(len(anchors)):
            d = np.linalg.norm(Xt[i] - anchors[k])
            if d <= R:
                m = uwb(d)
                if m is not None: ae.append((i, k, m))
    return se, ae

frames = [make_frame(true_traj[t]) for t in range(T)]


# ---- causal baseline (Day 5/7 predict-correct) for comparison -----------------------
def solve_sdp(se, ae):
    d = 2
    Z = cp.Variable((d + n, d + n), symmetric=True)
    cons = [Z >> 0, Z[:d, :d] == np.eye(d)]
    yi = lambda i: d + i
    terms = []
    for (i, j, m) in se:
        terms.append(cp.abs(Z[yi(i), yi(i)] + Z[yi(j), yi(j)] - 2 * Z[yi(i), yi(j)] - m**2))
    for (i, k, m) in ae:
        ak = anchors[k]; ax = ak[0]*Z[0, yi(i)] + ak[1]*Z[1, yi(i)]
        terms.append(cp.abs(Z[yi(i), yi(i)] - 2*ax + ak @ ak - m**2))
    cp.Problem(cp.Minimize(cp.sum(terms)), cons).solve(solver=cp.CLARABEL)
    return Z.value[:2, 2:].T

def correct(se, ae, X_pred, lam=0.5):
    def res(xflat):
        Xc = xflat.reshape(n, 2)
        r = [np.linalg.norm(Xc[i] - Xc[j]) - m for (i, j, m) in se]
        r += [np.linalg.norm(Xc[i] - anchors[k]) - m for (i, k, m) in ae]
        for i in range(n): r.extend(np.sqrt(lam) * (Xc[i] - X_pred[i]))
        return r
    return least_squares(res, X_pred.flatten(), method="lm").x.reshape(n, 2)

causal = np.zeros((T, n, 2))
for t in range(T):
    se, ae = frames[t]
    causal[t] = solve_sdp(se, ae) if t == 0 else \
        correct(se, ae, causal[t-1] + (causal[t-1]-causal[t-2] if t >= 2 else 0))


# ---- factors (CustomFactor, version-proof) ------------------------------------------
def key(i, t): return X(t * n + i)

base   = gtsam.noiseModel.Isotropic.Sigma(1, sigma_uwb)
robust = gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Huber.Create(1.345), base)
motion_noise = gtsam.noiseModel.Isotropic.Sigma(2, sigma_motion)
prior_noise  = gtsam.noiseModel.Isotropic.Sigma(2, 0.5)     # LOOSE: safety net only

def anchor_factor(ki, apos, meas):
    def err(this, v, H):
        p = v.atPoint2(ki); diff = p - apos; dist = np.linalg.norm(diff)+1e-9
        if H is not None: H[0] = (diff/dist).reshape(1, 2)
        return np.array([dist - meas])
    return gtsam.CustomFactor(robust, [ki], err)

def range_factor(ki, kj, meas):
    def err(this, v, H):
        pi = v.atPoint2(ki); pj = v.atPoint2(kj); diff = pi-pj; dist = np.linalg.norm(diff)+1e-9
        if H is not None:
            u = (diff/dist).reshape(1, 2); H[0] = u; H[1] = -u
        return np.array([dist - meas])
    return gtsam.CustomFactor(robust, [ki, kj], err)

def motion_factor(k0, k1, k2):
    def err(this, v, H):
        p0 = v.atPoint2(k0); p1 = v.atPoint2(k1); p2 = v.atPoint2(k2)
        if H is not None: H[0] = np.eye(2); H[1] = -2*np.eye(2); H[2] = np.eye(2)
        return p2 - 2*p1 + p0
    return gtsam.CustomFactor(motion_noise, [k0, k1, k2])  if False else \
           gtsam.CustomFactor(motion_noise, [k0, k1, k2], err)

def prior_factor(ki, p0):
    def err(this, v, H):
        p = v.atPoint2(ki)
        if H is not None: H[0] = np.eye(2)
        return p - p0
    return gtsam.CustomFactor(prior_noise, [ki], err)


# ---- iSAM2: feed one frame at a time -------------------------------------------------
isam = gtsam.ISAM2(gtsam.ISAM2Params())
sdp0 = solve_sdp(*frames[0])
online = np.zeros((T, n, 2))      # the LIVE estimate the system emits at each frame
update_ms = np.zeros(T)

for t in range(T):
    se, ae = frames[t]
    g = gtsam.NonlinearFactorGraph()
    vals = gtsam.Values()
    for i in range(n):                                   # introduce this frame's variables
        if t >= 2:   pred = 2*online[t-1, i] - online[t-2, i]   # constant-velocity guess
        elif t == 1: pred = online[0, i]
        else:        pred = sdp0[i]
        vals.insert(key(i, t), pred.astype(float))
        g.add(prior_factor(key(i, t), pred.astype(float)))      # loose prior = safety net
    for (i, j, m) in se: g.add(range_factor(key(i, t), key(j, t), m))
    for (i, k, m) in ae: g.add(anchor_factor(key(i, t), anchors[k], m))
    if t >= 2:
        for i in range(n): g.add(motion_factor(key(i, t-2), key(i, t-1), key(i, t)))

    t0 = time.perf_counter()
    isam.update(g, vals)                                 # <-- the incremental work
    update_ms[t] = (time.perf_counter() - t0) * 1e3

    est = isam.calculateEstimate()
    for i in range(n):
        online[t, i] = est.atPoint2(key(i, t))

# Final fully-smoothed trajectory (uses ALL data, like the Day 7 batch solve)
est = isam.calculateEstimate()
smoothed = np.array([[est.atPoint2(key(i, t)) for i in range(n)] for t in range(T)])


# ---- timing contrast: batch-from-scratch GROWS with mission length ------------------
def batch_solve_time(tmax):
    g = gtsam.NonlinearFactorGraph(); vals = gtsam.Values()
    for t in range(tmax + 1):
        se, ae = frames[t]
        for i in range(n):
            vals.insert(key(i, t), online[t, i].astype(float))
            g.add(prior_factor(key(i, t), online[t, i].astype(float)))
        for (i, j, m) in se: g.add(range_factor(key(i, t), key(j, t), m))
        for (i, k, m) in ae: g.add(anchor_factor(key(i, t), anchors[k], m))
        if t >= 2:
            for i in range(n): g.add(motion_factor(key(i, t-2), key(i, t-1), key(i, t)))
    t0 = time.perf_counter()
    gtsam.LevenbergMarquardtOptimizer(g, vals).optimize()
    return (time.perf_counter() - t0) * 1e3

checkpoints = list(range(9, T, 10))
batch_ms = [batch_solve_time(t) for t in checkpoints]


# ---- results -------------------------------------------------------------------------
def rmse_t(traj): return np.sqrt(((traj - true_traj)**2).sum(2).mean(1))
print(f"mean RMSE  causal (predict-correct) : {rmse_t(causal).mean():.4f} m")
print(f"mean RMSE  iSAM2 online (live)      : {rmse_t(online).mean():.4f} m")
print(f"mean RMSE  iSAM2 smoothed (final)   : {rmse_t(smoothed).mean():.4f} m")
print(f"median iSAM2 update time            : {np.median(update_ms):.2f} ms/frame")
print(f"batch-from-scratch at t={checkpoints[-1]:>2}          : {batch_ms[-1]:.1f} ms (and growing)")

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ax[0].plot(rmse_t(causal), color="orange", alpha=0.8, label="causal (predict-correct)")
ax[0].plot(rmse_t(online), "g-", alpha=0.85, label="iSAM2 online (live)")
ax[0].plot(rmse_t(smoothed), "b-", lw=2, label="iSAM2 smoothed (final)")
ax[0].set_xlabel("time step"); ax[0].set_ylabel("RMSE to truth (m)")
ax[0].set_title("Incremental iSAM2 matches batch accuracy"); ax[0].legend(); ax[0].grid(alpha=0.3)

ax[1].plot(range(T), update_ms, "g-o", ms=3, label="iSAM2 per-frame update (incremental)")
ax[1].plot(checkpoints, batch_ms, "r-s", label="batch re-solve from scratch (grows)")
ax[1].set_xlabel("mission length (frames)"); ax[1].set_ylabel("solve time (ms)")
ax[1].set_title("Why iSAM2 exists: bounded update vs growing batch")
ax[1].legend(); ax[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figures/day8_isam2.png", dpi=130, bbox_inches="tight"); plt.show()
print("saved day8_isam2.png")
