"""
swarm_movie.py  -  the inner-child payoff: watch the WHOLE swarm fly on your estimator.

Replays the real gym-pybullet-drones flight (trajectory.npy) and runs your Day 8 iSAM2
localizer live, animating all n drones at once:
    green dots   = true (real Crazyflie flight)
    red circles  = your iSAM2 estimate chasing them
    gray lines   = UWB drone-to-drone links (flicker as they form / drop)
    blue lines   = anchor measurements
    black ^      = anchors
    red sticks   = per-drone estimate error (shrink as the estimator locks on)

Run in the estimator venv:  python swarm_movie.py
Out: swarm_movie.gif
Needs: gtsam, numpy, matplotlib, pillow
"""

import numpy as np
import gtsam
from gtsam.symbol_shorthand import X
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ---------------------------------------------------------------- world (same as Day 8)
true_traj = np.load("data/trajectory.npy")
T, n = true_traj.shape[0], true_traj.shape[1]
anchors = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], float)
R = 0.55
sigma_uwb, p_nlos, nlos_scale, p_outlier, p_dropout = 0.015, 0.15, 0.05, 0.03, 0.10
sigma_motion = 0.10
rng = np.random.default_rng(7)

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
            dd = np.linalg.norm(Xt[i] - Xt[j])
            if dd <= R:
                m = uwb(dd)
                if m is not None: se.append((i, j, m))
        for k in range(len(anchors)):
            dd = np.linalg.norm(Xt[i] - anchors[k])
            if dd <= R:
                m = uwb(dd)
                if m is not None: ae.append((i, k, m))
    return se, ae

frames = [make_frame(true_traj[t]) for t in range(T)]

# ---------------------------------------------------------------- iSAM2 (same as Day 8)
def key(i, t): return X(t * n + i)
base   = gtsam.noiseModel.Isotropic.Sigma(1, sigma_uwb)
robust = gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Huber.Create(1.345), base)
motion_noise = gtsam.noiseModel.Isotropic.Sigma(2, sigma_motion)
prior_noise  = gtsam.noiseModel.Isotropic.Sigma(2, 0.5)

def anchor_factor(ki, apos, meas):
    def err(this, v, H):
        p = v.atPoint2(ki); diff = p - apos; dist = np.linalg.norm(diff) + 1e-9
        if H is not None: H[0] = (diff / dist).reshape(1, 2)
        return np.array([dist - meas])
    return gtsam.CustomFactor(robust, [ki], err)

def range_factor(ki, kj, meas):
    def err(this, v, H):
        pi = v.atPoint2(ki); pj = v.atPoint2(kj); diff = pi - pj; dist = np.linalg.norm(diff) + 1e-9
        if H is not None:
            u = (diff / dist).reshape(1, 2); H[0] = u; H[1] = -u
        return np.array([dist - meas])
    return gtsam.CustomFactor(robust, [ki, kj], err)

def motion_factor(k0, k1, k2):
    def err(this, v, H):
        p0 = v.atPoint2(k0); p1 = v.atPoint2(k1); p2 = v.atPoint2(k2)
        if H is not None: H[0] = np.eye(2); H[1] = -2*np.eye(2); H[2] = np.eye(2)
        return p2 - 2*p1 + p0
    return gtsam.CustomFactor(motion_noise, [k0, k1, k2], err)

def prior_factor(ki, p0):
    def err(this, v, H):
        p = v.atPoint2(ki)
        if H is not None: H[0] = np.eye(2)
        return p - p0
    return gtsam.CustomFactor(prior_noise, [ki], err)

isam = gtsam.ISAM2(gtsam.ISAM2Params())
online = np.zeros((T, n, 2))
print("running iSAM2 over the swarm ...")
for t in range(T):
    se, ae = frames[t]
    g, vals = gtsam.NonlinearFactorGraph(), gtsam.Values()
    for i in range(n):
        if t >= 2:   pred = 2*online[t-1, i] - online[t-2, i]
        elif t == 1: pred = online[0, i]
        else:        pred = true_traj[0, i] + rng.normal(0, 0.05, 2)
        vals.insert(key(i, t), pred.astype(float))
        g.add(prior_factor(key(i, t), pred.astype(float)))
    for (i, j, m) in se: g.add(range_factor(key(i, t), key(j, t), m))
    for (i, k, m) in ae: g.add(anchor_factor(key(i, t), anchors[k], m))
    if t >= 2:
        for i in range(n): g.add(motion_factor(key(i, t-2), key(i, t-1), key(i, t)))
    isam.update(g, vals)
    est = isam.calculateEstimate()
    for i in range(n):
        online[t, i] = est.atPoint2(key(i, t))

# ---------------------------------------------------------------- animate the whole swarm
print("rendering swarm_movie.gif ...")
colors = plt.cm.tab20(np.linspace(0, 1, n))
fig, ax = plt.subplots(figsize=(7.5, 7.5))

def draw(t):
    ax.clear()
    Xt, Et = true_traj[t], online[t]
    se, ae = frames[t]

    for (i, j, _) in se:
        ax.plot([Xt[i, 0], Xt[j, 0]], [Xt[i, 1], Xt[j, 1]], color="gray", lw=0.5, alpha=0.35, zorder=1)
    for (i, k, _) in ae:
        ax.plot([Xt[i, 0], anchors[k, 0]], [Xt[i, 1], anchors[k, 1]], color="steelblue", lw=0.4, alpha=0.3, zorder=1)
    for tt in range(max(0, t - 16), t):                    # true-flight trails, per drone color
        for i in range(n):
            ax.plot(true_traj[tt:tt+2, i, 0], true_traj[tt:tt+2, i, 1], color=colors[i], alpha=0.18, zorder=1)
    for i in range(n):                                     # error sticks
        ax.plot([Xt[i, 0], Et[i, 0]], [Xt[i, 1], Et[i, 1]], color="red", alpha=0.5, lw=1, zorder=2)

    ax.scatter(anchors[:, 0], anchors[:, 1], c="k", marker="^", s=130, zorder=4, label="anchors")
    ax.scatter(Xt[:, 0], Xt[:, 1], c=colors, s=55, zorder=4, edgecolors="green", linewidths=1.5)
    ax.scatter(Et[:, 0], Et[:, 1], facecolors="none", edgecolors="red", s=95, zorder=4)

    rmse = np.sqrt(((Et - Xt) ** 2).sum(1).mean())
    ax.set_xlim(-0.12, 1.12); ax.set_ylim(-0.12, 1.12); ax.set_aspect("equal")
    ax.set_title(f"GPS-denied cooperative swarm localization  ({n} drones)\n"
                 f"frame {t}/{T}   swarm RMSE = {rmse:.3f} m   (green=true  red=iSAM2 estimate)")
    ax.grid(alpha=0.3)

anim = animation.FuncAnimation(fig, draw, frames=T, interval=100)
anim.save("figures/swarm_movie.gif", writer=animation.PillowWriter(fps=12))
print("saved swarm_movie.gif  -  go watch your swarm fly.")
plt.show()
