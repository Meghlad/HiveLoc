"""
Layer 2 (part B): vision bearings inside the LIVE iSAM2 estimator.

Part A proved the geometry on static graphs. This runs the same marriage in the
Day 8 real-time pipeline, on the real Crazyflie trajectory: every frame, each
drone's forward camera may detect up to B neighbors, and each detection becomes
a bearing factor alongside the UWB range factors. Same iSAM2, same Huber
robustness, same loose-prior safety net - one new factor type.

The bearing factor mirrors the repo's range_factor style (CustomFactor on
Point2 variables, hand Jacobians):
    residual = wrap(atan2(pj - pi) - measured)
    d(theta)/d(pi) = [ dy, -dx] / d^2      d(theta)/d(pj) = [-dy,  dx] / d^2
Huber-wrapped, because a detector will sometimes box the WRONG neighbor (a
data-association error is a bearing outlier - Day 7's lesson, new sensor).

Two conditions, each run range-only vs range+bearing on IDENTICAL measurements:
  R = 0.55  healthy radio   - bearings should shave RMSE and tighten covariance
  R = 0.35  degraded radio  - the marginal regime; bearings should prevent collapse

Also exported per frame: each drone's MARGINAL COVARIANCE trace. That number is
the estimator saying "how much do I trust drone i right now" - it's the signal
Layer 3's safety supervisor gates plans on.

Run:  python src/vision/layer2_isam2_bearing.py     (~2-3 min: 4 full iSAM2 runs)
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import gtsam
from gtsam.symbol_shorthand import X

# ----------------------------------------------------------------------
# World: identical to Day 8, plus the camera
# ----------------------------------------------------------------------
true_traj = np.load("data/trajectory.npy")
T, n = true_traj.shape[0], true_traj.shape[1]
anchors = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], float)
sigma_uwb, p_nlos, nlos_scale, p_outlier, p_dropout = 0.015, 0.15, 0.05, 0.03, 0.10
sigma_motion = 0.06

FOV = np.deg2rad(90)
R_CAM = 0.65
SIGMA_BRG = np.deg2rad(2.0)
P_DET = 0.9
P_WRONG_ID = 0.03        # data-association error: detector boxes the wrong drone
B_CAP = 2                # detections per drone per frame

rng = np.random.default_rng(7)

def uwb(d):
    if rng.random() < p_dropout: return None
    e = rng.normal(0, sigma_uwb)
    if rng.random() < p_nlos:    e += rng.exponential(nlos_scale)
    if rng.random() < p_outlier: e += rng.uniform(0.2, 0.5)
    return d + e

R_ANCHOR = 0.55          # ground anchors are mains-powered infrastructure: their
                         # radio range does NOT degrade with the inter-drone links.
                         # (Without this, R=0.35 has ~0 anchor edges/frame and the
                         # whole gauge drifts - ranges and drone-drone bearings are
                         # relative; only anchors pin translation. Verified.)

def make_range_frame(Xt, R):
    se, ae = [], []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(Xt[i] - Xt[j])
            if d <= R:
                m = uwb(d)
                if m is not None: se.append((i, j, m))
        for k in range(len(anchors)):
            d = np.linalg.norm(Xt[i] - anchors[k])
            if d <= R_ANCHOR:
                m = uwb(d)
                if m is not None: ae.append((i, k, m))
    return se, ae

# Camera headings: forward-mounted lens looks along the velocity vector.
vel = np.gradient(true_traj, axis=0)
headings = np.arctan2(vel[..., 1], vel[..., 0])          # [T, n]

def make_bearing_frame(t):
    """Per-frame detections: up to B_CAP nearest in-FOV neighbors per drone.
    P_WRONG_ID of detections keep the true pixel bearing but report the wrong
    NEIGHBOR ID - the factor then connects the wrong pair. Pure outlier."""
    Xt = true_traj[t]
    be = []
    for i in range(n):
        cands = []
        for j in range(n):
            if j == i: continue
            diff = Xt[j] - Xt[i]
            d = np.linalg.norm(diff)
            if d > R_CAM: continue
            theta = np.arctan2(diff[1], diff[0])
            rel = (theta - headings[t, i] + np.pi) % (2 * np.pi) - np.pi
            if abs(rel) <= FOV / 2:
                cands.append((d, j, theta))
        cands.sort()
        for d, j, theta in cands[:B_CAP]:
            if rng.random() >= P_DET: continue
            jj = j
            if rng.random() < P_WRONG_ID:            # misassociation
                jj = int(rng.integers(0, n - 1))
                jj += (jj >= i)                      # any drone but myself
            be.append((i, jj, theta + rng.normal(0, SIGMA_BRG)))
    return be

# ----------------------------------------------------------------------
# Factors: Day 8's, plus the one new type
# ----------------------------------------------------------------------
def key(i, t): return X(t * n + i)

base_r   = gtsam.noiseModel.Isotropic.Sigma(1, sigma_uwb)
robust_r = gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Huber.Create(1.345), base_r)
base_b   = gtsam.noiseModel.Isotropic.Sigma(1, SIGMA_BRG)
robust_b = gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Huber.Create(1.345), base_b)
motion_noise = gtsam.noiseModel.Isotropic.Sigma(2, sigma_motion)
prior_noise  = gtsam.noiseModel.Isotropic.Sigma(2, 0.5)

def anchor_factor(ki, apos, meas):
    def err(this, v, H):
        p = v.atPoint2(ki); diff = p - apos; dist = np.linalg.norm(diff) + 1e-9
        if H is not None: H[0] = (diff / dist).reshape(1, 2)
        return np.array([dist - meas])
    return gtsam.CustomFactor(robust_r, [ki], err)

def range_factor(ki, kj, meas):
    def err(this, v, H):
        pi = v.atPoint2(ki); pj = v.atPoint2(kj); diff = pi - pj
        dist = np.linalg.norm(diff) + 1e-9
        if H is not None:
            u = (diff / dist).reshape(1, 2); H[0] = u; H[1] = -u
        return np.array([dist - meas])
    return gtsam.CustomFactor(robust_r, [ki, kj], err)

def bearing_factor(ki, kj, meas):
    """Observer at ki saw target at kj along world bearing `meas`."""
    def err(this, v, H):
        pi = v.atPoint2(ki); pj = v.atPoint2(kj)
        dx, dy = pj[0] - pi[0], pj[1] - pi[1]
        d2 = dx * dx + dy * dy + 1e-12
        if H is not None:
            H[0] = np.array([[ dy / d2, -dx / d2]])   # d(theta)/d(pi)
            H[1] = np.array([[-dy / d2,  dx / d2]])   # d(theta)/d(pj)
        r = np.arctan2(dy, dx) - meas
        return np.array([(r + np.pi) % (2 * np.pi) - np.pi])   # wrap
    return gtsam.CustomFactor(robust_b, [ki, kj], err)

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

# ----------------------------------------------------------------------
# One full iSAM2 mission, with or without vision
# ----------------------------------------------------------------------
def run_isam2(range_frames, bearing_frames, use_vision):
    isam = gtsam.ISAM2(gtsam.ISAM2Params())
    online = np.zeros((T, n, 2))
    cov_tr = np.zeros((T, n))            # marginal covariance trace per drone
    for t in range(T):
        se, ae = range_frames[t]
        g, vals = gtsam.NonlinearFactorGraph(), gtsam.Values()
        for i in range(n):
            if t >= 2:   pred = 2*online[t-1, i] - online[t-2, i]
            elif t == 1: pred = online[0, i]
            else:        pred = true_traj[0, i] + rng.normal(0, 0.05, 2)
            vals.insert(key(i, t), pred.astype(float))
            g.add(prior_factor(key(i, t), pred.astype(float)))
        for (i, j, m) in se: g.add(range_factor(key(i, t), key(j, t), m))
        for (i, k, m) in ae: g.add(anchor_factor(key(i, t), anchors[k], m))
        if use_vision:
            for (i, j, th) in bearing_frames[t]:
                g.add(bearing_factor(key(i, t), key(j, t), th))
        if t >= 2:
            for i in range(n): g.add(motion_factor(key(i, t-2), key(i, t-1), key(i, t)))
        isam.update(g, vals)
        est = isam.calculateEstimate()
        for i in range(n):
            online[t, i] = est.atPoint2(key(i, t))
            cov_tr[t, i] = np.trace(isam.marginalCovariance(key(i, t)))
    return online, cov_tr

def rmse_t(traj):
    return np.sqrt(((traj - true_traj) ** 2).sum(2).mean(1))

# ----------------------------------------------------------------------
# Two radio conditions x two sensor suites, IDENTICAL measurements within a pair
# ----------------------------------------------------------------------
results = {}
for R in (0.55, 0.35):
    rng = np.random.default_rng(7)                    # same measurement world per R
    range_frames = [make_range_frame(true_traj[t], R) for t in range(T)]
    bearing_frames = [make_bearing_frame(t) for t in range(T)]
    n_det = np.mean([len(b) for b in bearing_frames])
    print(f"\n=== R={R}: {np.mean([len(f[0]) for f in range_frames]):.1f} range edges/frame, "
          f"{n_det:.1f} vision detections/frame ===")
    for use_vision in (False, True):
        tag = "range+bearing" if use_vision else "range-only    "
        t0 = time.perf_counter()
        online, cov_tr = run_isam2(range_frames, bearing_frames, use_vision)
        dtm = time.perf_counter() - t0
        e = rmse_t(online)
        sig = np.sqrt(cov_tr / 2).mean(1)             # mean per-axis marginal sigma
        results[(R, use_vision)] = (online, e, sig, cov_tr)
        print(f"  {tag}: mean RMSE {e.mean():.4f} m   worst frame {e.max():.4f} m   "
              f"mean marginal sigma {sig.mean():.4f} m   ({dtm:.0f}s)")

# ----------------------------------------------------------------------
# Figure: RMSE and estimator confidence, healthy vs degraded radio
# ----------------------------------------------------------------------
fig, ax = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
for col, R in enumerate((0.55, 0.35)):
    for use_vision, color, lbl in ((False, "tab:orange", "range-only"),
                                   (True, "tab:blue", f"+vision (B={B_CAP}/drone)")):
        _, e, sig, _ = results[(R, use_vision)]
        ax[0, col].plot(e, color=color, label=lbl)
        ax[1, col].plot(sig, color=color, label=lbl)
    ax[0, col].set_title(f"{'healthy' if R > 0.5 else 'DEGRADED'} radio  R={R}")
    ax[0, col].set_ylabel("live RMSE to truth (m)")
    ax[1, col].set_ylabel("mean marginal sigma (m)\n= estimator's own confidence")
    ax[1, col].set_xlabel("frame")
    for a in (ax[0, col], ax[1, col]):
        a.grid(alpha=0.3); a.legend()
fig.suptitle("Bearing factors in the live iSAM2 estimator - accuracy AND confidence\n"
             "(marginal sigma is the trust signal the Layer-3 supervisor gates on)")
plt.tight_layout()
plt.savefig("figures/layer2_isam2_bearing.png", dpi=130, bbox_inches="tight")
print("\nsaved layer2_isam2_bearing.png")

np.savez("data/layer2_isam2_results.npz",
         online_r055=results[(0.55, True)][0],  cov_r055=results[(0.55, True)][3],
         online_r035=results[(0.35, True)][0],  cov_r035=results[(0.35, True)][3],
         online_r055_ro=results[(0.55, False)][0], cov_r055_ro=results[(0.55, False)][3],
         online_r035_ro=results[(0.35, False)][0], cov_r035_ro=results[(0.35, False)][3],
         true_traj=true_traj)
print("saved layer2_isam2_results.npz (poses + marginal covariances -> Layer 3 supervisor input)")
