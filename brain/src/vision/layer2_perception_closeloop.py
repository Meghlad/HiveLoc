"""
Layer 2 (part C, step 2): close the perception loop.

The Rust node (swarm-perception) turned rendered camera frames into raw bearing
observations via ONNX inference. This script feeds those REAL detections - not
oracle bearings - into the iSAM2 estimator, degraded-radio condition (R=0.35).

The new estimation problem detections force us to solve: DATA ASSOCIATION.
A detection is "a blob at bearing theta", with no idea which neighbor it is.
The estimator resolves identity from its own predicted swarm geometry:
  1. predict every drone's position (constant velocity, same as the iSAM2 seed)
  2. a detection matches the candidate whose predicted bearing is nearest,
     inside a gate; if TWO candidates are angularly close -> AMBIGUOUS, drop it
     (a wrong association is a lie; a dropped detection is only a missed meal)
  3. what survives becomes a Huber-robust bearing factor - so the occasional
     wrong match that slips through is absorbed, not obeyed (Day 7's lesson).

Runs compared on IDENTICAL range measurements:
  range-only            the degraded baseline
  oracle bearings       part B's synthetic detections (perfect identity)
  ONNX bearings         the full pipeline: pixels -> ort -> association -> factors

Also scores the detector itself against the withheld ground truth:
recall / precision / bearing RMS error / association purity.

Run:  python src/vision/layer2_perception_closeloop.py     (after swarm-perception)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import gtsam
from gtsam.symbol_shorthand import X

# ---- world: identical to layer2_isam2_bearing.py, degraded radio ------------
true_traj = np.load("data/trajectory.npy")
T, n = true_traj.shape[0], true_traj.shape[1]
anchors = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], float)
R_DD = 0.35              # degraded inter-drone radio
R_ANCHOR = 0.55          # mains-powered ground anchors keep their range
sigma_uwb, p_nlos, nlos_scale, p_outlier, p_dropout = 0.015, 0.15, 0.05, 0.03, 0.10
sigma_motion = 0.06

FOV = np.deg2rad(90)
R_CAM = 0.65
SIGMA_BRG = np.deg2rad(2.0)     # the ORACLE camera's noise (part B's model)
SIGMA_ONNX = np.deg2rad(0.3)    # the REAL pipeline, measured: 0.147 deg RMS + margin.
                                # Weighting vision at its true precision matters twice:
                                # correct matches pull 13x harder, and a WRONG match
                                # sits at ~100 sigma where Huber flattens it to nothing.
P_DET = 0.9
B_CAP = 2
BURN_IN = 5              # frames of range-only convergence before trusting vision
AMBIG = np.deg2rad(3)    # two candidates this close in angle -> drop detection
GATE_MIN = np.deg2rad(2)
GATE_MAX = np.deg2rad(25)

rng = np.random.default_rng(7)

def uwb(d):
    if rng.random() < p_dropout: return None
    e = rng.normal(0, sigma_uwb)
    if rng.random() < p_nlos:    e += rng.exponential(nlos_scale)
    if rng.random() < p_outlier: e += rng.uniform(0.2, 0.5)
    return d + e

def make_range_frame(Xt):
    se, ae = [], []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(Xt[i] - Xt[j])
            if d <= R_DD:
                m = uwb(d)
                if m is not None: se.append((i, j, m))
        for k in range(len(anchors)):
            d = np.linalg.norm(Xt[i] - anchors[k])
            if d <= R_ANCHOR:
                m = uwb(d)
                if m is not None: ae.append((i, k, m))
    return se, ae

vel = np.gradient(true_traj, axis=0)
headings = np.arctan2(vel[..., 1], vel[..., 0])

def make_oracle_bearings(t):
    """Part B's synthetic camera (perfect identity, 2 deg noise) - the upper bar."""
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
            if rng.random() < P_DET:
                be.append((i, j, theta + rng.normal(0, SIGMA_BRG)))
    return be

# ---- the ONNX detections, straight from the Rust node -----------------------
def load_onnx_bearings(path="bearings.jsonl"):
    obs = {}                                  # (t, observer) -> [bearing_world]
    with open(path) as f:
        for line in f:
            o = json.loads(line)
            obs.setdefault((o["t"], o["observer"]), []).append(o["bearing_world"])
    return obs

# ---- factors (same as part B) ------------------------------------------------
def key(i, t): return X(t * n + i)

base_r   = gtsam.noiseModel.Isotropic.Sigma(1, sigma_uwb)
robust_r = gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Huber.Create(1.345), base_r)
base_b   = gtsam.noiseModel.Isotropic.Sigma(1, SIGMA_BRG)
robust_b = gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Huber.Create(1.345), base_b)
base_bx  = gtsam.noiseModel.Isotropic.Sigma(1, SIGMA_ONNX)
robust_bx = gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Huber.Create(1.345), base_bx)
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

def bearing_factor(ki, kj, meas, noise=None):
    def err(this, v, H):
        pi = v.atPoint2(ki); pj = v.atPoint2(kj)
        dx, dy = pj[0] - pi[0], pj[1] - pi[1]
        d2 = dx * dx + dy * dy + 1e-12
        if H is not None:
            H[0] = np.array([[ dy / d2, -dx / d2]])
            H[1] = np.array([[-dy / d2,  dx / d2]])
        r = np.arctan2(dy, dx) - meas
        return np.array([(r + np.pi) % (2 * np.pi) - np.pi])
    return gtsam.CustomFactor(noise if noise is not None else robust_b, [ki, kj], err)

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

# ---- association: raw bearing -> which neighbor? -----------------------------
def associate(obs_list, i, X_pred, cov_prev):
    """Match each raw bearing to a predicted neighbor, or drop it.

    THE GATE IS SET BY THE ESTIMATOR'S OWN COVARIANCE, per pair:
        gate_ij = 3 * sigma_rel / d_pred  (+ detector noise floor)
    A fixed tight gate against a bad prior is CONFIRMATION BIAS: it rejects the
    correct detections (prediction is off by more than the gate) and keeps the
    coincidental ones that agree with the wrong geometry. Letting the marginal
    covariance widen the gate exactly when the estimate is untrustworthy - with
    the ambiguity margin as the safety - breaks that feedback loop.
    Returns [(j, bearing)] plus counters for the scoreboard."""
    matched, ambiguous, unmatched = [], 0, 0
    for th in obs_list:
        best, best_err, second_err, best_gate = None, np.inf, np.inf, GATE_MIN
        for j in range(n):
            if j == i:
                continue
            diff = X_pred[j] - X_pred[i]
            d = max(np.linalg.norm(diff), 0.10)
            if d > R_CAM * 1.3:                       # generous: prediction is fuzzy
                continue
            pred_th = np.arctan2(diff[1], diff[0])
            e = abs((th - pred_th + np.pi) % (2 * np.pi) - np.pi)
            if e < best_err:
                sig_rel = np.sqrt(cov_prev[i] + cov_prev[j])   # traces add
                gate = np.clip(3 * sig_rel / d + SIGMA_ONNX, GATE_MIN, GATE_MAX)
                best, second_err, best_err, best_gate = j, best_err, e, gate
            elif e < second_err:
                second_err = e
        if best is None or best_err > best_gate:
            unmatched += 1
        elif second_err - best_err < AMBIG:
            ambiguous += 1                            # two plausible ids: refuse
        else:
            matched.append((best, th))
    return matched, ambiguous, unmatched


# ---- track-based association: identity is EARNED, not claimed ----------------
# The naive matcher above COLLAPSES in the degraded condition (measured: 85% of
# its matches name the wrong drone). Why: relative prediction error ~0.2-0.3 m
# at target distances <= 0.65 m is 30-60 deg of bearing error, while candidate
# neighbors sit only 15-30 deg apart. Single-frame nearest-bearing is then
# near-random - and each wrong factor TIGHTENS the covariance around the wrong
# geometry, so the estimator becomes confidently wrong and the gate confirms it.
#
# The fix is temporal: a detection stream that really is drone j stays
# angularly consistent with j's predicted motion for its whole lifetime, while
# wrong candidates decorrelate as the swarm geometry evolves. So:
#   1. TRACK detections frame-to-frame in bearing space (targets move ~1.4
#      deg/frame - track continuity needs no identity at all)
#   2. score every candidate id against the track's whole history
#   3. admit factors only after the track is old enough AND the best id beats
#      the runner-up by a ratio test (SIFT-style, scale-free under a drifting
#      prior) - until then the track contributes NOTHING. A dropped detection
#      is a missed meal; a wrong association is poison.
TRACK_BASE_GATE = np.deg2rad(4)     # per-frame track continuity gate
TRACK_MAX_MISS = 2                  # frames a track survives unobserved
ADMIT_AGE = 5                       # matched frames before a track may vote
ADMIT_RATIO = 2.5                   # runner-up mean-sq residual must be this x worse
ADMIT_ABS = np.deg2rad(15)          # best candidate must actually fit this well
ADMIT_NOW = np.deg2rad(10)          # ...and fit TODAY, not just on average
RESID_CAP = np.deg2rad(25)          # per-frame residual cap in the cumulative score

class TrackBook:
    """Per-observer bearing tracks + per-candidate cumulative evidence."""
    def __init__(self, i):
        self.i = i
        self.tracks = []    # dict: brg, age, missed, score{j: sum e^2}, frames{j: count}

    def update(self, obs_list, preds):
        """One frame: continue/spawn tracks, accumulate evidence, emit
        admitted (j, bearing) factors."""
        i = self.i
        # 1. continuity: greedy nearest-bearing match of detections to tracks
        unused = list(obs_list)
        for tr in sorted(self.tracks, key=lambda tr: -tr["age"]):
            gate = TRACK_BASE_GATE * (1 + tr["missed"])
            best, best_e = None, gate
            for th in unused:
                e = abs((th - tr["brg"] + np.pi) % (2 * np.pi) - np.pi)
                if e < best_e:
                    best, best_e = th, e
            if best is None:
                tr["missed"] += 1
                tr["obs"] = None
            else:
                unused.remove(best)
                tr.update(brg=best, missed=0, obs=best)
                tr["age"] += 1
        self.tracks = [tr for tr in self.tracks if tr["missed"] <= TRACK_MAX_MISS]
        for th in unused:                              # newborn tracks
            self.tracks.append(dict(brg=th, age=1, missed=0, obs=th,
                                    score={}, frames={}))
        # 2. evidence: score this frame's residual against every candidate
        out = []
        for tr in self.tracks:
            th = tr["obs"]
            if th is None:
                continue
            for j in range(n):
                if j == i:
                    continue
                diff = preds[j] - preds[i]
                d = max(np.linalg.norm(diff), 0.10)
                if d > R_CAM * 1.3:
                    continue
                pred_th = np.arctan2(diff[1], diff[0])
                e = abs((th - pred_th + np.pi) % (2 * np.pi) - np.pi)
                e = min(e, RESID_CAP)
                tr["score"][j] = tr["score"].get(j, 0.0) + e * e
                tr["frames"][j] = tr["frames"].get(j, 0) + 1
            # 3. admission: age + absolute fit + ratio test
            if tr["age"] < ADMIT_AGE or not tr["score"]:
                continue
            means = {j: tr["score"][j] / tr["frames"][j] for j in tr["score"]}
            jbest = min(means, key=means.get)
            rest = [v for j, v in means.items() if j != jbest]
            if means[jbest] > ADMIT_ABS ** 2:
                continue
            if rest and min(rest) < ADMIT_RATIO * means[jbest]:
                continue
            # final sanity: TODAY'S residual must also fit - a track whose
            # history fits j but whose current bearing doesn't is mid-handoff
            # (crossing targets); sit this frame out.
            diff = preds[jbest] - preds[i]
            e_now = abs((th - np.arctan2(diff[1], diff[0]) + np.pi)
                        % (2 * np.pi) - np.pi)
            if e_now > ADMIT_NOW:
                continue
            out.append((jbest, th))
        return out

# ---- one mission -------------------------------------------------------------
def run(range_frames, vision, mode):
    """mode: 'none' | 'oracle' | 'onnx_naive' | 'onnx'  (onnx = track-based)"""
    books = [TrackBook(i) for i in range(n)]
    isam = gtsam.ISAM2(gtsam.ISAM2Params())
    online = np.zeros((T, n, 2))
    cov_tr = np.zeros((T, n))
    assoc_stats = np.zeros(3, int)                    # matched, ambiguous, unmatched
    assoc_log = []                                    # (t, i, j, bearing) for purity audit
    for t in range(T):
        se, ae = range_frames[t]
        g, vals = gtsam.NonlinearFactorGraph(), gtsam.Values()
        preds = np.zeros((n, 2))
        for i in range(n):
            if t >= 2:   pred = 2*online[t-1, i] - online[t-2, i]
            elif t == 1: pred = online[0, i]
            else:        pred = true_traj[0, i] + rng.normal(0, 0.05, 2)
            preds[i] = pred
            vals.insert(key(i, t), pred.astype(float))
            g.add(prior_factor(key(i, t), pred.astype(float)))
        for (i, j, m) in se: g.add(range_factor(key(i, t), key(j, t), m))
        for (i, k, m) in ae: g.add(anchor_factor(key(i, t), anchors[k], m))
        if mode == "oracle":
            for (i, j, th) in vision[t]:
                g.add(bearing_factor(key(i, t), key(j, t), th))
        elif mode == "onnx_naive" and t >= BURN_IN:   # single-frame matching (collapses)
            for i in range(n):
                obs_list = vision.get((t, i), [])
                matched, amb, unm = associate(obs_list, i, preds, cov_tr[t - 1])
                assoc_stats += [len(matched), amb, unm]
                for (j, th) in matched:
                    g.add(bearing_factor(key(i, t), key(j, t), th, robust_bx))
                    assoc_log.append((t, i, j, th))
        elif mode == "onnx" and t >= BURN_IN:         # track-based (identity earned)
            for i in range(n):
                obs_list = vision.get((t, i), [])
                admitted = books[i].update(obs_list, preds)
                assoc_stats += [len(admitted), 0, len(obs_list) - len(admitted)]
                for (j, th) in admitted:
                    g.add(bearing_factor(key(i, t), key(j, t), th, robust_bx))
                    assoc_log.append((t, i, j, th))
        if t >= 2:
            for i in range(n): g.add(motion_factor(key(i, t-2), key(i, t-1), key(i, t)))
        isam.update(g, vals)
        est = isam.calculateEstimate()
        for i in range(n):
            online[t, i] = est.atPoint2(key(i, t))
            cov_tr[t, i] = np.trace(isam.marginalCovariance(key(i, t)))
    return online, cov_tr, assoc_stats, assoc_log

def rmse_t(traj):
    return np.sqrt(((traj - true_traj) ** 2).sum(2).mean(1))

# ---- score the detector against the withheld truth ---------------------------
def score_detector():
    truth_by_img, det_by_img = {}, {}
    with open("frames/meta.jsonl") as f:
        for line in f:
            m = json.loads(line)
            truth_by_img[(m["t"], m["observer"])] = m["truth"]
    with open("bearings.jsonl") as f:
        for line in f:
            o = json.loads(line)
            det_by_img.setdefault((o["t"], o["observer"]), []).append(o)
    tp = fp = fn = 0
    brg_err = []
    for img_key, truth in truth_by_img.items():
        dets = det_by_img.get(img_key, [])
        used = set()
        for tr in truth:
            if not tr["rendered"]:
                continue
            best, best_d = None, 4.0                  # 4 px match radius
            for di, d in enumerate(dets):
                if di in used: continue
                dd = np.hypot(d["u"] - tr["u"], d["v"] - tr["v"])
                if dd < best_d:
                    best, best_d = di, dd
            if best is None:
                fn += 1
            else:
                used.add(best); tp += 1
                e = (dets[best]["bearing_world"] - tr["bearing_world"] + np.pi) \
                    % (2 * np.pi) - np.pi
                brg_err.append(np.degrees(e))
        fp += len(dets) - len(used)
    brg_err = np.array(brg_err)
    print("=== DETECTOR SCOREBOARD (vs withheld ground truth) ===")
    print(f"recall    : {tp/(tp+fn):.3f}   ({tp}/{tp+fn} rendered targets found)")
    print(f"precision : {tp/(tp+fp):.3f}   ({fp} false positives incl. clutter)")
    print(f"bearing error: RMS {np.sqrt((brg_err**2).mean()):.3f} deg   "
          f"p95 {np.percentile(np.abs(brg_err), 95):.3f} deg "
          f"(synthetic model assumed {np.degrees(SIGMA_BRG):.1f})")
    return tp, fp, fn, brg_err

# ---- go ----------------------------------------------------------------------
if __name__ == "__main__":
    tp, fp, fn, brg_err = score_detector()

    rng = np.random.default_rng(7)                    # same world as part B's R=0.35
    range_frames = [make_range_frame(true_traj[t]) for t in range(T)]
    oracle = [make_oracle_bearings(t) for t in range(T)]
    onnx_obs = load_onnx_bearings()
    n_onnx = sum(len(v) for v in onnx_obs.values())
    print(f"\nrange edges/frame: {np.mean([len(f[0]) for f in range_frames]):.1f}   "
          f"oracle det/frame: {np.mean([len(o) for o in oracle]):.1f}   "
          f"ONNX det/frame: {n_onnx/T:.1f}")

    truth_bearing = {}                    # (t, observer) -> [(bearing_true, id)]
    with open("frames/meta.jsonl") as f:
        for line in f:
            m = json.loads(line)
            truth_bearing[(m["t"], m["observer"])] = [
                (x["bearing_world"], x["id"]) for x in m["truth"] if x["rendered"]]

    def purity(assoc_log):
        """How many admitted factors named the RIGHT drone? (offline audit)"""
        pure = wrong = orphan = 0
        for (t, i, j, th) in assoc_log:
            cands = truth_bearing.get((t, i), [])
            if not cands:
                orphan += 1; continue
            tb, tid = min(cands,
                          key=lambda c: abs((th - c[0] + np.pi) % (2*np.pi) - np.pi))
            if abs((th - tb + np.pi) % (2 * np.pi) - np.pi) > np.deg2rad(1.5):
                orphan += 1               # detection was clutter/noise, not a drone
            elif tid == j:
                pure += 1
            else:
                wrong += 1
        tot = max(pure + wrong + orphan, 1)
        return 100 * pure / tot, 100 * wrong / tot, 100 * orphan / tot, tot

    runs = {}
    for mode, vision in (("none", None), ("oracle", oracle),
                         ("onnx_naive", onnx_obs), ("onnx", onnx_obs)):
        online, cov_tr, stats, assoc_log = run(range_frames, vision, mode)
        e = rmse_t(online)
        sig = np.sqrt(cov_tr / 2).mean(1)
        runs[mode] = (online, e, sig)
        extra = ""
        if mode.startswith("onnx"):
            p, w, o, tot = purity(assoc_log)
            extra = (f"   [{stats[0]} factors admitted, purity {p:.1f}% "
                     f"(wrong {w:.1f}%, clutter {o:.1f}%)]")
        print(f"{mode:10s}: mean RMSE {e.mean():.4f} m   worst {e.max():.4f} m   "
              f"mean sigma {sig.mean():.4f} m{extra}")

    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    for mode, color, lbl in (("none", "tab:orange", "range-only"),
                             ("oracle", "tab:green", "oracle bearings (perfect id)"),
                             ("onnx_naive", "tab:red", "ONNX naive assoc (collapses)"),
                             ("onnx", "tab:blue", "ONNX track-based assoc")):
        ax[0].plot(runs[mode][1], color=color, label=lbl)
        ax[1].plot(runs[mode][2], color=color, label=lbl)
    ax[0].set_title(f"live RMSE, degraded radio R={R_DD}")
    ax[0].set_xlabel("frame"); ax[0].set_ylabel("RMSE (m)")
    ax[1].set_title("estimator confidence (mean marginal sigma)")
    ax[1].set_xlabel("frame"); ax[1].set_ylabel("sigma (m)")
    for a in ax[:2]:
        a.grid(alpha=0.3); a.legend(fontsize=8)
    ax[2].hist(brg_err, bins=41, color="tab:blue", alpha=0.8)
    ax[2].set_title(f"ONNX bearing error (RMS {np.sqrt((brg_err**2).mean()):.2f} deg)\n"
                    f"recall {tp/(tp+fn):.2f}  precision {tp/(tp+fp):.2f}")
    ax[2].set_xlabel("bearing error (deg)"); ax[2].grid(alpha=0.3)
    fig.suptitle("The full perception loop: camera pixels -> ONNX (Rust) -> "
                 "data association -> iSAM2 factors")
    plt.tight_layout()
    plt.savefig("figures/layer2_perception_closeloop.png", dpi=130, bbox_inches="tight")
    print("\nsaved layer2_perception_closeloop.png")
