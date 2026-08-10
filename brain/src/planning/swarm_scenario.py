"""
swarm_scenario.py  -  the interactive, reactive 12-drone closed loop (headless).

Parts A/B built the estimator; Layer 3 built the "brain" (a language planner + the
Rust swarm-supervisor gate). But they never ran as a LOOP: close_the_loop.py replays
a fixed trajectory and flies only drone 0, and layer3_vlm_planner.py is one-shot.

This is the missing loop. A background thread runs the live world + iSAM2 estimator at
~10 Hz (drones fly toward their setpoints -> noisy UWB ranges -> incremental estimate
-> per-vehicle marginal covariance). You type instructions at a prompt; each one goes
    instruction -> make_plan() -> swarm-supervisor -> ACCEPT? -> new setpoints
so the swarm actually reconfigures under the same trust signal the supervisor gates on.

No SITL, no MAVLink: pure headless, runs in seconds. It is the shared substrate under a
later full-SITL flight demo AND the adversarial red/blue program (see the seams below:
`corrupt()` for attack injection, the swappable trust scalar for the eigenvalue gate).

Run (in the estimator venv, with the supervisor built):
    cargo build --release --manifest-path rust/Cargo.toml     # once
    python src/planning/swarm_scenario.py
Then type e.g.  `form a tight line along the north edge`  ·  `status`  ·  `radio 0.35`  ·  `quit`

The planner uses claude-opus-4-8 when ANTHROPIC_API_KEY is set, else a deterministic
offline geometric planner - identical Plan schema, identical supervisor path.
"""

import json
import threading
import time
from collections import deque

import numpy as np
import gtsam
from gtsam.symbol_shorthand import X

# The brain: import-safe (functions + constants only, __main__-guarded).
from layer3_vlm_planner import make_plan, supervise

# ----------------------------------------------------------------------
# World constants  (kept in step with layer2_isam2_bearing.py's model)
# ----------------------------------------------------------------------
n = 12
anchors = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], float)   # unit-square corners
R_ANCHOR = 0.55            # infra radios don't degrade with the inter-drone links
GEOFENCE = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]   # == supervisor default
TRUST_COV = 0.004          # cov-trace threshold the supervisor/planner call "trusted"

# UWB error model (identical to the estimator's world)
sigma_uwb, p_nlos, nlos_scale, p_outlier, p_dropout = 0.015, 0.15, 0.05, 0.03, 0.10
sigma_motion = 0.06
# Camera / bearing model
FOV, R_CAM, SIGMA_BRG, P_DET, P_WRONG_ID, B_CAP = np.deg2rad(90), 0.65, np.deg2rad(2.0), 0.9, 0.03, 2

# Reactive dynamics
TICK_HZ = 10.0
STEP_GAIN = 0.12           # fraction of the way to the setpoint per tick
V_MAX = 0.02               # speed cap (estimator units / tick) -> ~arena in 5 s

rng = np.random.default_rng(7)

# ----------------------------------------------------------------------
# Measurement generation (current true positions -> ranges + bearings)
# ----------------------------------------------------------------------
def uwb(d):
    if rng.random() < p_dropout: return None
    e = rng.normal(0, sigma_uwb)
    if rng.random() < p_nlos:    e += rng.exponential(nlos_scale)
    if rng.random() < p_outlier: e += rng.uniform(0.2, 0.5)
    return d + e

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

def make_bearing_frame(Xt, heading):
    """Up to B_CAP nearest in-FOV neighbors per drone; P_WRONG_ID misassociations."""
    be = []
    for i in range(n):
        cands = []
        for j in range(n):
            if j == i: continue
            diff = Xt[j] - Xt[i]
            d = np.linalg.norm(diff)
            if d > R_CAM: continue
            theta = np.arctan2(diff[1], diff[0])
            rel = (theta - heading[i] + np.pi) % (2 * np.pi) - np.pi
            if abs(rel) <= FOV / 2:
                cands.append((d, j, theta))
        cands.sort()
        for d, j, theta in cands[:B_CAP]:
            if rng.random() >= P_DET: continue
            jj = j
            if rng.random() < P_WRONG_ID:
                jj = int(rng.integers(0, n - 1)); jj += (jj >= i)
            be.append((i, jj, theta + rng.normal(0, SIGMA_BRG)))
    return be

def corrupt(se, ae, be):
    """ATTACK-INJECTION SEAM. Identity today; the Rust red-team middleware plugs in
    here (distance reduction/enlargement, Sybil ids, slow carry-off). Returns the
    (possibly) corrupted measurement lists."""
    return se, ae, be

# ----------------------------------------------------------------------
# Factors + iSAM2  (same forms the live estimator already uses)
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
    def err(this, v, H):
        pi = v.atPoint2(ki); pj = v.atPoint2(kj)
        dx, dy = pj[0] - pi[0], pj[1] - pi[1]
        d2 = dx * dx + dy * dy + 1e-12
        if H is not None:
            H[0] = np.array([[ dy / d2, -dx / d2]])
            H[1] = np.array([[-dy / d2,  dx / d2]])
        r = np.arctan2(dy, dx) - meas
        return np.array([(r + np.pi) % (2 * np.pi) - np.pi])
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
# Shared state between the sim thread (writer) and the REPL thread (reader)
# ----------------------------------------------------------------------
class Shared:
    def __init__(self):
        self.lock = threading.Lock()
        self.running = True
        # config the REPL can change live
        self.R = 0.55
        self.use_vision = True
        # true world + commands (REPL writes setpoints on ACCEPT)
        xs = np.linspace(0.2, 0.8, 4); ys = np.linspace(0.25, 0.75, 3)
        self.true_pos = np.array([[x, y] for y in ys for x in xs], float)   # 12 x 2
        self.setpoints = self.true_pos.copy()
        # latest published estimate snapshot (sim writes)
        self.snap = None            # dict: t, wall_ms, true_pos, est_pos, cov_trace
        # rolling history for the optional GIF, and the decisions audit log
        self.frames = deque(maxlen=4000)
        self.decisions = []

S = Shared()

# ----------------------------------------------------------------------
# The sim thread: reactive world + live iSAM2 estimator
# ----------------------------------------------------------------------
def sim_loop():
    isam = gtsam.ISAM2(gtsam.ISAM2Params())
    prev2 = prev1 = None                     # est at t-2, t-1 (for the CV seed)
    seed0 = None                             # rough initial seed (truth + noise, once)
    dt = 1.0 / TICK_HZ
    t = 0
    while True:
        with S.lock:
            if not S.running: break
            R = S.R; use_vision = S.use_vision
            sp = S.setpoints.copy()
            Xt = S.true_pos.copy()

        # 1. MOVE: speed-capped step toward each setpoint (replaces trajectory replay)
        delta = np.clip(STEP_GAIN * (sp - Xt), -V_MAX, V_MAX)
        Xt = Xt + delta
        heading = np.arctan2(delta[:, 1], delta[:, 0])    # forward-facing camera
        # for near-stationary drones keep a stable heading (avoid atan2(0,0) jitter)
        still = np.linalg.norm(delta, axis=1) < 1e-6
        heading[still] = 0.0

        # 2. SENSE (+ attack seam)
        se, ae = make_range_frame(Xt, R)
        be = make_bearing_frame(Xt, heading) if use_vision else []
        se, ae, be = corrupt(se, ae, be)

        # 3. ESTIMATE: one incremental iSAM2 update
        g, vals = gtsam.NonlinearFactorGraph(), gtsam.Values()
        for i in range(n):
            if prev2 is not None:   pred = 2 * prev1[i] - prev2[i]
            elif prev1 is not None: pred = prev1[i]
            else:                   pred = seed0[i] if seed0 is not None else Xt[i]
            vals.insert(key(i, t), pred.astype(float))
            g.add(prior_factor(key(i, t), pred.astype(float)))
        for (i, j, m) in se: g.add(range_factor(key(i, t), key(j, t), m))
        for (i, k, m) in ae: g.add(anchor_factor(key(i, t), anchors[k], m))
        for (i, j, th) in be: g.add(bearing_factor(key(i, t), key(j, t), th))
        if prev2 is not None:
            for i in range(n):
                g.add(motion_factor(key(i, t - 2), key(i, t - 1), key(i, t)))

        try:
            isam.update(g, vals)
            est = isam.calculateEstimate()
        except Exception as e:                # numerical hiccup: skip this frame
            print(f"\n[sim] iSAM2 update failed at t={t}: {e}", flush=True)
            time.sleep(dt); continue

        est_pos = np.zeros((n, 2))
        cov_trace = np.zeros(n)
        for i in range(n):
            est_pos[i] = est.atPoint2(key(i, t))
            # 4. TRUST SIGNAL (swappable scalar: trace today, min-eigenvalue later)
            try:
                cov_trace[i] = np.trace(isam.marginalCovariance(key(i, t)))
            except Exception:
                cov_trace[i] = 1.0            # can't recover -> declare untrusted

        if t == 0:                            # seed the CV predictor once
            seed0 = est_pos.copy()
        prev2, prev1 = prev1, est_pos

        with S.lock:
            S.true_pos = Xt
            S.snap = {"t": t, "wall_ms": int(time.time() * 1000),
                      "true_pos": Xt.copy(), "est_pos": est_pos, "cov_trace": cov_trace}
            S.frames.append((Xt.copy(), est_pos.copy(), sp.copy()))
        t += 1
        time.sleep(dt)

# ----------------------------------------------------------------------
# REPL helpers
# ----------------------------------------------------------------------
HELP = """commands:
  <instruction>     plan a formation, e.g. "form a tight line along the north edge",
                    "form a ring", "spread into a grid"
  status            per-drone estimate, covariance, trusted flag; RMSE + min spacing
  radio <R>         set inter-drone connectivity radius live (e.g. radio 0.35 degrades it)
  vision on|off     toggle the camera/bearing factors
  save              write swarm_scenario.gif + swarm_scenario_decisions.jsonl now
  help              this text
  quit | exit       stop, dump the decisions log, render the GIF"""

def get_snap(timeout=5.0):
    """Wait until the sim thread has published at least one estimate."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        with S.lock:
            if S.snap is not None:
                return dict(S.snap)
        time.sleep(0.05)
    return None

def cmd_status():
    snap = get_snap()
    if snap is None:
        print("(estimator warming up...)"); return
    est, cov, true = snap["est_pos"], snap["cov_trace"], snap["true_pos"]
    with S.lock:
        R, vis, sp = S.R, S.use_vision, S.setpoints.copy()
    trusted = cov <= TRUST_COV
    rmse = float(np.sqrt(((est - true) ** 2).sum(1).mean()))
    dmat = np.linalg.norm(true[:, None] - true[None, :], axis=2)
    min_sp = float(dmat[np.triu_indices(n, 1)].min())
    print(f"\n  t={snap['t']}   R={R}   vision={'on' if vis else 'off'}   "
          f"trusted {int(trusted.sum())}/{n}   RMSE(est,true)={rmse:.3f} m   "
          f"min spacing={min_sp:.3f}")
    print("  id   est (N,E)        cov_trace   trusted   setpoint (N,E)")
    for i in range(n):
        print(f"  {i:2d}  ({est[i,0]:+.2f},{est[i,1]:+.2f})    {cov[i]:.5f}    "
              f"{'yes' if trusted[i] else ' NO':>3}     ({sp[i,0]:+.2f},{sp[i,1]:+.2f})")

def cmd_instruction(instruction):
    snap = get_snap()
    if snap is None:
        print("(estimator not ready yet)"); return
    pos, cov = snap["est_pos"], snap["cov_trace"]
    print(f"planning: {instruction!r}  ({int((cov<=TRUST_COV).sum())} trusted)")
    plan, source = make_plan(instruction, pos, cov)
    plan["issued_unix_ms"] = int(time.time() * 1000)      # issue it *now* (freshness)

    verdict = supervise(plan, pos, cov)
    if verdict is None:
        print("[supervisor] binary not built - run: "
              "cargo build --release --manifest-path rust/Cargo.toml"); return
    code, out, err = verdict
    try:
        decision = json.loads(out)
    except json.JSONDecodeError:
        print(f"[supervisor] unparsable output:\n{out}\n{err}"); return

    accepted = decision.get("accepted", False)
    viols = decision.get("violations", [])
    tag = "ACCEPTED" if accepted else "REJECTED"
    print(f"  planner={source}   supervisor={tag}   ({len(plan['assignments'])} assignments)")
    if viols:
        kinds = ", ".join(sorted({v.get("kind", "?") for v in viols}))
        print(f"  violations: {kinds}")

    if accepted:
        with S.lock:
            for a in plan["assignments"]:
                S.setpoints[a["vehicle"]] = np.array(a["waypoint_ne"], float)
        print(f"  -> {len(plan['assignments'])} vehicles now moving to new waypoints")

    S.decisions.append({"ts_unix_ms": int(time.time() * 1000), "instruction": instruction,
                        "planner": source, "plan_id": plan.get("plan_id"),
                        "accepted": accepted, "violations": viols,
                        "assignments": plan["assignments"]})

def dump_decisions(path="swarm_scenario_decisions.jsonl"):
    with open(path, "w") as f:
        for d in S.decisions:
            f.write(json.dumps(d) + "\n")
    print(f"wrote {path} ({len(S.decisions)} decisions)")

def render_gif(path="figures/swarm_scenario.gif", max_frames=200):
    with S.lock:
        frames = list(S.frames)
    if not frames:
        print("no frames to render yet"); return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, PillowWriter
    except Exception as e:
        print(f"gif skipped (matplotlib unavailable: {e})"); return
    step = max(1, len(frames) // max_frames)
    frames = frames[::step]
    fig, ax = plt.subplots(figsize=(6, 6))
    def draw(fr):
        true, est, sp = fr
        ax.clear()
        fence = np.array(GEOFENCE + [GEOFENCE[0]])
        ax.plot(fence[:, 0], fence[:, 1], "k--", lw=1, alpha=0.5)
        ax.scatter(anchors[:, 0], anchors[:, 1], marker="s", c="k", s=60, label="anchors")
        ax.scatter(sp[:, 0], sp[:, 1], marker="x", c="tab:green", s=40, label="setpoints")
        ax.scatter(true[:, 0], true[:, 1], c="tab:blue", s=40, label="true")
        ax.scatter(est[:, 0], est[:, 1], facecolors="none", edgecolors="tab:orange",
                   s=70, label="estimate")
        ax.set_xlim(-0.1, 1.1); ax.set_ylim(-0.1, 1.1); ax.set_aspect("equal")
        ax.legend(loc="upper right", fontsize=7); ax.set_title("reactive 12-drone swarm")
    anim = FuncAnimation(fig, draw, frames=frames, interval=100)
    anim.save(path, writer=PillowWriter(fps=10))
    plt.close(fig)
    print(f"wrote {path} ({len(frames)} frames)")

# ----------------------------------------------------------------------
# Main: start the sim thread, run the interactive REPL
# ----------------------------------------------------------------------
def main():
    print(__doc__.strip().split("\n\n")[0])
    print("\nstarting live 12-drone estimator (background)...  type 'help' for commands.\n")
    sim = threading.Thread(target=sim_loop, daemon=True)
    sim.start()
    get_snap()                                # wait for the first estimate

    try:
        while True:
            try:
                line = input("> ").strip()
            except EOFError:
                break
            if not line:
                continue
            low = line.lower()
            if low in ("quit", "exit", "q"):
                break
            elif low == "help":
                print(HELP)
            elif low == "status":
                cmd_status()
            elif low == "save":
                dump_decisions(); render_gif()
            elif low.startswith("radio"):
                try:
                    val = float(line.split()[1])
                    with S.lock: S.R = val
                    print(f"inter-drone radius R = {val}")
                except (IndexError, ValueError):
                    print("usage: radio <R>   e.g. radio 0.35")
            elif low.startswith("vision"):
                parts = low.split()
                if len(parts) > 1 and parts[1] in ("on", "off"):
                    on = parts[1] == "on"
                    with S.lock: S.use_vision = on
                    print(f"vision {'on' if on else 'off'}")
                else:
                    print("usage: vision on|off")
            else:
                cmd_instruction(line)
    except KeyboardInterrupt:
        pass
    finally:
        with S.lock:
            S.running = False
        sim.join(timeout=2.0)
        print("\nstopping.")
        dump_decisions()
        render_gif()

if __name__ == "__main__":
    main()
