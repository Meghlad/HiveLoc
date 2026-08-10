"""
close_the_loop.py  -  the finale.

Your GPS-denied cooperative estimator (Day 8, iSAM2) computes where the swarm is,
frame by frame, from noisy inter-drone ranges. We take DRONE 0's estimated position
and stream it into ArduCopter SITL as external nav. The autopilot flies on the output
of the localizer you built from scratch.

SCOPE: your estimator localizes the whole swarm; SITL is one aircraft. So we feed
drone 0's estimate in as *its* position. Faithful end-to-end: real estimator -> real
autopilot, no pretending SITL is 12 drones.

TWO CLOCKS: the EKF wants ~20 Hz; the estimator ticks once per frame. We re-send the
last estimate several times per frame so the drone never starves. (Production splits
these into two threads; see the note near the send loop.)

Run in the ESTIMATOR venv (gtsam + pymavlink), with SITL running & external-nav params set:
    python src/flight/close_the_loop.py
"""

import time
import numpy as np
import gtsam
from gtsam.symbol_shorthand import X
from pymavlink import mavutil

# ----------------------------------------------------------------------
# 1. Load the real-flight swarm trajectory + rebuild the Day 8 measurement world.
#    (Same world as day8_isam2_traj.py. Trimmed here to the essentials.)
# ----------------------------------------------------------------------
true_traj = np.load("data/trajectory.npy")          # from gym-pybullet-drones export
T, n = true_traj.shape[0], true_traj.shape[1]
anchors = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], float)
R = 0.55
sigma_uwb, p_nlos, nlos_scale, p_outlier, p_dropout = 0.015, 0.15, 0.05, 0.03, 0.10
sigma_motion = 0.10                            # loosened for real (accelerating) motion
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

# ----------------------------------------------------------------------
# 2. GTSAM factors (same as Day 8) + iSAM2 setup
# ----------------------------------------------------------------------
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

# ----------------------------------------------------------------------
# 3. Connect to SITL and prepare to stream external nav
# ----------------------------------------------------------------------
# Use a DEDICATED port so we don't fight QGroundControl for 14550.
# In the MAVProxy console first run:  output add 127.0.0.1:14551
print("connecting to SITL udp:127.0.0.1:14551 ...")
master = mavutil.mavlink_connection("udpin:127.0.0.1:14551")
master.wait_heartbeat()
print(f"connected: system={master.target_system}\n")
boot = time.time()

def send_vision(north, east, down):
    usec = int((time.time() - boot) * 1e6)
    master.mav.vision_position_estimate_send(usec, north, east, down, 0.0, 0.0, 0.0)

def set_global_origin(lat_deg=-35.363261, lon_deg=149.165230, alt_m=584.0):
    """External nav gives a LOCAL origin only; ArduPilot needs a GLOBAL (lat/lon)
    origin before AHRS will set home (and let us arm). Coords are the SITL default
    home. The handler ignores repeats once an origin exists, so this is safe to
    call more than once."""
    lat, lon, alt_mm = int(lat_deg * 1e7), int(lon_deg * 1e7), int(alt_m * 1e3)
    master.mav.set_gps_global_origin_send(master.target_system, lat, lon, alt_mm)

# Map estimator's unit-square coords -> a few meters of NED so the drone actually moves.
SCALE = 5.0          # 1 estimator-unit = 5 meters of travel
ALT   = 2.0          # hold 2 m altitude (down = -2)
FRAME_HZ = 10        # estimator frame rate
RESENDS  = 2         # times we re-send each estimate (keeps EKF fed ~20 Hz)

# Prime the EKF with a few vision frames so it's actively using external nav, THEN
# give it a global origin so 'AHRS: waiting for home' clears.
print("priming external-nav stream, then setting global origin ...")
for _ in range(40):                 # ~2 s of vision at 20 Hz before we set origin
    send_vision(0.0, 0.0, -ALT)
    time.sleep(0.05)
for _ in range(5):                  # resend to survive packet loss (repeats are no-ops)
    set_global_origin()
    time.sleep(0.2)
print("sent SET_GPS_GLOBAL_ORIGIN -> AHRS can now set home\n")

# ----------------------------------------------------------------------
# 4. THE LOOP: step the estimator, feed drone 0's estimate to the autopilot
# ----------------------------------------------------------------------
print("closing the loop: iSAM2 estimate -> ArduPilot external nav")
print("(the drone now flies on YOUR localizer. Ctrl-C to stop.)\n")

DRONE = 0            # which swarm member we pilot in SITL
for t in range(T):
    se, ae = frames[t]
    g, vals = gtsam.NonlinearFactorGraph(), gtsam.Values()
    for i in range(n):
        if t >= 2:   pred = 2*online[t-1, i] - online[t-2, i]
        elif t == 1: pred = online[0, i]
        else:        pred = true_traj[0, i] + rng.normal(0, 0.05, 2)   # rough seed
        vals.insert(key(i, t), pred.astype(float))
        g.add(prior_factor(key(i, t), pred.astype(float)))
    for (i, j, m) in se: g.add(range_factor(key(i, t), key(j, t), m))
    for (i, k, m) in ae: g.add(anchor_factor(key(i, t), anchors[k], m))
    if t >= 2:
        for i in range(n): g.add(motion_factor(key(i, t-2), key(i, t-1), key(i, t)))

    isam.update(g, vals)                                  # incremental estimate
    est = isam.calculateEstimate()
    for i in range(n):
        online[t, i] = est.atPoint2(key(i, t))

    # --- feed DRONE 0's estimate to the autopilot as external nav (NED) ---
    px, py = online[t, DRONE]
    north, east, down = px * SCALE, py * SCALE, -ALT      # note the -ALT flip!
    # PRODUCTION NOTE: real systems run this send in a separate 20Hz thread that
    # re-sends the latest (north,east) continuously, decoupled from estimator speed.
    for _ in range(RESENDS):
        send_vision(north, east, down)
        time.sleep(1.0 / (FRAME_HZ * RESENDS))

    err = np.linalg.norm(online[t, DRONE] - true_traj[t, DRONE])
    print(f"t={t:3d}  drone0 est=({px:+.2f},{py:+.2f})  -> NED N={north:+.2f} E={east:+.2f}  "
          f"| est err={err:.3f}")

print("\nestimator finished all frames — now HOLDING last position and streaming forever.")
print("(the EKF needs a continuous source. Ctrl-C to stop.)")

# The nav heartbeat that never dies: keep whispering the final estimate so the
# EKF always has a live external-nav source (required to arm and to stay armed).
px, py = online[T-1, DRONE]
north, east, down = px * SCALE, py * SCALE, -ALT
while True:
    send_vision(north, east, down)
    time.sleep(0.05)                 # ~20 Hz, forever
