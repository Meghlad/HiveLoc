"""
Layer 2 (part A): vision bearings rescue a marginally-rigid range graph.

Day 2 showed the failure cliff: shrink the UWB radius and the range graph goes
FLOPPY - a flex exists, the SDP smears, localization is geometrically hopeless.
Day 2 also allowed reflections on purpose, because ranges *cannot* see a mirror
flip. Both failures are directional: ranges constrain "how far", never "which way".

A camera is the exact complement. When drone i detects drone j in frame, the
pixel column of the detection is a BEARING - it constrains "which way" and says
nothing about "how far". This script quantifies the marriage:

  PART A  Bearing measurement model - FOV-limited onboard camera, world-frame
          bearing = body bearing + known yaw (compass/IMU owns heading, same
          division of labor as EK3_SRC1_YAW=6 in the flight stack).
  PART B  Mixed rigidity - a bearing edge's rigidity row is perp() of the range
          edge's row. Same matrix, orthogonal constraint. Rank test unchanged.
  PART C  Bearings in the SDP - the killer detail: a bearing constraint is
          LINEAR in positions ((Xj-Xi) . n_hat = 0, plus the target-in-front
          ray inequality), so it drops into the Biswas-Ye relaxation with no
          extra relaxation gap of its own. The full estimate is SDP init ->
          whitened NLS polish, same division of labor as Days 4-8.
  PART D  THE DELIVERABLE - re-run the Day 2 sweep on the mixed graph:
          connectivity radius R x max detections per drone B. One plot answers
          "how many vision detections per frame make a marginally-rigid range
          graph well-conditioned?" - with conditioning MEASURED as the smallest
          eigenvalue of the stiffness matrix R^T R, not asserted.

Run:  python src/vision/layer2_bearing_phase_diagram.py     (sweep takes ~2 min)
"""

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# World + measurement parameters
# ----------------------------------------------------------------------
n = 12
anchors = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], float)
sigma_uwb = 0.02            # UWB range noise (m) - fixed; the sweep axis is geometry
FOV = np.deg2rad(90)        # camera field of view
R_CAM = 0.65                # visual detection range (cameras outrange marginal UWB)
SIGMA_BRG = np.deg2rad(2.0) # ~3 px at VGA with fx~300 - realistic detector jitter
P_DET = 0.9                 # per-candidate detection probability
W_BRG = 1.0                 # bearing term weight in the SDP objective


# ----------------------------------------------------------------------
# PART A - measurement generation (ranges reused from Day 2, bearings new)
# ----------------------------------------------------------------------
def generate_network(rng, R):
    """Ground truth + UWB range edges (Day 2's model, sigma fixed)."""
    X_true = rng.uniform(0.1, 0.9, size=(n, 2))
    se, ae = [], []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(X_true[i] - X_true[j])
            if d <= R:
                se.append((i, j, d + rng.normal(0, sigma_uwb)))
        for k in range(len(anchors)):
            d = np.linalg.norm(X_true[i] - anchors[k])
            if d <= R:
                ae.append((i, k, d + rng.normal(0, sigma_uwb)))
    return X_true, se, ae


def generate_bearings(rng, X_true, B):
    """Onboard-camera detections: drone i reports up to B world-frame bearings.

    Each drone gets a heading (where its camera points). Candidates are the
    neighbors inside the FOV cone and inside visual range, nearest first (big
    targets are the ones a detector actually finds). Each candidate is detected
    with prob P_DET and its bearing carries SIGMA_BRG of noise.

    NESTING GUARANTEE: detection coin-flips and noise are drawn for EVERY
    candidate regardless of the cap B, and the cap is applied to the candidate
    index. Re-running with the same rng state and a larger B returns a strict
    superset of detections - so a sweep over B compares nested measurement
    sets on the same network, and more vision can never mean less information.

    ASSUMPTION (stated, not hidden): yaw is known from compass/IMU, so a body-
    frame pixel bearing converts to a world-frame bearing. Heading is NOT part
    of the estimated state - same split as the flight stack, where the EKF's
    yaw source is separate from position.
    """
    be = []                              # (i, j, theta_world)
    headings = rng.uniform(0, 2 * np.pi, n)
    for i in range(n):
        cands = []
        for j in range(n):
            if j == i:
                continue
            diff = X_true[j] - X_true[i]
            d = np.linalg.norm(diff)
            if d > R_CAM:
                continue
            theta = np.arctan2(diff[1], diff[0])
            rel = (theta - headings[i] + np.pi) % (2 * np.pi) - np.pi
            if abs(rel) <= FOV / 2:
                cands.append((d, j, theta))
        cands.sort()                     # nearest (= largest in frame) first
        for rank_c, (d, j, theta) in enumerate(cands):
            hit = rng.random() < P_DET               # draw for EVERY candidate
            noise = rng.normal(0, SIGMA_BRG)         # (keeps rng state B-independent)
            if hit and rank_c < B:
                be.append((i, j, theta + noise))
    return be


# ----------------------------------------------------------------------
# PART B - mixed rigidity: ranges give diff rows, bearings give perp(diff) rows
# ----------------------------------------------------------------------
def perp(v):
    return np.array([-v[1], v[0]])


def mixed_rigidity(X, se, ae, be, d=2):
    """Anchored rigidity matrix with bearing rows appended.

    Range edge (i,j):   motions must preserve |Xi-Xj|      -> row uses  diff
    Bearing edge (i,j): motions must preserve atan2(Xj-Xi) -> row uses  perp(diff)
    Together one edge with both measurements pins the full relative position.
    Returns (rank, full_rank, lambda_min_of_stiffness). Rigid iff rank == d*n;
    lambda_min quantifies HOW rigid (stiffness of the weakest flex direction).
    """
    rows = []
    for (i, j, _) in se:
        row = np.zeros(d * n); diff = X[i] - X[j]
        row[d*i:d*i+d] = diff; row[d*j:d*j+d] = -diff
        rows.append(row)
    for (i, k, _) in ae:
        row = np.zeros(d * n)
        row[d*i:d*i+d] = X[i] - anchors[k]
        rows.append(row)
    for (i, j, _) in be:
        row = np.zeros(d * n); p = perp(X[j] - X[i])
        row[d*i:d*i+d] = -p; row[d*j:d*j+d] = p
        rows.append(row)
    if not rows:
        return 0, d * n, 0.0
    Rmat = np.array(rows)
    rank = np.linalg.matrix_rank(Rmat)
    eigs = np.linalg.eigvalsh(Rmat.T @ Rmat)     # stiffness spectrum
    return rank, d * n, max(eigs[0], 0.0)        # smallest = weakest direction


# ----------------------------------------------------------------------
# PART C - SDP with bearing terms (linear in position -> convex for free)
# ----------------------------------------------------------------------
def solve_sdp(se, ae, be):
    d = 2
    Z = cp.Variable((d + n, d + n), symmetric=True)
    cons = [Z >> 0, Z[:d, :d] == np.eye(d)]
    yi = lambda i: d + i
    terms = []
    for (i, j, m) in se:
        terms.append(cp.abs(Z[yi(i), yi(i)] + Z[yi(j), yi(j)]
                            - 2 * Z[yi(i), yi(j)] - m**2))
    for (i, k, m) in ae:
        ak = anchors[k]
        ax = ak[0] * Z[0, yi(i)] + ak[1] * Z[1, yi(i)]
        terms.append(cp.abs(Z[yi(i), yi(i)] - 2 * ax + ak @ ak - m**2))
    for (i, j, th) in be:
        # (Xj - Xi) must be parallel to (cos th, sin th): kill the perp component.
        # Positions live in Z[:2, 2+i] -> this is LINEAR in Z. No lifting needed.
        nx, ny = -np.sin(th), np.cos(th)
        resid = nx * (Z[0, yi(j)] - Z[0, yi(i)]) + ny * (Z[1, yi(j)] - Z[1, yi(i)])
        terms.append(W_BRG * cp.abs(resid))
        # A camera sees a RAY, not a line: the target is IN FRONT of the lens.
        # Also linear -> a hard convex constraint that a mirror flip violates.
        # This is what actually kills the reflection ambiguity ranges can't see.
        fwd = (np.cos(th) * (Z[0, yi(j)] - Z[0, yi(i)])
               + np.sin(th) * (Z[1, yi(j)] - Z[1, yi(i)]))
        cons.append(fwd >= 0)
    try:
        cp.Problem(cp.Minimize(cp.sum(terms)), cons).solve(solver=cp.CLARABEL)
    except cp.error.SolverError:
        return None
    if Z.value is None:
        return None
    return Z.value[:d, d:].T


def polish(se, ae, be, X_init):
    """Whitened range+bearing NLS from the SDP init - the ML refine.

    The division of labor mirrors Days 4-8: the SDP is the initialization-free
    convex START (it can't be fooled by a bad guess), the local polish is the
    maximum-likelihood FINISH (the SDP alone stays smeared whenever the
    relaxation is loose - rank(Z) > 2 - even on rigid graphs). Rigidity is the
    quantity that predicts whether the polish can lock in: on a floppy graph
    the flex direction is a flat valley and the polish wanders down it."""
    def res(xf):
        Xc = xf.reshape(n, 2)
        r = [(np.linalg.norm(Xc[i] - Xc[j]) - m) / sigma_uwb for (i, j, m) in se]
        r += [(np.linalg.norm(Xc[i] - anchors[k]) - m) / sigma_uwb for (i, k, m) in ae]
        for (i, j, th) in be:
            dv = Xc[j] - Xc[i]
            ang = (np.arctan2(dv[1], dv[0]) - th + np.pi) % (2 * np.pi) - np.pi
            r.append(ang / SIGMA_BRG)
        return r
    from scipy.optimize import least_squares
    return least_squares(res, X_init.flatten(), method="lm").x.reshape(n, 2)


def solve_full(se, ae, be):
    """SDP init -> NLS polish. Returns None if the SDP itself fails.
    With fewer measurements than unknowns the polish is underdetermined
    (LM refuses, rightly) - keep the SDP answer; the graph is hopeless anyway."""
    X_sdp = solve_sdp(se, ae, be)
    if X_sdp is None:
        return None
    if len(se) + len(ae) + len(be) < 2 * n:
        return X_sdp
    return polish(se, ae, be, X_sdp)


def rmse(X_est, X_true):
    return np.sqrt(np.mean(np.sum((X_est - X_true) ** 2, axis=1)))


# ----------------------------------------------------------------------
# Rescue demo: one marginal network, range-only vs +bearings
# ----------------------------------------------------------------------
def rescue_demo(R_marginal=0.35, B=2, max_seed=40):
    """Find a genuinely MARGINAL network (floppy range graph, bad RMSE) and show
    B bearings per drone snapping it. Marginal, not hopeless: Day 2's cliff is
    where a few missing constraints break everything - that's where a handful
    of bearings pays. (A graph missing 10 DOF needs radios, not cameras.)"""
    for seed in range(max_seed):
        rng = np.random.default_rng(seed)
        X_true, se, ae = generate_network(rng, R_marginal)
        rank0, full, lam0 = mixed_rigidity(X_true, se, ae, [])
        if rank0 >= full:                # range graph already rigid - not a demo
            continue
        be = generate_bearings(np.random.default_rng(7000 + seed), X_true, B)
        rank1, _, lam1 = mixed_rigidity(X_true, se, ae, be)
        if rank1 < full:                 # bearings didn't complete it - keep looking
            continue
        X0 = solve_full(se, ae, [])
        X1 = solve_full(se, ae, be)
        e0 = rmse(X0, X_true) if X0 is not None else np.nan
        e1 = rmse(X1, X_true) if X1 is not None else np.nan
        if e0 > 0.15 and e1 < e0 / 3:    # a rescue worth showing
            break
    else:
        raise RuntimeError("no marginal rescue case found - widen the search")

    print(f"\n=== RESCUE DEMO (R={R_marginal}, {len(se)} range edges) ===")
    print(f"range-only : rank {rank0}/{full}  lambda_min {lam0:.2e}  RMSE {e0:.3f} m")
    print(f"+{len(be)} bearings: rank {rank1}/{full}  lambda_min {lam1:.2e}  RMSE {e1:.3f} m")

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 6))
    for a, (Xe, be_used, ttl) in enumerate([
            (X0, [], f"range-only\nrank {rank0}/{full}, RMSE {e0:.3f} m"),
            (X1, be, f"+ {len(be)} vision bearings\nrank {rank1}/{full}, RMSE {e1:.3f} m")]):
        for (i, j, _) in se:
            ax[a].plot([X_true[i, 0], X_true[j, 0]], [X_true[i, 1], X_true[j, 1]],
                       color="gray", lw=0.6, alpha=0.4, zorder=1)
        for (i, j, _) in be_used:
            ax[a].annotate("", xy=X_true[j], xytext=X_true[i],
                           arrowprops=dict(arrowstyle="->", color="purple",
                                           lw=1.4, alpha=0.7), zorder=2)
        ax[a].scatter(anchors[:, 0], anchors[:, 1], c="k", marker="^", s=110, zorder=3)
        ax[a].scatter(X_true[:, 0], X_true[:, 1], c="green", s=45, zorder=3,
                      label="true")
        if Xe is not None:
            ax[a].scatter(Xe[:, 0], Xe[:, 1], facecolors="none", edgecolors="red",
                          s=90, zorder=3, label="SDP estimate")
            for i in range(n):
                ax[a].plot([X_true[i, 0], Xe[i, 0]], [X_true[i, 1], Xe[i, 1]],
                           "r-", alpha=0.5, lw=1, zorder=2)
        ax[a].set_title(ttl)
        ax[a].axis("equal"); ax[a].grid(alpha=0.3); ax[a].legend(loc="upper right")
    fig.suptitle("A floppy range graph, rescued by bearings (purple arrows = detections)")
    plt.tight_layout()
    plt.savefig("figures/layer2_bearing_rescue.png", dpi=130, bbox_inches="tight")
    print("saved layer2_bearing_rescue.png")


# ----------------------------------------------------------------------
# PART D - THE SWEEP: connectivity radius R x detections-per-drone B
# ----------------------------------------------------------------------
def sweep():
    print("\n=== SWEEP: R x max-detections-per-drone (takes ~2 min) ===")
    Rs = np.linspace(0.22, 0.40, 7)     # brackets Day 2's rigidity cliff
    Bs = np.arange(0, 5)                # 0 = Day 2 exactly; then add vision
    K = 6

    rmse_grid  = np.zeros((len(Bs), len(Rs)))
    rigid_grid = np.zeros((len(Bs), len(Rs)))
    lam_grid   = np.zeros((len(Bs), len(Rs)))
    det_grid   = np.zeros((len(Bs), len(Rs)))   # realized detections per frame

    # CONTROLLED COMPARISON: for each (R, trial) the network is generated ONCE
    # and shared across every B; the camera rng is re-seeded identically per B,
    # so (with the nesting guarantee above) B=2's detections are a strict subset
    # of B=3's. Any change along the B axis is *information*, not resampling noise.
    errs   = np.zeros((len(Bs), len(Rs), K))
    rigids = np.zeros((len(Bs), len(Rs), K))
    lams   = np.zeros((len(Bs), len(Rs), K))
    dets   = np.zeros((len(Bs), len(Rs), K))
    for a, R in enumerate(Rs):
        for t in range(K):
            rng_net = np.random.default_rng(1000 * a + t)
            X_true, se, ae = generate_network(rng_net, R)
            for b, B in enumerate(Bs):
                rng_cam = np.random.default_rng(50000 + 1000 * a + t)
                be = generate_bearings(rng_cam, X_true, B)
                rank, full, lam = mixed_rigidity(X_true, se, ae, be)
                rigids[b, a, t] = (rank == full)
                lams[b, a, t] = lam
                dets[b, a, t] = len(be)
                X_est = solve_full(se, ae, be)
                errs[b, a, t] = rmse(X_est, X_true) if X_est is not None else np.nan
        print(f"  R={R:.2f} column done")
    rmse_grid  = np.nanmedian(errs, axis=2)
    rigid_grid = rigids.mean(axis=2)
    lam_grid   = np.median(lams, axis=2)
    det_grid   = dets.mean(axis=2)

    # ---- the headline figure --------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    ext = [Rs[0], Rs[-1], Bs[0] - 0.5, Bs[-1] + 0.5]

    im0 = ax[0].imshow(rmse_grid, origin="lower", aspect="auto", extent=ext,
                       vmax=0.25, cmap="viridis")
    ax[0].set_title("median RMSE (m) - dark = good\nB=0 bottom row IS Day 2's cliff")
    ax[0].set_xlabel("UWB connectivity radius R")
    ax[0].set_ylabel("max vision detections per drone B")
    fig.colorbar(im0, ax=ax[0])

    im1 = ax[1].imshow(rigid_grid, origin="lower", aspect="auto", extent=ext,
                       vmin=0, vmax=1, cmap="RdYlGn")
    ax[1].set_title("fraction rigid (mixed range+bearing graph)")
    ax[1].set_xlabel("UWB connectivity radius R")
    ax[1].set_ylabel("max vision detections per drone B")
    fig.colorbar(im1, ax=ax[1])

    # money panel: pick the MARGINAL column (range-only rigidity ~ 50%)
    marg = int(np.argmin(np.abs(rigid_grid[0, :] - 0.5)))
    R_m = Rs[marg]
    ax2 = ax[2]
    ax2.plot(det_grid[:, marg], rmse_grid[:, marg], "o-", color="tab:blue",
             label="median RMSE (m)")
    ax2.set_xlabel(f"realized vision detections per frame  (at marginal R={R_m:.2f})")
    ax2.set_ylabel("median RMSE (m)", color="tab:blue")
    ax2.grid(alpha=0.3)
    ax3 = ax2.twinx()
    ax3.plot(det_grid[:, marg], rigid_grid[:, marg], "s--", color="tab:green",
             label="fraction rigid")
    ax3.plot(det_grid[:, marg], lam_grid[:, marg] / max(lam_grid[:, marg].max(), 1e-12),
             "^:", color="tab:purple", label="stiffness $\\lambda_{min}$ (norm.)")
    ax3.set_ylabel("fraction rigid / normalized stiffness")
    ax3.set_ylim(-0.05, 1.05)
    lines = ax2.get_lines() + ax3.get_lines()
    ax2.legend(lines, [l.get_label() for l in lines], loc="center right")
    ax2.set_title(f"How many detections fix a marginal graph?\n(R={R_m:.2f}: "
                  f"range-only rigid only {100*rigid_grid[0, marg]:.0f}% of the time)")

    plt.tight_layout()
    plt.savefig("figures/layer2_bearing_phase_diagram.png", dpi=130, bbox_inches="tight")
    print("saved layer2_bearing_phase_diagram.png")

    # ---- the quotable numbers -------------------------------------------------
    print(f"\nAt the marginal radius R={R_m:.2f}:")
    for b, B in enumerate(Bs):
        print(f"  B={B} (~{det_grid[b, marg]:4.1f} det/frame): "
              f"rigid {100*rigid_grid[b, marg]:3.0f}%   "
              f"RMSE {rmse_grid[b, marg]:.3f} m   "
              f"lambda_min {lam_grid[b, marg]:.2e}")
    fixed = np.where(rigid_grid[:, marg] >= 0.99)[0]
    if len(fixed):
        b = fixed[0]
        print(f"\n>>> {det_grid[b, marg]:.1f} detections/frame "
              f"(B={Bs[b]} per drone) turn a {100*rigid_grid[0, marg]:.0f}%-rigid "
              f"range graph into a 100%-rigid mixed graph, cutting median RMSE "
              f"{rmse_grid[0, marg]:.3f} m -> {rmse_grid[b, marg]:.3f} m.")
    np.savez("layer2_sweep_results.npz", Rs=Rs, Bs=Bs, rmse=rmse_grid,
             rigid=rigid_grid, lam=lam_grid, det=det_grid)
    print("saved layer2_sweep_results.npz")


if __name__ == "__main__":
    rescue_demo()
    sweep()
