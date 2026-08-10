"""Figures for hive_assist.

One place for every plot so the visual language stays consistent across domains:
same palette, same recessive chrome, same "the story is in the labels, not the
decoration" bias.

Palette: categorical slots 1-2 (blue/orange) and the fixed status pair. Validated
for colour-vision deficiency at all-pairs CVD dE 24.7 / normal-vision dE 33.6,
comfortably clear of the floors. Status colours never carry meaning alone — every
status-coloured mark here also carries a direct numeric label.
"""

from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIGDIR = pathlib.Path(__file__).resolve().parents[1] / "figures"

# -- palette ---------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8880"
GRID = "#e4e3de"

SERIES_1 = "#2a78d6"     # blue
SERIES_2 = "#eb6834"     # orange
SERIES_3 = "#1baf7a"     # aqua — sub-3:1 on the light surface, so every use of
                         # it below also carries a visible direct label
GOOD = "#0ca30c"
CRITICAL = "#d03b3b"


def _style(ax):
    """Recessive chrome: the data is the ink, the axes are furniture."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_2, labelsize=8.5, length=3, width=0.8)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)


def _save(fig, name: str) -> pathlib.Path:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    path = FIGDIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# D1.2 — the null-space study
# --------------------------------------------------------------------------
def plot_nullspace_study(scn, rows, centroid) -> pathlib.Path:
    """Two panels sharing one category axis.

    Left  — dim ker(H) per configuration. The rank result, as an integer.
    Right — what that rank costs you in metres: the swarm centroid's position
            uncertainty resolved radially and tangentially about the anchor.

    Reading them together is the point. The left panel says "one direction is
    free"; the right panel says "and it is worth two kilometres of tangential
    position error". Rank alone never conveys the second part.
    """
    labels = [r["label"].strip() for r in rows]
    kdim = [r["kernel_dim"] for r in rows]
    keys = [r["config"] for r in rows]
    y = np.arange(len(rows))

    fig, (ax0, ax1) = plt.subplots(
        1, 2, figsize=(13.0, 4.6), sharey=True,
        gridspec_kw={"width_ratios": [1.0, 1.85], "wspace": 0.06},
    )
    fig.patch.set_facecolor(SURFACE)

    # -- left: dim ker(H) --------------------------------------------------
    colors = [GOOD if k == 0 else CRITICAL for k in kdim]
    ax0.barh(y, kdim, height=0.55, color=colors, zorder=3)
    for yi, k in zip(y, kdim):
        txt = "0  full rank" if k == 0 else f"{k}"
        ax0.text(k + 0.08, yi, txt, va="center", ha="left",
                 fontsize=8.5, color=INK if k else GOOD,
                 fontweight="bold" if k == 0 else "normal", zorder=4)

    ax0.set_xlim(0, 3.9)
    ax0.set_xticks([0, 1, 2, 3])
    ax0.set_xlabel("dim ker(H)   — free directions left in the estimate", fontsize=9)
    ax0.set_title("Gauge freedom", fontsize=10.5, color=INK, loc="left",
                  fontweight="bold", pad=10)
    ax0.xaxis.grid(True, color=GRID, linewidth=0.7, zorder=0)
    ax0.set_axisbelow(True)
    _style(ax0)

    ax0.set_yticks(y)
    ax0.set_yticklabels(labels, fontsize=8.5, color=INK)
    ax0.invert_yaxis()

    # -- right: what it costs in metres -----------------------------------
    rad = np.array([centroid[k]["radial_m"] for k in keys])
    tan = np.array([centroid[k]["tangential_m"] for k in keys])
    h = 0.30
    ax1.barh(y - h / 2 - 0.012, rad, height=h, color=SERIES_1,
             label="radial (toward the anchor)", zorder=3)
    ax1.barh(y + h / 2 + 0.012, tan, height=h, color=SERIES_2,
             label="tangential (around the anchor)", zorder=3)

    ax1.set_xscale("log")
    ax1.set_xlim(3e-3, 8e4)

    # the band where the weak 1 km prior is the only thing holding the estimate
    ax1.axvspan(1e1, 8e4, color=CRITICAL, alpha=0.055, zorder=0)
    ax1.text(6.5e4, -0.72, "prior-limited\n(unbounded)", fontsize=8,
             color=CRITICAL, ha="right", va="center", linespacing=1.35)
    ax1.axvline(1e1, color=CRITICAL, linewidth=0.8, alpha=0.35, zorder=1)

    # direct-label only the pairs that carry the story, always outside the mark
    for yi, key in enumerate(keys):
        if key in ("mesh_motion", "mesh_anchor_bearing", "two_anchors_range"):
            ax1.text(rad[yi] * 1.35, yi - h / 2 - 0.012, f"{rad[yi]*100:.1f} cm",
                     va="center", ha="left", fontsize=8, color=INK_2, zorder=4)
        if key in ("mesh_anchor_bearing", "two_anchors_range"):
            ax1.text(tan[yi] * 1.35, yi + h / 2 + 0.012, f"{tan[yi]*100:.1f} cm",
                     va="center", ha="left", fontsize=8, color=INK_2, zorder=4)
    ax1.tick_params(axis="y", length=0)

    ax1.set_xlabel("swarm-centroid position uncertainty, 1σ  [m, log scale]",
                   fontsize=9)
    ax1.set_title("What the free direction costs you", fontsize=10.5, color=INK,
                  loc="left", fontweight="bold", pad=10)
    ax1.xaxis.grid(True, color=GRID, linewidth=0.7, zorder=0)
    ax1.set_axisbelow(True)
    _style(ax1)

    leg = ax1.legend(loc="lower right", frameon=False, fontsize=8.5,
                     handlelength=1.1, borderpad=0.2)
    for t in leg.get_texts():
        t.set_color(INK_2)

    fig.suptitle(
        "A single surveyed anchor collapses the SE(2) gauge — but only if it "
        "supplies a heading",
        fontsize=12.5, color=INK, x=0.005, ha="left", y=1.045, fontweight="bold",
    )
    fig.text(
        0.005, 0.975,
        f"{scn.n_agents} agents × {scn.n_keyframes} keyframes, dim x = "
        f"{rows[0]['state_dim']}.  Ranges alone pin how FAR the swarm is from "
        f"the anchor, never which way round: the surviving null direction is "
        f"exactly rotation about the anchor.",
        fontsize=9, color=INK_2, ha="left", va="bottom",
    )
    return _save(fig, "d1_nullspace_study.png")


# --------------------------------------------------------------------------
# D1.3 — anchored iSAM2 over time
# --------------------------------------------------------------------------
def plot_anchored_isam2(t, err_anchored, err_pinned, sig_anchored, sig_pinned,
                        final_pinned, settle=20) -> pathlib.Path:
    """Position error and reported σ, anchored vs conventionally-pinned.

    Both runs are full rank. The figure's job is to show that being full rank is
    not the same as being attached to the world: after the cold start the pinned
    run walks steadily away while its own σ creeps up as √t, because its only
    link to an absolute position is a single prior that recedes into the past.
    """
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9.6, 6.6), sharex=True,
                                   gridspec_kw={"hspace": 0.16})
    fig.patch.set_facecolor(SURFACE)

    for ax in (ax0, ax1):
        # the cold start is real and belongs in the picture, but it is not the
        # phenomenon — mark it so the eye reads past it
        ax.axvspan(t[0], settle, color=INK_MUTED, alpha=0.07, zorder=0)

    ax0.plot(t, err_pinned, color=SERIES_2, linewidth=2.0,
             label="pinned — drone 0 declared the origin", zorder=3)
    ax0.plot(t, err_anchored, color=SERIES_1, linewidth=2.0,
             label="anchored — surveyed range + bearing", zorder=4)
    ax0.set_ylabel("swarm RMSE vs truth  [m]", fontsize=9)
    ax0.set_yscale("log")
    ax0.yaxis.grid(True, color=GRID, linewidth=0.7, zorder=1)
    ax0.set_axisbelow(True)
    _style(ax0)
    leg = ax0.legend(loc="upper right", frameon=False, fontsize=8.5,
                     handlelength=1.4)
    for txt in leg.get_texts():
        txt.set_color(INK_2)
    ax0.text(t[-1], err_pinned[-1], f"  {err_pinned[-1]:.2f} m", va="center",
             fontsize=8.5, color=SERIES_2, fontweight="bold")
    ax0.text(t[-1], err_anchored[-1], f"  {err_anchored[-1]:.3f} m", va="center",
             fontsize=8.5, color=SERIES_1, fontweight="bold")

    mid = len(t) // 2
    ax0.annotate(
        "drifts, monotonically", xy=(t[mid], err_pinned[mid]),
        xytext=(t[mid] - 34, err_pinned[mid] * 3.1), fontsize=8.5,
        color=SERIES_2,
        arrowprops=dict(arrowstyle="-", color=SERIES_2, linewidth=0.9,
                        alpha=0.7, shrinkB=3),
    )
    ax0.set_ylim(err_anchored.min() * 0.30, err_pinned.max() * 1.6)
    ax0.text(t[-2], err_anchored[mid] * 0.46, "flat for the whole mission",
             fontsize=8.5, color=SERIES_1, ha="right", va="top")
    ax0.text(settle - 1.5, err_pinned.max() * 0.62,
             "cold start:\nrelative-only graph is\nnearly indeterminate",
             fontsize=7.8, color=INK_MUTED, ha="right", va="top", linespacing=1.4)

    # -- consistency: does the estimator know how wrong it is? -------------
    ratio_pinned = err_pinned / np.maximum(sig_pinned, 1e-9)
    ratio_anchored = err_anchored / np.maximum(sig_anchored, 1e-9)

    ax1.axhline(1.0, color=INK_MUTED, linewidth=1.0, linestyle=(0, (4, 3)),
                zorder=2)
    ax1.text(t[-2], 0.88, "consistent: error matches the reported σ",
             fontsize=8, color=INK_MUTED, va="top", ha="right")
    ax1.plot(t, ratio_pinned, color=SERIES_2, linewidth=2.0, zorder=3)
    ax1.plot(t, ratio_anchored, color=SERIES_1, linewidth=2.0, zorder=4)
    ax1.set_xlabel("keyframe", fontsize=9)
    ax1.set_ylabel("actual error ÷ reported σ", fontsize=9, linespacing=1.4)
    ax1.yaxis.grid(True, color=GRID, linewidth=0.7, zorder=1)
    ax1.set_axisbelow(True)
    _style(ax1)
    ax1.set_ylim(0, max(ratio_pinned[settle:].max() * 1.35, 2.2))
    ax1.text(t[-1], ratio_pinned[-1], f"  {ratio_pinned[-1]:.1f}×",
             fontsize=8.5, color=SERIES_2, va="center", fontweight="bold")
    ax1.text(t[-1], ratio_anchored[-1], f"  {ratio_anchored[-1]:.1f}×",
             fontsize=8.5, color=SERIES_1, va="center", fontweight="bold")
    ax1.text(t[mid], ratio_pinned[settle:].max() * 1.16,
             "pinned becomes OVERCONFIDENT — σ plateaus while the error keeps "
             "growing.\nThis is what the supervisor's covariance gate would be "
             "reading.",
             fontsize=8.5, color=SERIES_2, ha="center", va="top", linespacing=1.5)

    fig.suptitle(
        "A convention removes the singularity. Only a survey removes the drift.",
        fontsize=12.5, color=INK, x=0.005, ha="left", y=1.055, fontweight="bold",
    )
    fig.text(
        0.005, 0.965,
        f"Both runs are full rank at every keyframe. The pinned one still ends "
        f"{final_pinned:.2f} m from truth while reporting σ = {sig_pinned[-1]:.2f} m "
        f"— it does not merely drift,\nit stops knowing that it is drifting. Its "
        f"only absolute reference is one prior at t = 0, and relative "
        f"measurements have nothing to hold still against.",
        fontsize=9, color=INK_2, ha="left", va="bottom", linespacing=1.5,
    )
    return _save(fig, "d1_anchored_isam2.png")


# --------------------------------------------------------------------------
# D3 — standoff approach
# --------------------------------------------------------------------------
def plot_standoff_approach(x_tac, start, task: str = "inspection"
                           ) -> pathlib.Path:
    """Plan view of the dispatch, plus the cross-track error that defines it.

    The left panel's job is to show what the plan explicitly ruled out: the
    trajectory ends on a perimeter *station*, at a task-appropriate bearing,
    and never converges on the target coordinate itself.
    """
    from hive.standoff import (
        TASKS,
        GuidanceLimits,
        dispatch_on_signal,
        standoff_station,
    )

    x = np.asarray(x_tac, dtype=float)
    lim = GuidanceLimits()
    d = dispatch_on_signal(0, start, x, task)
    r = d.result

    fig, (ax0, ax1) = plt.subplots(
        1, 2, figsize=(13.2, 5.6),
        gridspec_kw={"width_ratios": [1.25, 1.0], "wspace": 0.22})
    fig.patch.set_facecolor(SURFACE)

    # -- left: plan view ---------------------------------------------------
    g = TASKS[task]
    ring = plt.Circle(tuple(x), g.standoff_m, fill=False, linestyle=(0, (5, 4)),
                      linewidth=1.0, color=INK_MUTED, zorder=2)
    ax0.add_patch(ring)
    ax0.text(x[0], x[1] + g.standoff_m + 0.9, f"standoff perimeter  d_s = "
             f"{g.standoff_m:.0f} m", fontsize=8, color=INK_MUTED, ha="center")

    # the other tasks' stations, to show the geometry is per-task
    for name in TASKS:
        s, _ = standoff_station(x, name)
        if name == task:
            continue
        ax0.plot(*s, marker="o", markersize=5, color=INK_MUTED, alpha=0.55,
                 zorder=3)
        ax0.text(s[0], s[1] - 1.5, name, fontsize=7.5, color=INK_MUTED,
                 ha="center")

    ref = d.path.polyline(400)
    ax0.plot(ref[:, 0], ref[:, 1], color=SERIES_2, linewidth=4.5, alpha=0.85,
             label=f"Dubins reference ({d.path.word})", zorder=4)
    ax0.plot(r.pos[:, 0], r.pos[:, 1], color=SERIES_1, linewidth=1.8,
             label="flown trajectory", zorder=5)

    ax0.plot(start[0], start[1], marker="o", markersize=8, color=SERIES_1,
             markeredgecolor=SURFACE, markeredgewidth=2, zorder=6)
    ax0.annotate("go-signal received here", xy=tuple(start[:2]),
                 xytext=(start[0] + 2.5, start[1] + 3.4), fontsize=8.5,
                 color=INK_2, ha="left",
                 arrowprops=dict(arrowstyle="-", color=INK_MUTED,
                                 linewidth=0.9, shrinkB=6))

    ax0.plot(*d.station, marker="o", markersize=10, color=SERIES_1,
             markeredgecolor=SURFACE, markeredgewidth=2, zorder=6)
    ax0.text(d.station[0] - 0.8, d.station[1] - 2.2, "standoff station",
             fontsize=8.5, color=INK, ha="center", fontweight="bold")

    ax0.plot(*x, marker="X", markersize=13, color=INK, zorder=6)
    ax0.text(x[0] + 1.2, x[1] + 1.0, "X_tac (target)", fontsize=9, color=INK,
             va="bottom", fontweight="bold")

    ax0.annotate(
        "", xy=tuple(x), xytext=tuple(d.station),
        arrowprops=dict(arrowstyle="-", color=INK_MUTED, linewidth=0.9,
                        linestyle=(0, (2, 2))))
    mid = 0.5 * (x + d.station)
    ax0.text(mid[0] - 1.0, mid[1] - 2.0,
             f"{np.linalg.norm(d.station - x):.0f} m — never closed",
             fontsize=8, color=INK_MUTED, ha="center")

    ax0.set_aspect("equal")
    ax0.set_xlabel("TacFrame x  [m]", fontsize=9)
    ax0.set_ylabel("TacFrame y  [m]", fontsize=9)
    ax0.set_title(f"Event-triggered approach — {task}", fontsize=10.5,
                  color=INK, loc="left", fontweight="bold", pad=10)
    ax0.grid(True, color=GRID, linewidth=0.7)
    ax0.set_axisbelow(True)
    _style(ax0)
    leg = ax0.legend(loc="upper left", frameon=False, fontsize=8.5,
                     handlelength=1.6)
    for t_ in leg.get_texts():
        t_.set_color(INK_2)

    # -- right: the convergence claim -------------------------------------
    t = np.arange(len(r.cross_track)) * lim.dt
    switch = int(np.argmax(r.gamma >= 1.0 - 1e-9)) if (r.gamma >= 1.0 - 1e-9).any() \
        else len(r.gamma) - 1
    # the follow/capture handover is where gamma stops advancing
    adv = np.flatnonzero(np.diff(r.gamma) <= 1e-12)
    if adv.size:
        switch = int(adv[0]) + 1
    t_switch = t[min(switch, len(t) - 1)]

    ax1.axvspan(t_switch, t[-1], color=INK_MUTED, alpha=0.07, zorder=0)
    ax1.axhline(0.0, color=INK_MUTED, linewidth=1.0, linestyle=(0, (4, 3)),
                zorder=2)
    ax1.plot(t, r.cross_track, color=SERIES_1, linewidth=2.0, zorder=3)
    ax1.fill_between(t, 0, r.cross_track, color=SERIES_1, alpha=0.10, zorder=1)

    ax1.set_xlabel("time since go-signal  [s]", fontsize=9)
    ax1.set_ylabel("cross-track error  e  [m]", fontsize=9)
    ax1.set_title("Cross-track error → 0 in both phases", fontsize=10.5,
                  color=INK, loc="left", fontweight="bold", pad=10)
    ax1.grid(True, color=GRID, linewidth=0.7)
    ax1.set_axisbelow(True)
    _style(ax1)

    peak = float(np.abs(r.cross_track).max())
    lo, hi = float(r.cross_track.min()), float(r.cross_track.max())
    pad = max(0.18 * (hi - lo), 0.05)
    ax1.set_ylim(lo - pad, hi + pad * 2.4)

    ax1.text(t_switch * 0.5, hi + pad * 1.5, "path following",
             fontsize=8.5, color=INK_2, ha="center")
    ax1.text((t_switch + t[-1]) / 2, hi + pad * 1.5, "station capture",
             fontsize=8.5, color=INK_2, ha="center")
    ax1.axvline(t_switch, color=INK_MUTED, linewidth=0.9, zorder=2)

    ax1.text(t[-1], r.cross_track[-1], f"  {abs(r.cross_track[-1]):.3f} m",
             fontsize=8.5, color=SERIES_1, va="center", fontweight="bold")
    ax1.text(0.03, 0.06,
             f"peak |e| {peak * 100:.0f} cm  →  "
             f"{abs(r.cross_track[-1]) * 100:.1f} cm on station\n"
             f"measured against the nearest point on the path, in both phases",
             transform=ax1.transAxes, fontsize=8, color=INK_2, linespacing=1.5)

    fig.suptitle(
        "Dispatch converges to a standoff station — not to the target",
        fontsize=12.5, color=INK, x=0.005, ha="left", y=1.10,
        fontweight="bold")
    fig.text(
        0.005, 1.005,
        f"One agent, one go-signal, one curvature-bounded path "
        f"({d.path.word}, R = {d.path.radius:.1f} m — sized to the vehicle's own "
        f"v/ω limit, not chosen). No impact-time consensus, no simultaneity\n"
        f"coupling, no terminal homing: the vehicle stops "
        f"{np.linalg.norm(d.station - x):.0f} m from X_tac at the bearing the "
        f"task asked for, and every emitted setpoint passes the same supervisor "
        f"gate as everything else.",
        fontsize=9, color=INK_2, ha="left", va="bottom", linespacing=1.5)
    return _save(fig, "d3_standoff_approach.png")


# --------------------------------------------------------------------------
# D4.3 — safe hold under packet loss
# --------------------------------------------------------------------------
def _stacked_labels(ax, entries, x, gap_frac=0.062):
    """Direct-label several series at the same x, pushing apart any that
    coincide. Two of these configurations sit on exactly the same value, so
    without this their labels overprint into noise."""
    lo, hi = ax.get_ylim()
    span = hi - lo
    items = sorted(entries, key=lambda e: e[0])
    placed = []
    for y, text, color in items:
        yy = y
        for py in placed:
            if abs(yy - py) < gap_frac * span:
                yy = py + gap_frac * span
        placed.append(yy)
        ax.annotate(text, xy=(x, y), xytext=(x + 1.2, yy), fontsize=8.5,
                    color=color, va="center", fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=color, linewidth=0.8,
                                    alpha=0.55, shrinkA=0, shrinkB=2))


def plot_safe_hold(res) -> pathlib.Path:
    """Two panels, because safe-hold is two claims that trade against each other.

    Left  — does the vehicle ever get commanded to move further than it can fly?
    Right — does the mission still finish?

    Either one alone is easy to satisfy and useless. A planner with no gate
    always finishes and lunges; a gate with no re-planning never lunges and
    never arrives. Only the third configuration does both, which is why the
    figure refuses to show just one axis.
    """
    from hive.loss_model import CONFIGS

    step = res["step"]
    loss = np.asarray(res["loss"]) * 100.0
    colors = {"stale_ungated": SERIES_2, "stale_gated": SERIES_3,
              "replan_gated": SERIES_1}
    short = {"stale_ungated": "stale stream, no slew gate",
             "stale_gated": "stale stream, slew gate",
             "replan_gated": "re-plan on hold, slew gate"}
    # secondary encoding: two of these lie on identical values, so colour alone
    # would hide one of them completely
    styles = {"stale_ungated": ("-", "o"), "stale_gated": ((0, (5, 3)), "s"),
              "replan_gated": ("-", "^")}

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13.0, 5.0),
                                   gridspec_kw={"wspace": 0.30})
    fig.patch.set_facecolor(SURFACE)

    # -- left: commanded jump ---------------------------------------------
    ax0.axhspan(step, 1e3, color=CRITICAL, alpha=0.055, zorder=0)
    ax0.axhline(step, color=CRITICAL, linewidth=1.0, alpha=0.45, zorder=2)
    ax0.text(0.6, step * 1.14,
             f"{step:.1f} m = one tick of legal motion", fontsize=8,
             color=CRITICAL, ha="left", va="bottom")

    ends0 = []
    for key in CONFIGS:
        y = np.array([r["max_jump"] for r in res[key]])
        ls, mk = styles[key]
        ax0.plot(loss, y, color=colors[key], linewidth=2.2, linestyle=ls,
                 marker=mk, markersize=5, markeredgecolor=SURFACE,
                 markeredgewidth=1.4, zorder=4)
        ends0.append((float(y[-1]), short[key], colors[key]))

    ax0.set_xlabel("packet loss  [%]", fontsize=9)
    ax0.set_ylabel("worst commanded jump  [m]", fontsize=9)
    ax0.set_title("Does it ever lunge?", fontsize=10.5, color=INK, loc="left",
                  fontweight="bold", pad=10)
    ax0.set_xlim(-1, loss[-1] + 16)
    ax0.set_ylim(0, max(r["max_jump"] for r in res["stale_ungated"]) * 1.25)
    ax0.grid(True, color=GRID, linewidth=0.7)
    ax0.set_axisbelow(True)
    _style(ax0)
    _stacked_labels(ax0, ends0, loss[-1])

    # -- right: mission progress -------------------------------------------
    ends1 = []
    for key in CONFIGS:
        y = np.array([r["progress"] for r in res[key]]) * 100.0
        ls, mk = styles[key]
        ax1.plot(loss, y, color=colors[key], linewidth=2.2, linestyle=ls,
                 marker=mk, markersize=5, markeredgecolor=SURFACE,
                 markeredgewidth=1.4, zorder=4)
        ends1.append((float(y[-1]), short[key], colors[key]))

    ax1.set_xlabel("packet loss  [%]", fontsize=9)
    ax1.set_ylabel("mission progress toward the station  [%]", fontsize=9)
    ax1.set_title("Does it still get there?", fontsize=10.5, color=INK,
                  loc="left", fontweight="bold", pad=10)
    ax1.set_xlim(-1, loss[-1] + 16)
    ax1.set_ylim(-4, 118)
    ax1.grid(True, color=GRID, linewidth=0.7)
    ax1.set_axisbelow(True)
    _style(ax1)
    _stacked_labels(ax1, ends1, loss[-1])

    stalled = res["stale_gated"][-1]["progress"] * 100.0
    ax1.annotate("safe, but stalled",
                 xy=(loss[-1], stalled), xytext=(loss[-1] - 11, stalled + 26),
                 fontsize=8.5, color=SERIES_3,
                 arrowprops=dict(arrowstyle="-", color=SERIES_3, linewidth=0.9,
                                 alpha=0.8, shrinkB=5))

    worst = max(r["max_jump"] for r in res["stale_ungated"])
    fig.suptitle(
        "\"Freeze safely\" needs a gate the supervisor does not currently have",
        fontsize=12.5, color=INK, x=0.005, ha="left", y=1.055,
        fontweight="bold")
    fig.text(
        0.005, 0.965,
        f"During an outage nothing is emitted and the vehicle holds — that half "
        f"already works. The lunge is in the RECOVERY: the planner keeps "
        f"advancing while the vehicle is\nfrozen, so the first plan to land is "
        f"{worst:.1f} m away ({worst/step:.0f} ticks' worth) and passes every "
        f"existing gate, because freshness bounds a plan's age and nothing "
        f"bounds its distance.",
        fontsize=9, color=INK_2, ha="left", va="bottom", linespacing=1.5)
    return _save(fig, "d4_safe_hold.png")
