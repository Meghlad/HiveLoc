"""
animate_swarm.py  -  the payoff visual: real-flight swarm + your iSAM2 estimate.

Renders from arrays already produced by day8_isam2_traj.py:
    true_traj  [T,n,2]   real Crazyflie flight (ground truth)
    online     [T,n,2]   your LIVE iSAM2 estimate at each frame
    frames     list of (sensor_edges, anchor_edges) per frame  (the UWB sensing graph)
    anchors    [m,2]

EASIEST USE: paste this block at the very END of day8_isam2_traj.py (all arrays are in
scope there). Otherwise import/recompute those arrays first.

Output: swarm_real_flight.gif
Needs:  pillow  (pip install pillow)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

T = true_traj.shape[0]
n = true_traj.shape[1]

fig, ax = plt.subplots(figsize=(7, 7))

def draw(t):
    ax.clear()
    Xt, Et = true_traj[t], online[t]
    se, ae = frames[t]

    # UWB sensing graph for this frame (flickers as links form / drop out)
    for (i, j, _) in se:
        ax.plot([Xt[i, 0], Xt[j, 0]], [Xt[i, 1], Xt[j, 1]],
                color="gray", lw=0.5, alpha=0.35, zorder=1)
    for (i, k, _) in ae:
        ax.plot([Xt[i, 0], anchors[k, 0]], [Xt[i, 1], anchors[k, 1]],
                color="steelblue", lw=0.4, alpha=0.30, zorder=1)

    # short trails of the true flight
    for tt in range(max(0, t - 14), t):
        ax.plot(true_traj[tt:tt + 2, :, 0], true_traj[tt:tt + 2, :, 1],
                color="green", alpha=0.12, zorder=1)

    # estimate-to-truth error sticks
    for i in range(n):
        ax.plot([Xt[i, 0], Et[i, 0]], [Xt[i, 1], Et[i, 1]],
                color="red", alpha=0.5, lw=1, zorder=2)

    ax.scatter(anchors[:, 0], anchors[:, 1], c="k", marker="^", s=120,
               zorder=4, label="anchors")
    ax.scatter(Xt[:, 0], Xt[:, 1], c="green", s=45, zorder=4, label="true (real flight)")
    ax.scatter(Et[:, 0], Et[:, 1], facecolors="none", edgecolors="red", s=85,
               zorder=4, label="iSAM2 estimate")

    err = np.sqrt(((Et - Xt) ** 2).sum(1).mean())
    ax.set_xlim(-0.1, 1.1); ax.set_ylim(-0.1, 1.1); ax.set_aspect("equal")
    ax.set_title(f"GPS-denied cooperative localization on real flight physics\n"
                 f"frame {t}/{T}   live RMSE = {err:.3f} m")
    ax.legend(loc="upper left", fontsize=8); ax.grid(alpha=0.3)

anim = animation.FuncAnimation(fig, draw, frames=T, interval=120)
anim.save("figures/swarm_real_flight.gif", writer=animation.PillowWriter(fps=10))
print("saved swarm_real_flight.gif")
plt.show()
