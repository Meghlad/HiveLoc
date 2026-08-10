"""
layer1_bench.py - THE number: estimate-to-wire latency, Python vs Rust, N=1 vs N=12.

Drives both transport implementations through the identical protocol:
  - frames are REAL Layer-2 estimates (layer2_isam2_results.npz, healthy radio),
    replayed at 10 Hz - the estimator's actual output rate, not synthetic noise
  - each frame carries all N vehicle positions as one UDP JSON datagram
  - the sender-under-test fans out per-vehicle MAVLink VISION_POSITION_ESTIMATE
  - estimate-to-wire is measured INSIDE each sender, identically: ingest
    timestamp -> first transmission of that frame per vehicle

Both implementations share the same architecture (news sent immediately,
20 Hz heartbeat re-sends, per-vehicle senders, latest-snapshot handoff), so
the histogram difference is the runtime: GC/GIL/thread-wakeup vs tokio.

Run:  python src/transport/layer1_bench.py            (~2.5 min: 4 configs x 35 s)
"""

import json
import pathlib
import socket
import subprocess
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent          # src/transport/
ROOT = HERE.parents[1]                                  # repo root
RUST_BIN = ROOT / "rust/target/release/swarm-link"
PY = sys.executable
INGEST = 47001
FRAMES = 300           # 30 s at 10 Hz per config
RATE_HZ = 10.0

# real estimator output as the replay source
est = np.load(ROOT / "data/layer2_isam2_results.npz")["online_r055"]   # [120, 12, 2]


def run_config(impl, n_vehicles):
    csv = ROOT / f"layer1_{impl}_n{n_vehicles}.csv"
    if impl == "rust":
        cmd = [str(RUST_BIN), "--vehicles", str(n_vehicles),
               "--ingest-port", str(INGEST), "--csv", str(csv)]
    else:
        cmd = [PY, str(HERE / "layer1_python_sender.py"),
               "--vehicles", str(n_vehicles),
               "--ingest-port", str(INGEST), "--csv", str(csv)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True)
    time.sleep(1.0)                                   # let it bind

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    period = 1.0 / RATE_HZ
    next_t = time.perf_counter()
    for seq in range(1, FRAMES + 1):
        pos = est[seq % len(est), :n_vehicles].tolist()
        sock.sendto(json.dumps({"seq": seq, "pos": pos}).encode(),
                    ("127.0.0.1", INGEST))
        next_t += period
        dt = next_t - time.perf_counter()
        if dt > 0:
            time.sleep(dt)
    sock.sendto(b'{"end":true}', ("127.0.0.1", INGEST))
    out, _ = proc.communicate(timeout=30)

    lat = np.loadtxt(csv, delimiter=",", skiprows=1, usecols=2, ndmin=1)
    return lat, out


def pct(a, p):
    return float(np.percentile(a, p))


if __name__ == "__main__":
    results = {}
    for impl in ("python", "rust"):
        for n in (1, 12):
            print(f"=== {impl} N={n} ({FRAMES} frames at {RATE_HZ:.0f} Hz) ===")
            lat, out = run_config(impl, n)
            results[(impl, n)] = lat
            for line in out.strip().splitlines():
                if "TOTAL" in line or "achieved" in line and n == 1:
                    print("  " + line.strip())
            print(f"  -> n={len(lat)} samples  p50 {pct(lat,50):.0f} us  "
                  f"p99 {pct(lat,99):.0f} us  max {lat.max():.0f} us\n")

    # ---- the money table -----------------------------------------------------
    print(f"{'config':<14}{'p50 (us)':>10}{'p90 (us)':>10}{'p99 (us)':>10}{'max (us)':>10}")
    for (impl, n), lat in results.items():
        print(f"{impl + ' N=' + str(n):<14}"
              f"{pct(lat,50):>10.0f}{pct(lat,90):>10.0f}{pct(lat,99):>10.0f}"
              f"{lat.max():>10.0f}")
    r, p = results[("rust", 12)], results[("python", 12)]
    print(f"\nAt N=12: Rust p99 is {pct(p,99)/pct(r,99):.0f}x lower than Python "
          f"({pct(r,99):.0f} vs {pct(p,99):.0f} us)")

    # ---- the money plot ------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    styles = {("python", 1): ("tab:orange", ":"), ("python", 12): ("tab:orange", "-"),
              ("rust", 1): ("tab:blue", ":"), ("rust", 12): ("tab:blue", "-")}
    for (impl, n), lat in results.items():
        c, ls = styles[(impl, n)]
        s = np.sort(lat)
        ax[0].plot(s, np.linspace(0, 1, len(s)), color=c, ls=ls,
                   label=f"{impl} N={n}")
    ax[0].set_xscale("log")
    ax[0].set_xlabel("estimate-to-wire latency (us, log)")
    ax[0].set_ylabel("CDF")
    ax[0].set_title("Latency CDF: same architecture, different runtime")
    ax[0].grid(alpha=0.3, which="both"); ax[0].legend()

    labels = [f"{i} N={n}" for (i, n) in results]
    x = np.arange(len(results))
    p50s = [pct(l, 50) for l in results.values()]
    p99s = [pct(l, 99) for l in results.values()]
    ax[1].bar(x - 0.2, p50s, 0.4, label="p50", color="tab:gray")
    ax[1].bar(x + 0.2, p99s, 0.4, label="p99", color="tab:red")
    ax[1].set_yscale("log")
    ax[1].set_xticks(x, labels)
    ax[1].set_ylabel("latency (us, log)")
    ax[1].set_title("p50 / p99 estimate-to-wire")
    ax[1].grid(alpha=0.3, axis="y", which="both"); ax[1].legend()
    fig.suptitle("swarm-link (Rust/tokio) vs identical-architecture Python sender")
    plt.tight_layout()
    plt.savefig("figures/layer1_latency.png", dpi=130, bbox_inches="tight")
    print("saved layer1_latency.png")
