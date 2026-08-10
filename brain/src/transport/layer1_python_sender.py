"""
layer1_python_sender.py - the Python twin of swarm-link, for the benchmark.

Same architecture, faithfully mirrored so the comparison is language-vs-language,
not design-vs-design:
  - ingest thread owns the estimate (UDP JSON in, same protocol)
  - one sender THREAD per vehicle (Rust uses tokio tasks)
  - per-vehicle "latest snapshot" slot (Rust uses a watch channel; here a
    GIL-atomic tuple assignment + per-vehicle Condition for wakeup)
  - news is sent IMMEDIATELY on arrival; a 20 Hz heartbeat re-sends the last
    snapshot when no news lands (identical to swarm-link's select! loop)
  - MAVLink v2 VISION_POSITION_ESTIMATE via pymavlink, sendto, same as the
    close_the_loop.py wire format
  - estimate-to-wire latency recorded on the FIRST transmission of each seq

Run:  python src/transport/layer1_python_sender.py --vehicles 12 --csv py_latencies.csv
Feed: {"seq":1,"pos":[[x,y],...]}\n UDP datagrams to --ingest-port, then {"end":true}
"""

import os
os.environ["MAVLINK20"] = "1"                    # MAVLink v2 frames, like the Rust side

import argparse
import json
import math
import socket
import threading
import time

from pymavlink import mavutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vehicles", type=int, default=12)
    ap.add_argument("--ingest-port", type=int, default=47001)
    ap.add_argument("--base-port", type=int, default=14551)
    ap.add_argument("--port-stride", type=int, default=10)
    ap.add_argument("--rate", type=float, default=20.0)
    ap.add_argument("--scale", type=float, default=5.0)
    ap.add_argument("--map-rotation-deg", type=float, default=0.0)
    ap.add_argument("--alt", type=float, default=2.0)
    ap.add_argument("--csv", default="python_sender_latencies.csv")
    args = ap.parse_args()

    n = args.vehicles
    th = math.radians(args.map_rotation_deg)
    a11, a12 = math.cos(th) * args.scale, -math.sin(th) * args.scale
    a21, a22 = math.sin(th) * args.scale, math.cos(th) * args.scale
    down = -args.alt
    boot = time.perf_counter()

    # per-vehicle latest-snapshot slot: (seq, north, east, recv_ns) or None
    latest = [None] * n
    conds = [threading.Condition() for _ in range(n)]
    stop = threading.Event()
    reports = [None] * n

    def sender(v):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        target = ("127.0.0.1", args.base_port + v * args.port_stride)
        mav = mavutil.mavlink.MAVLink(None, srcSystem=255, srcComponent=191)
        period = 1.0 / args.rate
        last_seq = -1
        sends = 0
        samples = []
        t0 = time.perf_counter()
        my_cond = conds[v]
        while not stop.is_set():
            with my_cond:
                my_cond.wait(timeout=period)         # news wakes us; timeout = heartbeat
            snap = latest[v]
            if snap is None:
                continue
            seq, north, east, recv_ns = snap
            usec = int((time.perf_counter() - boot) * 1e6)
            msg = mav.vision_position_estimate_encode(
                usec, north, east, down, 0.0, 0.0, 0.0)
            sock.sendto(msg.pack(mav), target)
            sends += 1
            if seq != last_seq:                      # first transmission = news
                last_seq = seq
                samples.append((seq, (time.perf_counter_ns() - recv_ns) // 1000))
        reports[v] = (sends, time.perf_counter() - t0, samples)

    threads = [threading.Thread(target=sender, args=(v,), daemon=True) for v in range(n)]
    for t in threads:
        t.start()

    ingest = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    ingest.bind(("127.0.0.1", args.ingest_port))
    print(f"python-sender: {n} vehicles at {args.rate:.0f} Hz, "
          f"ingest udp:{args.ingest_port}", flush=True)

    frames = 0
    while True:
        data, _ = ingest.recvfrom(65536)
        recv_ns = time.perf_counter_ns()             # estimate-to-wire clock starts HERE
        try:
            frame = json.loads(data)
        except json.JSONDecodeError as e:
            print(f"bad frame ignored: {e}", flush=True)
            continue
        if frame.get("end"):
            break
        pos = frame.get("pos", [])
        if len(pos) < n:
            continue
        seq = frame.get("seq", 0)
        for v in range(n):
            x, y = pos[v]
            north = a11 * x + a12 * y
            east = a21 * x + a22 * y
            latest[v] = (seq, north, east, recv_ns)
            with conds[v]:
                conds[v].notify()
        frames += 1

    stop.set()
    for c in conds:
        with c:
            c.notify()
    for t in threads:
        t.join(timeout=2)

    all_lat = []
    lines = []
    csv_rows = ["vehicle,seq,latency_us"]
    for v, rep in enumerate(reports):
        if rep is None:
            continue
        sends, elapsed, samples = rep
        lats = sorted(s[1] for s in samples)
        if lats:
            p50 = lats[len(lats) // 2]
            p99 = lats[min(len(lats) - 1, int(0.99 * len(lats)))]
            lines.append(f"  vehicle {v:2}: p50 {p50:>6} us  p99 {p99:>7} us  "
                         f"max {lats[-1]:>7} us  achieved {sends/elapsed:>5.1f} Hz")
        all_lat.extend(lats)
        csv_rows += [f"{v},{s},{l}" for (s, l) in samples]
    all_lat.sort()
    print(f"python-sender: {frames} frames ingested", flush=True)
    for l in lines:
        print(l, flush=True)
    if all_lat:
        q = lambda p: all_lat[min(len(all_lat) - 1, int(p * len(all_lat)))]
        print(f"TOTAL estimate-to-wire: p50 {q(.5)} us  p90 {q(.9)} us  "
              f"p99 {q(.99)} us  p99.9 {q(.999)} us  (n={len(all_lat)})", flush=True)
    with open(args.csv, "w") as f:
        f.write("\n".join(csv_rows) + "\n")
    print(f"wrote {args.csv}", flush=True)


if __name__ == "__main__":
    main()
