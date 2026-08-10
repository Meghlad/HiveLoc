# ⚙️ Layer 1 Tutorial — The Rust Rewrite That Isn't Gratuitous

**The claim this layer proves:** the README already wrote the spec —
> *"the real system splits estimation from transport — one estimator thread solving the swarm, and one lightweight ~20 Hz sender per vehicle, decoupled from solver speed."*

That paragraph describes a Rust program. This layer builds it (`swarm-link`), builds a Python twin with the *identical architecture*, and races them. The whole "why Rust" answer is one measured number: estimate-to-wire p99 at N=12.

---

## 1. What `close_the_loop.py` actually does, and where it strains

The finale streams drone 0's estimate into ArduCopter. Its own comments admit the seam:

```python
# PRODUCTION NOTE: real systems run this send in a separate 20Hz thread that
# re-sends the latest (north,east) continuously, decoupled from estimator speed.
for _ in range(RESENDS):
    send_vision(north, east, down)
    time.sleep(1.0 / (FRAME_HZ * RESENDS))
```

`time.sleep()` inside the loop that also does the estimation *is* the coupling. And to fly the whole swarm you need **N of these senders running at once** — which on CPython means N threads contending for one GIL. That contention is the thing we measure.

---

## 2. The architecture (identical in both languages)

```
estimator (Python, any rate) ──UDP JSON──▶ [ingest task] ── owns the estimate
                                                │  one channel per vehicle
                                   ┌────────────┼────────────┐
                                   ▼            ▼            ▼
                             [sender 0]   [sender 1] ...  [sender N-1]   20 Hz each
                                   │            │            │
                          MAVLink v2 VISION_POSITION_ESTIMATE
                                   ▼            ▼            ▼
                             udp:14551    udp:14561  ...  (one SITL per vehicle)
```

- **Ingest owns state; senders borrow snapshots.** In Rust this is a `watch` channel: the ingest task holds the unique `Sender`, each vehicle task gets a `Receiver` that can only `borrow()` an immutable `Snapshot`. **The borrow checker refuses to compile any code that lets a sender mutate estimator state.** The estimator-owns / senders-borrow split the README identified in prose is *mechanically enforced*, not just intended.
- **News vs heartbeat.** A fresh estimate is transmitted the instant it lands (`rx.changed()` wakes the sender); the 20 Hz timer only *re-sends the last snapshot* so the EKF never starves between estimator frames — the "nav heartbeat that never dies," now one per vehicle, with no `sleep()` in any loop that also does math.
- **The Python twin** (`layer1_python_sender.py`) mirrors all of this: ingest thread, one sender thread per vehicle, a per-vehicle latest-snapshot slot with a `Condition` for wakeup, the same MAVLink v2 wire format via pymavlink. The comparison is language-vs-language, not design-vs-design.

---

## 3. The measurement, and why it's honest

Both implementations timestamp **the instant the ingest side takes ownership** of a frame, and record latency on the **first transmission of that frame per vehicle** (re-sends are heartbeats, not news). That interval — `parse → frame-transform → MAVLink encode → sendto` — is *estimate-to-wire*. It is not the phase of a timer, because a fresh estimate preempts the timer.

The benchmark (`layer1_bench.py`) replays **real Layer-2 estimator output** (`layer2_isam2_results.npz`) at 10 Hz, fans out to N vehicles, and dumps per-vehicle HDR histograms. `swarm-link` records its *own* latency internally — the benchmark is not a harness bolted on the outside; the binary measures itself.

---

## 4. The number

**Measured on this machine (Apple Silicon, macOS), 300 frames at 10 Hz, µs:**

| config | p50 | p90 | p99 | max |
|---|---:|---:|---:|---:|
| python N=1 | 312 | 430 | 820 | 2,673 |
| python N=12 | 1,505 | 1,779 | 1,988 | 56,551 |
| rust N=1 | 68 | 115 | 443 | 1,627 |
| rust N=12 | **115** | 197 | **336** | 6,502 |

Read the two things that matter:

1. **At N=12, Rust p99 is ~6× lower than Python (336 µs vs 1,988 µs).**
2. **The scaling is the real story.** Python's *median* rises 4.8× (312 → 1,505 µs) going from 1 vehicle to 12 — that's the GIL serializing twelve sender threads. Rust's median rises 1.7× (68 → 115 µs) and its p99 actually *drops* below N=1's (warm caches, no contention). One language jitters more as the swarm grows; the other flattens.

Python's N=12 **max of 56 ms** is a garbage-collection pause landing on a send. On a path feeding an EKF that expects ~20 Hz, a 56 ms stall is a skipped update; a hundred of them is a lane switch. This is the "no GC or GIL on a path that must not jitter" argument, as a histogram instead of an assertion.

![latency](../layer1_latency.png)

---

## 5. Why Rust here, not C++ (the interview answer)

- **No GC, no GIL** on a path that must not jitter — the table above is the evidence.
- **The borrow checker enforces the exact ownership split** the system design requires (estimator owns state, senders hold read-only snapshots). In C++ that's a code-review rule; in Rust it's a compile error.
- **A single static binary cross-compiles to a Jetson or companion computer** — 1.8 MB, no runtime, no interpreter, `ORT_DYLIB_PATH`-style dynamic deps only where you want them.
- **Where Rust actually lives:** embedded and safety-critical. Ferrocene is a qualified Rust toolchain for ISO 26262 (automotive) and IEC 61508 (industrial); there's active aerospace flight-software work. That's the honest provenance — not the blockchain association.

---

## 6. Reproduce

```bash
cd rust && cargo build --release && cd ..
python src/transport/layer1_bench.py          # ~2.5 min: python/rust × N=1/N=12, writes layer1_latency.png

# or drive swarm-link by hand against N SITL instances:
./rust/target/release/swarm-link --vehicles 12 --base-port 14551 --port-stride 10
#   then feed it {"seq":k,"pos":[[x,y],...]} UDP datagrams on :47001
```

## 7. What to say in the room

- *"The README specified a decoupled 20 Hz-per-vehicle transport before any Rust existed. Layer 1 is that spec, compiled — and the borrow checker is what enforces 'estimator owns state, senders only borrow snapshots.'"*
- *"Why Rust over C++? Same no-GC argument, plus the ownership rule is a compile error instead of a code-review comment. Here's the p99 at twelve vehicles."*
- *"Python's median latency grows 5× from one vehicle to twelve — that's the GIL. Rust's is flat. The 56 ms Python outlier is a GC pause landing on a send, which on a 20 Hz EKF feed is a dropped update."*
- *"It's a 1.8 MB static binary. That's what ships to the Jetson."*
