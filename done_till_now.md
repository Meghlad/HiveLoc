# done_till_now

Status of the GPS-denied anchored swarm stack. Written to be read by someone
deciding what to trust, so the failures are stated as plainly as the wins and
every claim carries the measurement behind it.

The detailed engineering record is `hive_assist/ANCHOR_SWARM_PLAN.md`, whose
findings table runs to 45 entries. This file is the summary.

---

## 1. Status at a glance

| area | state | evidence |
|---|---|---|
| Offline layers (`brain/`) | done | Layer 1-3 tutorials + figures |
| Unit tests | passing | **224 passed, 3 skipped, 0 failures** |
| SITL stack (gz + PX4 + ROS 2) | running | 4 vehicles, RTF 0.418 |
| Estimator **at rest** | **calibrated** | 0.065 m err / 0.083 sigma = **0.78** |
| Estimator **in flight** | **BROKEN** | 14.9 m err / 0.101 sigma = **147x overconfident** |
| Single-vehicle hover | works | held 3.87-4.22 m for 25 s, landed on command |
| Fleet flight | **not achieved** | only vehicle 0 has ever flown |
| 12 vehicles | not started | blocked behind fleet flight |
| netem fault sweep | not started | written, never run |
| Link security (H0) | **done, opt-in** | unsigned command ACKed before, refused after — measured against SITL |
| Link security (H1-H5) | not started | see `COMMS_HARDENING_PLAN.md` §8 |
| Hardware | never | simulation only, start to finish |

---

## 2. What works, with the number that proves it

**The offline layers and the maths.** 224 unit tests pass across frames, CBBA,
null-space, mission FSM, safe-hold, supervisor I/O and the anchored iSAM2
solver. Figures regenerate from `make figures`.

**The simulator runs at usable speed.** RTF went **0.034 -> 0.418** on the live
4-vehicle stack. The cause was not the missing GPU and not physics — it was
rendering both OakD-Lite sensors at 30 Hz when nothing downstream consumes
faster than the 10-15 Hz keyframe rate:

| camera `update_rate` | RTF (render only, 2 camera vehicles) |
|---|---|
| 30 Hz | 0.494 |
| 15 Hz | 0.959 |
| **10 Hz** | **1.000** |

**The estimator is genuinely calibrated when nothing is moving.** 0.065 m mean
error against truth while reporting 0.083 m sigma — ratio 0.78. Error and sigma
agree, which is the actual claim; a tight covariance on its own proves nothing.

**One vehicle completes a GPS-denied, compass-free flight.** Armed, climbed,
**held 3.87-4.22 m for 25 s**, landed on command, disarmed. No failsafe. Flying
with no GPS and no magnetometer, on the anchored estimate alone.

**All four vehicles now reach `Ready for takeoff`** — once the start order is
right (see §5).

**The MAVLink links can be authenticated, and the proof is a refusal.**
`COMMS_HARDENING_PLAN.md` stage H0 is built and wired: per-vehicle MAVLink 2
signing keys, encrypted at rest, pushed with `SETUP_SIGNING`, unsigned frames
refused. Run it with `--sign`. The number that matters is not a passing test —
it is the same probe against the same vehicle, before and after:

| link | signing | an UNSIGNED command is |
|---|---|---|
| `--serial0` (the old transport) | provisioned | **ACKed and executed** |
| `--serial2` (now) | provisioned | refused |
| `--serial2` | none | ACKed and executed |

The middle row is why the transport moved. ArduPilot gates unsigned frames with
a compiled-in callback, not a parameter, and `GCS_Signing.cpp:116` accepts
everything on `MAVLINK_COMM_0` unconditionally — which serial0 is. The loop had
been sitting on the one channel where signing cannot do anything, so turning
signing on there would have produced healthy counters and no security. Reading
the source found it; running the probe proved it.

Cost against the 25 ms loop budget, 2 vehicles, signed vs unsigned back to
back: p50 4.2 ms both, p99 16.4 -> 18.4 ms. Signing is not the constraint.

It is **opt-in**, not default, because §3.1's estimator failure is still the
thing under repair and an unsigned run has to stay reproducible. There is no
argument for leaving it off once that lands.

---

## 3. Where it fails

### 3.1 The estimator lies as soon as anything moves — THE headline failure

At rest it is calibrated. In flight it is not, and it is not close:

| condition | error | sigma | ratio |
|---|---|---|---|
| at rest | 0.065 m | 0.083 m | 0.78 CALIBRATED |
| in flight, before VIO gate | 8.3 x 10^11 m | 1000 (untrusted) | detonated |
| in flight, after VIO gate | **14.9 m** | **0.101 m** | **147x OVERCONFIDENT** |

The gate stopped the detonation — eleven orders of magnitude — and the flight
now completes without a failsafe. It did **not** make the estimate correct.
14.9 m of error reported as 10 cm of confidence is the single worst property of
this system, because the supervisor's trust gate reads that sigma and would
certify the estimate as good.

This is the project's own central thesis turned on itself: *a confidently wrong
estimate is worse than no estimate*, because the safety gate will pass it.

### 3.2 The VIO front end produces garbage in both directions

Root cause of 3.1, and it is upstream of the estimator. Checked against the
preintegrated IMU — an independent second opinion over the same interval,
needing no ground truth, so the check would survive on real hardware:

```
|vio| = 0.228 m   |imu| = 1.917 m    ->  84 sigma      (moving, VIO says still)
|vio| = 0.111 m   |imu| = 3.755 m    -> 182 sigma
|vio| = 0.000 m                      ->  "I did not move", at full weight
|vio| = 7.238 m   |imu| = 0.057 m    -> 359 sigma      (STILL, VIO says moving)
```

Seven metres of motion reported on a stationary airframe. **45 rejections in one
flight, every one of them vehicle 0.** The front end is not merely noisy — it is
wrong in both directions and confident about it.

Two compounding design faults made the estimator swallow it:

1. `(reproj_px/fx)^2 / n` is ~4e-7 at 50 inliers, so `sqrt(cov_sum)` is
   sub-millimetre and the **0.02 m floor always wins**. That covariance
   describes reprojection geometry — it cannot see that the *matches* are wrong.
2. `BetweenFactorPose3` carried **no robust kernel**, while the inter-agent
   ranges have carried Huber all along for exactly this reason.

Fixed at the factor level (IMU-disagreement gate + Huber). The front end itself
is untouched and is the real remaining bug. Prime suspects: depth scale, and the
deterministic field of near-identical boxes giving ORB wrong matches to lock on.

### 3.3 Only one vehicle has ever flown

Vehicles 1-3 have never left the ground. They reach `Ready for takeoff` now, but
the flight test has only ever produced `Takeoff detected` on vehicle 0. Until
3.1 and 3.2 are fixed there is no fleet flight claim to make.

### 3.4 The hover sits 1.1 m high

Commanded 3.0 m, held ~4.1 m. Height is on the barometer while the setpoint is
in the EV frame — a frame offset to chase, not a controller gain.

### 3.5 IMU samples are dropped, worst on the camera vehicles

`gaps [9, 5, 3, 3]` in one flight — keyframes where no IMU arrived at all.
Vehicles 0 and 1 carry the cameras, and their IMU stream competes with image and
depth traffic for the same subscriber. Currently bridged with a loose relative
factor so the solve stays non-singular, but the starvation itself is real and
unfixed.

### 3.6 The GPU cannot be used for rendering, and this is now proven

Not a configuration failure — a genuine incompatibility, documented so nobody
retries it blind. Hardware GL *is* reachable: Mesa's d3d12 driver reports
`D3D12 (NVIDIA GeForce RTX 4060 Laptop GPU)`, GL 4.6 core, from inside the
container. But **Ogre-Next 2.3.3's headless path only accepts the EGL device
platform, and Mesa d3d12 publishes no EGL device** — it exists only on the
surfaceless platform. So Ogre enumerates one software device, fails
`eglInitialize`, and segfaults. Vulkan is shut too: `RenderSystem_Vulkan.so`
ships, but Vulkan-on-D3D12 needs Mesa's `dzn`, which Ubuntu does not build.

Rendering stays on CPU. A Linux host with a real `/dev/dri` would work.

---

## 4. What is lacking / never built

- **Hardware.** Nothing has flown on a real airframe. Simulation only.
- **Fleet flight.** 4 vehicles arm; 1 flies.
- **12-vehicle scale.** Designed for it, RTF budgeted for it, never run.
- **netem fault sweep.** Script exists, never executed.
- **Full mission closed loop.** CBBA auction, mission FSM, standoff controller
  and the Rust supervisor gate are written and unit-tested, but have not been
  exercised end-to-end against live physics with a flying fleet.
- **VIO validation.** There is no test that would have caught a front end
  reporting 7 m of motion on a stationary vehicle. The IMU cross-check added
  during debugging is the beginning of one.
- **Everything in the hardening plan past H0.** No confidentiality tunnel, no
  key rotation schedule, no supervisor plausibility gate against a spoofed
  `VISION_POSITION_ESTIMATE` — which is the plan's own H1 and the highest-value
  remaining item, because it is the surface unique to this design. Signing
  stops an outsider forging that stream; it does nothing about one that is
  authenticated and wrong, and §3.1 is the standing proof that this estimator
  can be confidently wrong on its own.
- **RTF headroom.** 0.418 at 4 vehicles. Render is proven free at 10 Hz, so the
  remainder is PX4 lockstep plus consumers — untouched, and it will get worse
  at 12.

---

## 5. Operational gotchas that cost real time

**Start order is gz -> estimator -> PX4.** EKF2 converges on external vision it
has seen since *its own* boot. Restarting the estimator under a live fleet
resets the keyframe index and steps the EV pose; every vehicle then sits at
`Preflight Fail: height estimate not stable` forever. Wrong order: 0 of 4 armed.
Right order: 4 of 4.

**Relaunching a camera fleet segfaults a live gz server.** Removing depth-camera
models kills Ogre2, so gz must be restarted first.

**A camera with no subscriber does not render.** gz-sim skips the render
entirely, so an RTF measured without consumers attached is fiction — it reads
~1.0 at 22% CPU and means nothing. This produced two wrong RTF readings before
it was caught. Confirm a subscriber exists before believing any RTF number.

**A flight test on vehicles that never armed still prints `done`.** It streams
setpoints at the ground and reports success, which silently invalidates any
"after a flight" measurement.

---

## 6. Fragile right now

The VIO gate, Huber kernel, IMU-gap bridge, keyframe self-heal and the
`[frame]` / `[vio]` diagnostics are **live in the container layer only**. The
repo source has them; the `hive-assist/ros2:jazzy` image does not. A
`docker rm` or `compose up --force-recreate` loses them.

**Rebuild the image before doing anything else.**

---

## 7. What I would do next, in order

1. **Rebuild the image** so the fixes survive.
2. **Fix `rgbd_vio.py`** — §3.2 is the blocker for everything else. Check depth
   scale first (a units error would explain motion wrong by a constant factor),
   then feature ambiguity against the repeated box field.
3. **Add a VIO regression test**: a stationary vehicle must produce near-zero
   relative pose. That single assertion would have caught this.
4. **Re-measure error/sigma in flight.** Ratio near 1.0 is the acceptance bar,
   not a small error and not a small sigma.
5. Only then: fleet flight, then 12 vehicles, then netem.

---

## 8. Honest one-paragraph version

A GPS-denied swarm estimator that is provably calibrated at rest (0.065 m error
against 0.083 m reported sigma) and flies one vehicle in a stable hover with no
GPS and no compass, inside a full PX4 SITL + Gazebo + ROS 2 stack running at
0.418 real-time. It does not yet fly a fleet, and its estimate degrades to 14.9 m
of error while still reporting 10 cm of confidence once vehicles move — traced to
a visual-odometry front end that reports 7 m of motion on a stationary airframe,
which the estimator was fusing at a 2 cm noise floor with no robust kernel. That
fault is now gated and instrumented but not repaired. Nothing has flown on
hardware.
