# 📷 Layer 2 Tutorial — Vision Bearings: the Sensor That Fixes What Ranges Can't

**The claim this layer proves:** a range-only swarm graph fails in exactly two directional ways — flexes and mirror flips — and a camera is the exact complementary sensor. We quantify *how much* vision it takes to fix a marginal graph, then build the full pipeline: rendered camera frames → ONNX inference in Rust → data association → bearing factors in the live iSAM2 estimator.

---

## 1. Why bearings, from first principles

Day 2 established the two failure modes of range-only localization:

| Failure | Cause | What a range can never tell you |
|---|---|---|
| **Floppy graph** | too few edges → rigidity rank < 2n | *which way* a joint flexed |
| **Mirror flip** | reflection preserves every distance | *which of two worlds* you're in |

Both failures are **directional**. A range constrains *how far*, never *which way*.

Now put a camera on drone *i*. When it detects drone *j* in frame, the pixel column of that detection is a **bearing** — pure direction, zero distance. On the shared edge (i,j):

```
range row in the rigidity matrix   :  (Xi - Xj)          "don't change length"
bearing row in the rigidity matrix :  perp(Xi - Xj)      "don't change direction"
```

They are literally **orthogonal constraints on the same edge**. One edge carrying both pins the full relative position.

**Assumption stated up front:** yaw is known (compass/IMU), so a body-frame pixel bearing converts to a world-frame bearing. Heading is *not* estimated — the same division of labor as the flight stack, where `EK3_SRC1_YAW` is its own source, separate from position.

### The convexity gift

The Day 1 SDP relaxes *quadratic* range constraints. A bearing constraint needs no relaxation at all:

```
(Xj - Xi) · n̂ = 0     n̂ ⊥ measured direction     ← LINEAR in positions
(Xj - Xi) · û ≥ 0     target is IN FRONT of lens  ← LINEAR inequality
```

Both drop into the Biswas–Ye SDP with **zero added relaxation gap** (positions live in the linear block `Z[:2, 2:]`). The ray inequality is what actually kills the mirror flip: a reflection satisfies the perp equation but puts the target *behind* the camera.

One estimator subtlety we hit and kept honest: even on a rigid mixed graph, the SDP alone can stay smeared (rank(Z) > 2). The full solve is **SDP init → whitened NLS polish** — the same division of labor as Days 4–8. Rigidity is the property that predicts whether the polish can lock in: on a floppy graph the flex direction is a flat valley and the polish wanders down it.

**Files:** `layer2_bearing_phase_diagram.py` → `layer2_bearing_rescue.png`, `layer2_bearing_phase_diagram.png`, `layer2_sweep_results.npz`

### The deliverable numbers (measured, not asserted)

Sweep: connectivity radius R × max detections per drone B, **same networks shared across the B axis, nested camera sampling** (B=2's detections are a strict subset of B=3's — any change along B is information, not resampling noise). At the marginal radius **R = 0.37** (range-only graph rigid only 33% of the time):

| B per drone | det/frame (whole swarm) | rigid | median RMSE |
|---|---|---|---|
| 0 (Day 2) | 0 | 33% | 0.263 m |
| 1 | ~7 | 83% | 0.153 m |
| 2 | ~13 | 83% | 0.120 m |
| 4 | ~20 | 83% | **0.055 m** |

**One detection per drone triples the rigid fraction. A full camera load is a 5× RMSE improvement.** And the honest boundary: below R ≈ 0.31 even B=4 doesn't help — vision rescues the *marginal* regime, not the disconnected one. A graph missing 10 DOF needs radios, not cameras.

Conditioning is *measured* as λ_min of the stiffness matrix RᵀR (the weakest flex direction's stiffness), not eyeballed: the rescue demo takes one marginal network from rank 22/24, λ_min = 0, RMSE 0.267 m to **rank 24/24, λ_min = 2.6·10⁻³, RMSE 0.029 m** with 14 bearings.

---

## 2. Bearings in the live estimator (iSAM2)

**File:** `layer2_isam2_bearing.py` → `layer2_isam2_bearing.png`, `layer2_isam2_results.npz`

The bearing factor mirrors the repo's `range_factor` style — `CustomFactor` on `Point2` variables, hand Jacobians:

```python
residual = wrap(atan2(pj - pi) - measured)
d(theta)/d(pi) = [ dy, -dx] / d²        d(theta)/d(pj) = [-dy,  dx] / d²
```

Huber-wrapped, because a detector will sometimes box the wrong neighbor — a data-association error is a bearing outlier (Day 7's lesson, new sensor). Cameras face the direction of travel (forward-mounted lens), FOV 90°, up to B=2 detections per drone per frame.

Two radio conditions, each range-only vs range+bearing on **identical measurements**:

| Condition | range-only | + vision | note |
|---|---|---|---|
| R=0.55 healthy | 0.093 m | **0.033 m** | vision shaves 2.8× |
| R=0.35 degraded | 0.279 m | **0.052 m** | vision prevents collapse, 5.4× |

**A modeling bug we caught and what it taught:** with anchors on the same radio model, R=0.35 gave ~0 anchor edges/frame and *both* runs diverged to meters. Ranges and drone-drone bearings are both **relative** — bearings pin shape and rotation, but only anchors pin *translation*. Fix: `R_ANCHOR = 0.55` stays fixed (mains-powered ground infrastructure doesn't degrade with the inter-drone links). Know which gauge freedoms each sensor closes.

**The Layer-3 foreshadow:** the script exports every drone's per-frame **marginal covariance**. Look at the degraded range-only row: the estimator *believed* σ = 0.042 m while actually erring 0.279 m — **a floppy graph makes the covariance a liar**. With bearings, confidence and truth agree. This is precisely why the Layer-3 safety supervisor can gate plans on marginal covariance — that signal is only meaningful because Layer 2 made the graph well-conditioned.

---

## 3. The real pipeline: pixels → ONNX (Rust) → factors

**Files:** `layer2_make_dataset.py`, `rust/swarm-perception/`, `layer2_perception_closeloop.py` → `bearings.jsonl`, `layer2_perception_closeloop.png`

### 3a. The camera frames

`layer2_make_dataset.py` renders every drone's onboard view: 120 frames × 12 drones = 1,440 grayscale 320×240 images. Pinhole projection (`u = cx + fx·tan(bearing_body)`, fx = 160 for a 90° FOV), closer targets bigger and brighter, `P_DET = 0.9` dropouts (motion blur/occlusion — the physical origin of Part A's detection probability), 5% clutter blobs (birds, glints), sensor noise. Plus `frames/meta.jsonl`: per-image heading + intrinsics (legitimate onboard info) *and* a ground-truth table used **only** for scoring — the Rust node never reads it.

### 3b. The detector — and why it's honest

`detector.onnx` is a 9×9 difference-of-Gaussians blob detector expressed as a Conv→ReLU ONNX graph with **fixed weights**. Deliberately not a trained network: the contract (image in → response map out) is identical to what a trained YOLO head satisfies, and the weights are math we can verify. Swapping in a learned model later changes one file and zero lines of Rust.

### 3c. The Rust node

`rust/swarm-perception` (the `ort` crate = ONNX Runtime bindings):

```
PNG → f32 tensor → ort inference → strict local maxima → NMS →
response-weighted 7×7 centroid (sub-pixel) → u → bearing_body → + heading → world bearing
```

- Emits **raw bearings with no identity** — deciding *which* neighbor a blob is belongs to the estimator, which has the predicted swarm state. That seam keeps the node swappable for a real camera feed.
- macOS build note: the prebuilt static onnxruntime needs a ≥14.4 SDK for its CoreML symbols; we use ort's `load-dynamic` feature and point `ORT_DYLIB_PATH` at the venv's pip-installed dylib. On a Jetson you'd point it at the JetPack build — that *is* the deployment pattern.

**Throughput: 1,440 images → 3,457 bearings in 2.3 s (629 img/s; p50 0.67 ms/image).**

Detector scoreboard vs the withheld truth: **recall 94.1%, precision 96.0%, bearing error RMS 0.147° (p95 0.35°)** — 13× better than the 2° the synthetic model assumed. Sub-pixel centroiding is where that comes from (1 pixel = 0.36°).

### 3d. Data association — where the naive thing collapses, measurably

First attempt: match each detection to the nearest predicted bearing inside a gate. Result, audited offline against truth:

> **onnx_naive: purity 10.7% — 85% of matched factors named the WRONG drone. RMSE 0.32 m, worse than no vision at all.**

Why it *must* fail here: relative prediction error ~0.2–0.3 m at target distances ≤ 0.65 m is **30–60° of bearing error**, while candidate neighbors sit only 15–30° apart. Single-frame nearest-bearing is near-random — and each wrong factor *tightens the covariance around the wrong geometry*, so the gate then confirms the wrong world. A tight gate against a bad prior is confirmation bias with feedback.

The fix is temporal — **identity is earned, not claimed**:

1. **Track** detections frame-to-frame in bearing space (targets drift ~1.4°/frame; continuity needs no identity at all)
2. **Score** every candidate id against the track's whole history (wrong candidates decorrelate as the swarm geometry evolves)
3. **Admit** factors only when the track is ≥5 frames old, the best id fits absolutely (<15° mean), beats the runner-up by a 2.5× ratio test (SIFT-style — scale-free under a drifting prior), *and* fits today (<10° instantaneous). Until then the track contributes nothing: a dropped detection is a missed meal; a wrong association is poison.

### The final scoreboard (degraded radio R=0.35, identical ranges everywhere)

| Run | mean RMSE | worst frame | factors | purity |
|---|---|---|---|---|
| range-only | 0.204 m | 0.422 m | — | — |
| oracle bearings (perfect id) | 0.041 m | 0.188 m | ~1,700 | 100% |
| ONNX + naive assoc | 0.322 m | 0.775 m | 1,751 | **10.7%** |
| ONNX + track assoc | **0.076 m** | 0.272 m | 366 | **82.0%** |

**366 carefully-earned factors beat 1,751 promiscuous ones by 4×.** The full real pipeline lands within 2× of the perfect-identity oracle and 2.7× better than range-only — and Huber quietly absorbs the residual 17% wrong ids.

---

## 4. Reproduce

```bash
# science
python src/vision/layer2_bearing_phase_diagram.py     # ~2 min: rescue demo + R×B sweep
python src/vision/layer2_isam2_bearing.py             # ~1 min: live iSAM2, 2 radios × 2 suites

# perception pipeline
python src/vision/layer2_make_dataset.py              # renders 1,440 frames + detector.onnx
cd rust && cargo build --release && cd ..
ORT_DYLIB_PATH=$PWD/.venv/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.1.27.0.dylib \
  ./rust/target/release/swarm-perception --frames frames --model data/detector.onnx --out bearings.jsonl
python src/vision/layer2_perception_closeloop.py      # association + 4-way iSAM2 comparison
```

## 5. What to say in the room

- *"Ranges constrain how far, bearings constrain which way — on the rigidity matrix they're literally orthogonal rows on the same edge."*
- *"Bearing constraints are linear in position, so they enter the convex relaxation with no relaxation gap — and the ray inequality is what kills the mirror flip."*
- *"One detection per drone tripled the rigid fraction at the marginal radius; the number came from a controlled sweep with nested measurement sets, not a demo."*
- *"My first associator was 85% wrong and made the estimator worse — confidently worse, because wrong bearings shrink covariance too. The fix was making identity something a track earns over five frames with a ratio test. 366 earned factors beat 1,751 claimed ones by 4×."*
- *"The covariance the supervisor gates on is only trustworthy because the graph is well-conditioned — I can show you the frame where the range-only estimator says 4 cm and is wrong by 28."*
