"""
Layer 2 (part C, step 1): render the swarm's onboard camera frames + build the
ONNX detector the Rust perception node runs.

WHAT GETS RENDERED. For every frame t and every drone i, a 320x240 grayscale
image from i's forward-mounted camera (pinhole, 90 deg horizontal FOV, fx=160):
  - each neighbor j within R_CAM and inside the FOV becomes a Gaussian blob at
    column u = cx + fx*tan(bearing_body), row v = horizon +- relative altitude;
    closer targets are BIGGER and BRIGHTER (that's why "nearest-first" was the
    right candidate order in part A);
  - with prob 1-P_DET the blob is NOT rendered (motion blur / occlusion - the
    physical origin of part A's detection probability);
  - with small prob a CLUTTER blob appears (bird, glint) - a false positive the
    downstream data association must survive;
  - sensor noise on top.

WHAT THE DETECTOR IS. detector.onnx is a difference-of-Gaussians blob detector
expressed as a single-channel Conv->ReLU ONNX graph with FIXED weights. It is
deliberately not a trained network: the contract (image in -> response map out)
is identical to what a trained head would satisfy, the weights are just math we
can verify. Swapping in a learned model later changes one file, zero code.

WHAT LEAVES THIS SCRIPT.
  frames/*.png            the camera frames  (what the Rust node consumes)
  frames/meta.jsonl       per-image: heading + intrinsics (legit onboard info)
                          + ground-truth target table (EVAL ONLY - the Rust
                          node never reads the truth fields)
  detector.onnx           the fixed-weight conv detector
Run:  python src/vision/layer2_make_dataset.py
"""

import json
import pathlib
import numpy as np
import onnx
from onnx import helper, TensorProto

# ---- camera + world constants (must match layer2_isam2_bearing.py) ----------
W, H = 320, 240
FX = FY = 160.0                 # 90 deg horizontal FOV: fx = (W/2)/tan(45)
CX, CY = 160.0, 120.0
FOV = np.deg2rad(90)
R_CAM = 0.65
P_DET = 0.9
P_CLUTTER = 0.05
BG_MEAN, BG_STD = 0.07, 0.025

true_traj = np.load("data/trajectory.npy")
T, n = true_traj.shape[0], true_traj.shape[1]
rng = np.random.default_rng(21)

# forward-mounted lens looks along the velocity vector (same as part B)
vel = np.gradient(true_traj, axis=0)
headings = np.arctan2(vel[..., 1], vel[..., 0])

# small static altitude offsets so targets sit near, not on, the horizon
alt = rng.normal(0.0, 0.03, n)

out = pathlib.Path("frames")
out.mkdir(exist_ok=True)


def render(t, i):
    """One onboard frame + its ground-truth table."""
    img = rng.normal(BG_MEAN, BG_STD, (H, W)).clip(0, 1)
    truth = []
    Xt = true_traj[t]
    for j in range(n):
        if j == i:
            continue
        diff = Xt[j] - Xt[i]
        d = float(np.linalg.norm(diff))
        if d > R_CAM:
            continue
        theta = float(np.arctan2(diff[1], diff[0]))
        rel = (theta - headings[t, i] + np.pi) % (2 * np.pi) - np.pi
        if abs(rel) > FOV / 2:
            continue
        u = CX + FX * np.tan(rel)
        v = CY + FY * (alt[j] - alt[i]) / d
        rendered = bool((rng.random() < P_DET) and (0 <= u < W) and (0 <= v < H))
        truth.append(dict(id=j, u=round(float(u), 2), v=round(float(v), 2),
                          d=round(d, 3), bearing_world=round(theta, 5),
                          rendered=rendered))
        if rendered:
            amp = min(1.0, 0.45 + 0.12 / d)
            sig = 1.3 + 0.4 / d
            yy, xx = np.mgrid[0:H, 0:W]
            img += amp * np.exp(-(((xx - u) ** 2 + (yy - v) ** 2) / (2 * sig ** 2)))
    if rng.random() < P_CLUTTER:                       # a bird, a glint
        cu, cv = rng.uniform(10, W - 10), rng.uniform(10, H - 10)
        yy, xx = np.mgrid[0:H, 0:W]
        img += rng.uniform(0.3, 0.8) * np.exp(
            -(((xx - cu) ** 2 + (yy - cv) ** 2) / (2 * rng.uniform(1.2, 2.5) ** 2)))
    return img.clip(0, 1), truth


print(f"rendering {T} frames x {n} drones = {T*n} images ...")
try:
    from PIL import Image
except ImportError:
    raise SystemExit("pip install pillow")

with open(out / "meta.jsonl", "w") as meta:
    for t in range(T):
        for i in range(n):
            img, truth = render(t, i)
            name = f"t{t:03d}_d{i:02d}.png"
            Image.fromarray((img * 255).astype(np.uint8), "L").save(out / name)
            meta.write(json.dumps(dict(
                file=name, t=t, observer=i,
                heading=round(float(headings[t, i]), 5),
                K=dict(fx=FX, fy=FY, cx=CX, cy=CY),
                truth=truth)) + "\n")
        if t % 20 == 0:
            print(f"  t={t} done")
print("saved frames/*.png + frames/meta.jsonl")

# ----------------------------------------------------------------------
# The ONNX detector: 9x9 difference-of-Gaussians conv, ReLU. Fixed weights.
# ----------------------------------------------------------------------
K = 9
ax = np.arange(K) - K // 2
xx, yy = np.meshgrid(ax, ax)
g = lambda s: np.exp(-(xx ** 2 + yy ** 2) / (2 * s ** 2))
dog = g(1.5) / g(1.5).sum() - g(3.0) / g(3.0).sum()    # center-on, surround-off
w = dog.astype(np.float32).reshape(1, 1, K, K)

graph = helper.make_graph(
    nodes=[
        helper.make_node("Conv", ["image", "dog_kernel"], ["response_raw"],
                         pads=[K // 2] * 4, name="dog_conv"),
        helper.make_node("Relu", ["response_raw"], ["response"], name="relu"),
    ],
    name="dog_blob_detector",
    inputs=[helper.make_tensor_value_info("image", TensorProto.FLOAT,
                                          [1, 1, H, W])],
    outputs=[helper.make_tensor_value_info("response", TensorProto.FLOAT,
                                           [1, 1, H, W])],
    initializer=[helper.make_tensor("dog_kernel", TensorProto.FLOAT,
                                    w.shape, w.flatten())],
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)],
                          producer_name="coop-swarm-layer2")
model.ir_version = 8
onnx.checker.check_model(model)
onnx.save(model, "data/detector.onnx")
print("saved detector.onnx (fixed-weight DoG conv, opset 17)")

# ---- sanity: run it in onnxruntime on one frame, check peaks near truth -----
import onnxruntime as ort_py
sess = ort_py.InferenceSession("data/detector.onnx")
img = np.asarray(Image.open(out / "t010_d00.png"), np.float32)[None, None] / 255
resp = sess.run(None, {"image": img})[0][0, 0]
with open(out / "meta.jsonl") as f:
    for line in f:
        m = json.loads(line)
        if m["file"] == "t010_d00.png":
            break
peaks = np.argwhere(resp > 0.05)
print(f"sanity t010_d00: {len(m['truth'])} truth targets "
      f"({sum(x['rendered'] for x in m['truth'])} rendered), "
      f"{len(peaks)} response pixels > 0.05, max resp {resp.max():.3f}")
