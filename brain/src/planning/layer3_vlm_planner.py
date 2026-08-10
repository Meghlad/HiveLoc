"""
Layer 3: a language-conditioned mission planner with a safety supervisor.

NOT a VLA. The action space here is waypoints; the low-level policy is a flight
controller better than anything we'd train; and the open problem in a GPS-denied
swarm isn't motor control, it's whether the estimate is trustworthy enough to act
on. So this layer is: operator instruction + downlinked mosaic -> Claude ->
STRUCTURED JSON plan (waypoints, spacing, per-vehicle assignments) -> the Rust
supervisor (swarm-supervisor), which is the only thing that can emit a setpoint.

Two guardrails, and the second is the point:
  1. CONSTRAINED DECODING. The model is called with output_config.format +
     a strict json_schema. Free text cannot reach the aircraft because the
     response is constrained to the Plan schema at decode time - there is no
     channel for prose.
  2. THE SUPERVISOR ASSUMES THE MODEL IS WRONG. Even a schema-valid plan is
     rejected unless every waypoint is inside the geofence, spacing holds,
     the estimator's marginal covariance for each assigned vehicle is below
     threshold, and the plan is fresh. That's ~400 lines of Rust with no model
     in it (rust/swarm-supervisor). A VLM will eventually hallucinate a
     waypoint into a hillside; the supervisor is why it never reaches MAVLink.

Uses the current swarm ESTIMATE from Layer 2 (poses + marginal covariance) as
the world state, so the planner and the supervisor gate on the SAME trust
signal the estimator produces.

Model: claude-opus-4-8, adaptive thinking. Runs offline (deterministic
geometric planner) when no API credential is available, so the pipeline is
always demonstrable - the offline path is clearly labeled and produces the
identical Plan schema.

Run:  python src/planning/layer3_vlm_planner.py "form a tight line along the north edge"
"""

import json
import os
import sys
import time
import numpy as np

# ----------------------------------------------------------------------
# World state: the live estimate from Layer 2 (poses + marginal covariance)
# ----------------------------------------------------------------------
def load_world(frame=100, condition="r055"):
    """Latest swarm estimate + per-vehicle trust from the Layer-2 iSAM2 run."""
    d = np.load("data/layer2_isam2_results.npz")
    pos = d[f"online_{condition}"][frame]        # [n, 2]
    cov = d[f"cov_{condition}"][frame]           # [n] marginal covariance trace
    return pos, cov

GEOFENCE = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]   # unit-square arena
MIN_SPACING = 0.08

# ----------------------------------------------------------------------
# The Plan schema - the ONLY shape that can leave this planner. Shared,
# byte-for-byte, with the Rust supervisor's serde-deserialized Plan struct.
# ----------------------------------------------------------------------
PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "plan_id": {"type": "string"},
        "issued_unix_ms": {"type": "integer"},
        "reasoning": {"type": "string", "maxLength": 200,
                      "description": "one line: why this formation, for the log"},
        "min_spacing_m": {"type": "number"},
        "assignments": {
            "type": "array",
            "maxItems": 64,                    # bound the array: a small local model
            "items": {                         # under a grammar can otherwise loop
                "type": "object",
                "properties": {
                    "vehicle": {"type": "integer"},
                    "waypoint_ne": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 2, "maxItems": 2,
                        # Anthropic structured outputs IGNORE these length bounds
                        # (the supervisor enforces exactly-2); Ollama's grammar
                        # ENFORCES them - which also stops the 1B runaway array.
                    },
                },
                "required": ["vehicle", "waypoint_ne"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["plan_id", "issued_unix_ms", "reasoning",
                 "min_spacing_m", "assignments"],
    "additionalProperties": False,
}

# A stripped schema for small LOCAL models: assignments ONLY, zero free-text
# fields. llama.cpp's grammar enforces array length bounds but NOT string
# maxLength, so a 1B model asked for a `reasoning` string runs away and truncates
# the JSON. Emitting only the numeric assignments keeps it fast and always valid;
# _sanitize_plan() fills plan_id / issued_unix_ms / reasoning / min_spacing_m.
# (This is also the tighter safety posture: the local model emits no prose at all.)
OLLAMA_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "assignments": {
            "type": "array",
            "maxItems": 64,
            "items": {
                "type": "object",
                "properties": {
                    "vehicle": {"type": "integer"},
                    "waypoint_ne": {"type": "array", "items": {"type": "number"},
                                    "minItems": 2, "maxItems": 2},
                },
                "required": ["vehicle", "waypoint_ne"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["assignments"],
    "additionalProperties": False,
}

SYSTEM = """You are the mission-planning stage of a GPS-denied drone swarm. You \
translate one operator instruction, plus the swarm's current estimated state, into \
a formation plan: a waypoint for each assigned vehicle.

Hard rules you must follow (a downstream safety supervisor enforces them and will \
reject the whole plan if you break any, so respect them up front):
- Every waypoint must lie strictly inside the geofence polygon you are given.
- Keep assigned vehicles at least min_spacing_m apart (you may request wider).
- Do NOT assign a vehicle whose position the estimator is uncertain about - you \
are given each vehicle's marginal covariance trace; leave high-covariance \
vehicles where they are (simply omit them from assignments).
- Coordinates are in the estimator's local NE frame (north, east), same units \
as the geofence and the reported positions.

Plan the formation the operator asked for using the vehicles you trust. Put a \
one-line justification in `reasoning`. Emit only the structured plan."""


def build_state_text(instruction, pos, cov):
    """The operator instruction + the swarm's current estimated state, as one
    prompt string. Shared by every backend (Claude, Ollama) so they all see the
    identical world."""
    n = len(pos)
    state = {
        "geofence": GEOFENCE,
        "min_spacing_m": MIN_SPACING,
        "max_trust_covariance": 0.004,
        "vehicles": [
            {"vehicle": i,
             "position_ne": [round(float(pos[i, 0]), 3), round(float(pos[i, 1]), 3)],
             "marginal_cov_trace": round(float(cov[i]), 5),
             "trusted": bool(cov[i] <= 0.004)}
            for i in range(n)
        ],
    }
    return (f"Operator instruction: {instruction}\n\n"
            f"Current swarm state (from the GPS-denied estimator):\n"
            f"{json.dumps(state, indent=2)}\n\n"
            f"Emit a formation plan.")


def build_user_content(instruction, pos, cov, mosaic_b64=None):
    content = [{"type": "text", "text": build_state_text(instruction, pos, cov)}]
    if mosaic_b64:                                    # a downlinked top-down mosaic
        content.insert(0, {"type": "image", "source": {
            "type": "base64", "media_type": "image/png", "data": mosaic_b64}})
    return content


# ----------------------------------------------------------------------
# The model call: claude-opus-4-8, constrained to PLAN_SCHEMA at decode time.
# ----------------------------------------------------------------------
def plan_with_claude(instruction, pos, cov, mosaic_b64=None):
    import anthropic
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=4000,
        thinking={"type": "adaptive"},
        system=SYSTEM,
        messages=[{"role": "user",
                   "content": build_user_content(instruction, pos, cov, mosaic_b64)}],
        # CONSTRAINED DECODING: the response is forced to the Plan schema. There
        # is no free-text channel to the aircraft - prose cannot be emitted here.
        output_config={"format": {"type": "json_schema", "schema": PLAN_SCHEMA}},
    )
    text = next(b.text for b in resp.content if b.type == "text")
    return json.loads(text), "claude-opus-4-8"


# ----------------------------------------------------------------------
# Local model call: Ollama, constrained to PLAN_SCHEMA via its structured-output
# `format` field (llama.cpp GBNF grammar). Same guarantee as Claude's constrained
# decoding - the response is forced to the Plan schema, so no free text can reach
# the supervisor. Select with env OLLAMA_MODEL (e.g. "llama3.2:1b"); host via
# OLLAMA_HOST (default http://localhost:11434). No pip dependency - stdlib urllib.
# ----------------------------------------------------------------------
def _sanitize_plan(plan, source_tag):
    """Coerce a (possibly sloppy) small-model plan into the exact shape the Rust
    supervisor parses: waypoint_ne must be EXACTLY two numbers, or serde rejects
    the whole plan before it can even be gated. Malformed assignments are dropped
    (the supervisor still decides on what's left). This keeps a weak local model
    safe - never a parse crash, always a clean accept/reject."""
    clean = []
    for a in (plan.get("assignments") or []):
        try:
            wp = a["waypoint_ne"]
            clean.append({"vehicle": int(a["vehicle"]),
                          "waypoint_ne": [float(wp[0]), float(wp[1])]})
        except (KeyError, IndexError, TypeError, ValueError):
            continue                                  # drop the malformed one
    return {
        "plan_id": str(plan.get("plan_id") or f"{source_tag}-{int(time.time())}"),
        "issued_unix_ms": int(plan.get("issued_unix_ms") or time.time() * 1000),
        "reasoning": str(plan.get("reasoning", ""))[:200],
        "min_spacing_m": float(plan.get("min_spacing_m") or MIN_SPACING),
        "assignments": clean,
    }


def plan_with_ollama(instruction, pos, cov, mosaic_b64=None):
    import urllib.request
    model = os.environ["OLLAMA_MODEL"]
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    user_msg = {"role": "user", "content": build_state_text(instruction, pos, cov)}
    if mosaic_b64:                                    # needs a vision model (llava/…)
        user_msg["images"] = [mosaic_b64]
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM}, user_msg],
        "format": OLLAMA_PLAN_SCHEMA,                 # assignments-only, always valid
        "stream": False,
        "options": {"temperature": 0, "num_predict": 1500},   # hard runaway backstop
    }
    req = urllib.request.Request(
        f"{host}/api/chat", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        body = json.loads(r.read())
    plan = json.loads(body["message"]["content"])     # `format` -> content IS JSON
    return _sanitize_plan(plan, f"ollama-{model}"), f"ollama:{model}"


# ----------------------------------------------------------------------
# Offline fallback: a deterministic geometric planner. Same Plan schema, so the
# supervisor path is identical whether the plan came from the model or from here.
# Interprets a few formation keywords; this is NOT the interesting part - the
# supervisor is. It exists so the pipeline runs with no API credential.
# ----------------------------------------------------------------------
def plan_offline(instruction, pos, cov):
    trusted = [i for i in range(len(pos)) if cov[i] <= 0.004]
    ins = instruction.lower()
    k = len(trusted)
    if k == 0:
        waypts = []
    elif "line" in ins:
        edge = 0.85 if "north" in ins else (0.15 if "south" in ins else 0.5)
        xs = np.linspace(0.15, 0.85, k)
        waypts = [[edge, x] if ("north" in ins or "south" in ins)
                  else [x, edge] for x in xs]
    elif "circle" in ins or "ring" in ins or "orbit" in ins:
        ang = np.linspace(0, 2 * np.pi, k, endpoint=False)
        waypts = [[0.5 + 0.3 * np.cos(a), 0.5 + 0.3 * np.sin(a)] for a in ang]
    elif "grid" in ins or "spread" in ins:
        side = int(np.ceil(np.sqrt(k)))
        gx = np.linspace(0.2, 0.8, side)
        waypts = [[gx[j % side], gx[j // side]] for j in range(k)]
    else:                                             # "hold" / unknown: gentle recenter
        waypts = [[np.clip(pos[i, 0], 0.1, 0.9), np.clip(pos[i, 1], 0.1, 0.9)]
                  for i in trusted]
    return {
        "plan_id": f"offline-{int(time.time())}",
        "issued_unix_ms": int(time.time() * 1000),
        "reasoning": f"deterministic geometric planner: '{instruction}' over "
                     f"{k} trusted vehicles",
        "min_spacing_m": MIN_SPACING,
        "assignments": [{"vehicle": int(v), "waypoint_ne": [round(w[0], 3), round(w[1], 3)]}
                        for v, w in zip(trusted, waypts)],
    }, "offline-geometric"


def make_plan(instruction, pos, cov, mosaic_b64=None):
    """Pick a planner, always fall back to the deterministic one so the pipeline
    is never dead. Priority: local Ollama (if OLLAMA_MODEL set) -> Claude (if a
    credential is present) -> offline geometric. Every path emits the identical
    Plan schema and goes through the same supervisor."""
    if os.environ.get("OLLAMA_MODEL"):
        try:
            return plan_with_ollama(instruction, pos, cov, mosaic_b64)
        except Exception as e:                        # server down, model error, ...
            print(f"[planner] Ollama call failed ({type(e).__name__}: {e}); "
                  f"using offline planner", file=sys.stderr)
        return plan_offline(instruction, pos, cov)

    have_cred = os.environ.get("ANTHROPIC_API_KEY") or os.path.exists(
        os.path.expanduser("~/.config/anthropic"))
    if have_cred:
        try:
            return plan_with_claude(instruction, pos, cov, mosaic_b64)
        except Exception as e:                        # network, auth, quota, ...
            print(f"[planner] Claude call failed ({type(e).__name__}: {e}); "
                  f"using offline planner", file=sys.stderr)
    else:
        print("[planner] no ANTHROPIC_API_KEY / profile; using offline planner",
              file=sys.stderr)
    return plan_offline(instruction, pos, cov)


# ----------------------------------------------------------------------
# Hand the plan + world state to the Rust supervisor and print its verdict.
# ----------------------------------------------------------------------
def supervise(plan, pos, cov, emit=False):
    import subprocess, tempfile, pathlib
    root = pathlib.Path(__file__).parent
    sup = root / "rust/target/release/swarm-supervisor"
    if not sup.exists():
        print("[supervisor] binary not built; run: "
              "cargo build --release --manifest-path rust/Cargo.toml", file=sys.stderr)
        return None
    est = {"frame_index": 100, "stamp_unix_ms": int(time.time() * 1000),
           "pos": pos.tolist(), "cov_trace": cov.tolist()}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as pf, \
         tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as ef:
        json.dump(plan, pf); json.dump(est, ef)
        plan_path, est_path = pf.name, ef.name
    cmd = [str(sup), "--plan", plan_path, "--estimate", est_path]
    if emit:
        cmd.append("--emit")
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode, r.stdout, r.stderr


if __name__ == "__main__":
    instruction = " ".join(sys.argv[1:]) or "form a tight line along the north edge"
    pos, cov = load_world()
    print(f"=== instruction: {instruction!r} ===")
    print(f"swarm: {len(pos)} vehicles, "
          f"{int((cov <= 0.004).sum())} trusted (cov <= 0.004)\n")

    plan, source = make_plan(instruction, pos, cov)
    print(f"--- PLAN (source: {source}) ---")
    print(json.dumps(plan, indent=2))

    verdict = supervise(plan, pos, cov)
    if verdict is not None:
        code, out, err = verdict
        print(f"\n--- SUPERVISOR VERDICT (exit {code}) ---")
        print(out.strip())
        if err.strip():
            print(err.strip())
