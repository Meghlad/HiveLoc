"""The hand-off to the native Rust supervisor.

`run_supervisor.sh --loop` runs the real gate natively on the host, reading
`/tmp/hive/estimate.json` and `/tmp/hive/plan.json`. The gate is a separate
process — a one-shot binary re-invoked per plan — so the contract between it and
the Python loop is the filesystem: this module writes the two JSON documents the
Rust binary already parses (`EstimateSnapshot`, `Plan` — field names mirrored
exactly in hive.supervisor_gate).

Nothing here transmits. That property is the whole architecture: the Python side
computes and pre-validates, the Rust side is the only component that can emit a
SET_POSITION_TARGET_LOCAL_NED.

THE FRAME CHANGES HERE, AND ONLY HERE. Everything upstream — frames.py,
the estimator, the FSM, the guards, the geofence in supervisor.json — is ENU
(x = East, y = North), because TacFrame is ENU about the surveyed anchor. The
Rust struct field is `waypoint_ne` and `main.rs` maps `[0] -> x (North)`,
`[1] -> y (East)`. Writing ENU into it mirrors every commanded position about
the x=y line: an 8 m easting becomes an 8 m northing and the vehicle flies at
right angles to the plan, inside the fence and past every gate. The swap is
applied on the way out and nowhere else, so there is exactly one place to look
when a coordinate seems rotated.

Writes are atomic (tmp file + os.replace) because the supervisor polls these
paths on its own clock. A half-written plan that parses is worse than one that
does not — os.replace makes a torn read impossible rather than unlikely.
"""

from __future__ import annotations

import json
import os
import pathlib
import time

HIVE_DIR = pathlib.Path(os.environ.get("HIVE_SHARED_DIR", "/tmp/hive"))
ESTIMATE_PATH = HIVE_DIR / "estimate.json"
PLAN_PATH = HIVE_DIR / "plan.json"
DECISION_PATH = HIVE_DIR / "decision.json"


def now_ms() -> int:
    """Absolute unix time, for stamps the Rust supervisor compares against.

    Wall clock, because that is the contract: `EstimateSnapshot.stamp_unix_ms`
    and `Plan.issued_unix_ms` are unix milliseconds and the gate subtracts them
    from its own `SystemTime::now()`. Do NOT use this to measure a duration —
    see mono_ms().
    """
    return int(time.time() * 1000)


def mono_ms() -> int:
    """Elapsed time, for every age/staleness decision inside a process.

    THE WALL CLOCK IS NOT MONOTONIC, and on some hosts it is not even close.
    Measured on WSL2 while bringing this stack up: `time.time()` jumped +6.4 s
    and back -6.9 s roughly every 5.5 s, indefinitely, because systemd-timesyncd
    and the Hyper-V PTP device were both steering it. Freshness checks built on
    it therefore saw a dead pose stream every few seconds, the estimator
    correctly refused to publish, the FSM correctly held, and the fleet never
    moved — a total mission failure caused entirely by the clock, with every
    component behaving exactly as designed. The M1 host does not have that
    pathology, but the lesson outlives the host that taught it.

    CLOCK_MONOTONIC_RAW, not CLOCK_MONOTONIC. Plain monotonic never steps
    backward but it IS rate-adjusted by NTP, and on that same host it ran ~9%
    off RAW while the two time services fought. A 9% error on a 200 ms tick is
    not fatal on its own; it becomes fatal when it is the number a slew budget
    or a staleness bound is computed from. RAW is the hardware counter and is
    immune to both steps and slewing.

    Absolute stamps still have to be wall clock for the cross-process contract
    in now_ms(), which is why both exist and why they must not be mixed.
    """
    return int(time.clock_gettime(time.CLOCK_MONOTONIC_RAW) * 1000)


def enu_to_ne(xy) -> list[float]:
    """ENU (East, North) -> the Rust gate's (North, East)."""
    return [float(xy[1]), float(xy[0])]


def _write_atomic(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    # allow_nan=False: json.dumps would otherwise emit bare Infinity/NaN tokens,
    # which serde_json rejects — taking the supervisor down with a parse error
    # for the WHOLE fleet instead of excluding one vehicle. Upstream uses a
    # large finite sentinel; this is the backstop that makes that a hard error
    # here rather than a silent outage there.
    tmp.write_text(json.dumps(payload, allow_nan=False))
    os.replace(tmp, path)


def publish_estimate(est) -> None:
    payload = est.to_json()
    payload["pos"] = [enu_to_ne(p) for p in payload["pos"]]
    _write_atomic(ESTIMATE_PATH, payload)


def publish_plan(plan) -> None:
    payload = plan.to_json()
    for a in payload["assignments"]:
        a["waypoint_ne"] = enu_to_ne(a["waypoint_ne"])
    _write_atomic(PLAN_PATH, payload)


def read_decision() -> dict | None:
    """The supervisor's last verdict, or None if it has not written one yet."""
    try:
        return json.loads(DECISION_PATH.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None
