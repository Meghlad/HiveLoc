"""H2.2 — rotate a signing key without bricking the link that carries it.

`COMMS_HARDENING_PLAN.md` §3.3 asks for "a rotation schedule and an
on-compromise rotation path". The schedule is policy and lives in
`docs/provisioning.md`; the path is this file, and it is more delicate than it
first appears because **the key being replaced is the key authenticating the
message that replaces it.**

That is the whole problem in one sentence. Every other credential rotation you
have done had an out-of-band channel: a console, an SSH session, a web login
that still worked while the old token was revoked. Here the MAVLink link *is*
the channel, and a rotation that half-lands leaves an airframe that will not
accept a single further command from you. The recovery is then physical —
`disable_signing()` if it can still hear you at all, and erasing FRAM if it
cannot.

---------------------------------------------------------------------------
TWO-PHASE COMMIT, AND WHY THE KEYSTORE IS WRITTEN LAST
---------------------------------------------------------------------------

The naive order — generate, save, push — is wrong in a way that is invisible
until the day it matters. If the push fails after the save, the keystore now
records a key the vehicle never received. Next run, the ground station signs
with a key the airframe does not know, every frame is refused, and the symptom
(a silent, dead, authenticated-looking link) is identical to the symptom of a
swapped or tampered airframe. You cannot tell a lost packet from an attack.

So:

    1. generate a candidate in memory                    (nothing on disk yet)
    2. push it with SETUP_SIGNING, signed with the OLD key
       — which the vehicle still holds, so the re-key is authenticated
         end to end, and an attacker without the old key cannot rotate us
    3. switch our own codec to the candidate
    4. PROVE the vehicle answers on it — a signed round trip, not an assumption
    5. only now write it to the keystore                 (`Keystore.commit`)

If step 4 fails, `_recover` puts our side back on the old key and probes again.
Two outcomes, both survivable and both reported:

  * the old key still works  → the vehicle never took the candidate. Nothing
    was written; the fleet is exactly as it was. Retry is safe.
  * neither key works        → the link is down for a reason rotation cannot
    fix. Reported loudly with the recovery path, and still nothing was written,
    so the keystore continues to describe the last key known to work.

`SETUP_SIGNING` has no ACK and the link is UDP, which is why the push repeats.
It is also refused while armed (`GCS_Signing.cpp:74`) with nothing but a
STATUSTEXT to say so — so the armed check happens here, loudly, rather than
being discovered as a silent no-op afterwards.
"""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass, field

from .audit_log import NullAuditLog
from .enable_signing import (MAV_CMD_REQUEST_MESSAGE, MAVLINK_MSG_ID_SYSTEM_TIME,
                             SigningConfig, enable_link_signing,
                             mavlink_timestamp)
from .keystore import KEY_BYTES, Keystore


@dataclass
class RotationResult:
    vehicle: int
    ok: bool
    reason: str = ""
    old_fingerprint: str = ""
    new_fingerprint: str = ""
    rolled_back: bool = False
    link_dead: bool = False

    def to_json(self) -> dict:
        return {"vehicle": int(self.vehicle), "ok": bool(self.ok),
                "reason": self.reason,
                "old_fingerprint": self.old_fingerprint,
                "new_fingerprint": self.new_fingerprint,
                "rolled_back": bool(self.rolled_back),
                "link_dead": bool(self.link_dead)}


@dataclass
class RotationPolicy:
    """When a key is considered too old to keep using.

    Age is tracked in the audit log rather than the keystore: the keystore holds
    secrets and should hold as little else as possible, and the log is already
    the tamper-evident record of every previous rotation. `due()` therefore
    takes the last-rotation time from the caller, who reads it from the log.
    """

    max_age_s: float = 30 * 24 * 3600.0     # 30 days, a starting point
    rotate_on_compromise: bool = True

    def due(self, last_rotation_unix: float | None,
            now: float | None = None) -> bool:
        if last_rotation_unix is None:
            return True
        now = time.time() if now is None else now
        return (now - float(last_rotation_unix)) >= self.max_age_s


def _fingerprint(key: bytes) -> str:
    import hashlib
    return hashlib.sha256(key).hexdigest()[:8]


def verify_signed_link(link, timeout_s: float = 3.0) -> bool:
    """Prove the far end accepts frames signed with our current key.

    Sends a signed `MAV_CMD_REQUEST_MESSAGE` (harmless — it asks for
    `SYSTEM_TIME` and nothing else) and waits for the `COMMAND_ACK`. An ack can
    only follow a signature the vehicle verified, so this is positive proof of a
    shared key rather than an absence of complaints.

    The inverse of `enable_signing.probe_unsigned_rejected`: that one proves an
    unsigned command is refused, this one proves a signed command is accepted.
    A link needs both to be called healthy, and neither implies the other.
    """
    while link.recv_match(type="COMMAND_ACK", blocking=False) is not None:
        pass
    link.mav.command_long_send(
        link.target_system, link.target_component,
        MAV_CMD_REQUEST_MESSAGE, 0,
        float(MAVLINK_MSG_ID_SYSTEM_TIME), 0, 0, 0, 0, 0, 0)

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        ack = link.recv_match(type="COMMAND_ACK", blocking=True, timeout=0.25)
        if ack is not None and ack.command == MAV_CMD_REQUEST_MESSAGE:
            return True
    return False


def _push(link, key: bytes, repeats: int, settle_s: float) -> int:
    """Send SETUP_SIGNING under whatever key the link is currently using."""
    stamp = mavlink_timestamp()
    for _ in range(repeats):
        link.mav.setup_signing_send(link.target_system, link.target_component,
                                    bytearray(key), stamp)
        time.sleep(settle_s)
    return stamp


def rotate_vehicle(link, keystore: Keystore, vehicle_id: int,
                   cfg: SigningConfig | None = None, armed: bool | None = None,
                   repeats: int = 3, settle_s: float = 0.2,
                   timeout_s: float = 3.0, audit=None) -> RotationResult:
    """Replace one vehicle's signing key, verified before it is written down.

    `armed` is the caller's knowledge of vehicle state. Passing None skips the
    check, which is only correct offline — a rotation attempted against an armed
    airframe silently does nothing.
    """
    cfg = cfg or SigningConfig()
    audit = audit if audit is not None else NullAuditLog()
    vid = int(vehicle_id)

    if armed:
        audit.append("key_rotation_refused", vehicle=vid, reason="armed")
        return RotationResult(vid, False, "vehicle is armed: ArduPilot "
                              "discards SETUP_SIGNING while armed and only "
                              "warns by STATUSTEXT")

    old = keystore.key(vid)
    old_fp = _fingerprint(old)
    candidate = secrets.token_bytes(KEY_BYTES)
    new_fp = _fingerprint(candidate)

    audit.append("key_rotation_started", vehicle=vid, old_fingerprint=old_fp,
                 new_fingerprint=new_fp)

    # Phase 2 — carry the candidate, authenticated by the key being retired.
    _push(link, candidate, repeats, settle_s)

    # Phase 3 — our side moves to the candidate.
    enable_link_signing(link, candidate, cfg,
                        initial_timestamp=mavlink_timestamp())

    # Phase 4 — proof, not assumption.
    if verify_signed_link(link, timeout_s):
        keystore.commit(vid, candidate)          # Phase 5, and only now.
        audit.append("key_rotated", vehicle=vid, old_fingerprint=old_fp,
                     new_fingerprint=new_fp)
        return RotationResult(vid, True, "verified", old_fp, new_fp)

    return _recover(link, keystore, vid, old, old_fp, new_fp, cfg, repeats,
                    settle_s, timeout_s, audit)


def _recover(link, keystore, vid, old, old_fp, new_fp, cfg, repeats, settle_s,
             timeout_s, audit) -> RotationResult:
    """The candidate did not verify. Put the link back if it can be put back."""
    enable_link_signing(link, old, cfg, initial_timestamp=mavlink_timestamp())

    if verify_signed_link(link, timeout_s):
        audit.append("key_rotation_rolled_back", vehicle=vid,
                     old_fingerprint=old_fp, new_fingerprint=new_fp,
                     reason="candidate never took; old key still live")
        return RotationResult(
            vid, False,
            "candidate was not accepted; rolled back to the previous key, "
            "which still works. Nothing was written to the keystore.",
            old_fp, new_fp, rolled_back=True)

    # Neither key answers. One more attempt: re-seat the old key by pushing it
    # as a fresh SETUP_SIGNING. If the vehicle actually did take the candidate
    # and we simply never heard the ack, our codec is on the old key and this
    # push will not verify either — but it costs three packets to rule out the
    # far more likely case, which is that the candidate push was lost and the
    # vehicle is sitting on a key we still hold.
    _push(link, old, repeats, settle_s)
    if verify_signed_link(link, timeout_s):
        audit.append("key_rotation_recovered", vehicle=vid,
                     old_fingerprint=old_fp)
        return RotationResult(
            vid, False, "candidate failed; old key re-seated and verified",
            old_fp, new_fp, rolled_back=True)

    audit.append("key_rotation_link_dead", vehicle=vid,
                 old_fingerprint=old_fp, new_fingerprint=new_fp)
    return RotationResult(
        vid, False,
        "LINK DEAD: neither the old nor the new key gets a response. The "
        "keystore still holds the old key, so nothing is lost on this side. "
        "Recover with security.enable_signing.disable_signing() if the vehicle "
        "can still hear anything, otherwise erase its FRAM.",
        old_fp, new_fp, link_dead=True)


@dataclass
class FleetRotation:
    results: list[RotationResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(r.ok for r in self.results)

    @property
    def dead_links(self) -> list[int]:
        return [r.vehicle for r in self.results if r.link_dead]

    def to_json(self) -> dict:
        return {"ok": self.ok, "dead_links": self.dead_links,
                "results": [r.to_json() for r in self.results]}


def rotate_fleet(links, keystore: Keystore, cfg: SigningConfig | None = None,
                 armed=None, audit=None, stop_on_dead: bool = True,
                 verbose: bool = True) -> FleetRotation:
    """Rotate every vehicle, one at a time, stopping if one goes dark.

    Sequential and interruptible on purpose. Rotating a fleet in parallel means
    a systematic fault — a bad build, a wrong keystore, a firmware that refuses
    the message — takes every airframe off the air simultaneously. One at a
    time, with `stop_on_dead`, the same fault costs exactly one vehicle and
    leaves the rest reachable to be diagnosed with.
    """
    audit = audit if audit is not None else NullAuditLog()
    out = FleetRotation()
    for i, link in enumerate(links):
        is_armed = None if armed is None else bool(armed[i])
        res = rotate_vehicle(link, keystore, i, cfg, armed=is_armed,
                             audit=audit)
        out.results.append(res)
        if verbose:
            state = "ok" if res.ok else ("DEAD" if res.link_dead else "failed")
            print(f"  vehicle {i}: {state}  {res.old_fingerprint} -> "
                  f"{res.new_fingerprint}  {res.reason}")
        if res.link_dead and stop_on_dead:
            if verbose:
                print("  stopping: one link is dead, refusing to risk the rest")
            break
    return out
