"""H1.1 — MAVLink 2 signing on every link, and unsigned frames refused.

Stage H0 of `COMMS_HARDENING_PLAN.md`, and the plan is right that it is the
highest value per hour available: without it the D4 loop is plaintext,
unauthenticated MAVLink over UDP, so anyone who can put a packet on the
loopback interface or the eventual radio link can inject a
`SET_POSITION_TARGET_LOCAL_NED` or — much worse, per §1.3 — a forged
`VISION_POSITION_ESTIMATE` and redefine where the vehicle believes it is.

Signing appends an HMAC-SHA256-derived 6-byte signature, a link id and a
48-bit timestamp to every frame. Authenticity kills injection; the monotonic
per-stream timestamp kills replay. That is the entire mechanism, and it is
cheap: `sign_packet` is one SHA-256 over a sub-300-byte buffer, microseconds
against the 25 ms loop budget of D4.7.

---------------------------------------------------------------------------
THE FINDING THAT DECIDES WHETHER ANY OF THIS WORKS: SERIAL0 IS CHANNEL 0
---------------------------------------------------------------------------

ArduPilot does not gate unsigned frames with a parameter. It uses a compiled-in
callback, `accept_unsigned_callback` in `libraries/GCS_MAVLink/GCS_Signing.cpp`
(verified in the local tree, lines 114-127), which reads:

    if (status == mavlink_get_channel_status(MAVLINK_COMM_0)) {
        // always accept channel 0, assumed to be secure channel. This
        // is USB on PX4 boards
        return true;
    }

Channel 0 accepts **every unsigned frame, unconditionally**, on the assumption
that it is a physically-secure USB console. Our closed loop is not on USB. The
loop originally ran on `--serial0 udpclient:127.0.0.1:<port>`, which IS
`MAVLINK_COMM_0` — so provisioning a key hardened our *outgoing* frames and our
*inbound* parsing, and hardened the vehicle's inbound path **not at all**. An
attacker could still inject unsigned commands into serial0 and ArduPilot would
act on every one, while every counter in `signing_report()` looked healthy.

**RESOLVED — the fix was transport, not code.** `sim/run_fleet.sh` now launches
each instance with `--serial0 tcp:0` (left as the local console it is assumed to
be) and `--serial2 udpclient:127.0.0.1:<port>` carrying the closed loop, with
serial1 reserved for the read-only GCS observer. serial2 lands on channel 1 or
2 — either way non-zero, which is the only thing that matters. See that script's
transport comment for why `tcp:0` has to be passed explicitly.

`probe_unsigned_rejected()` below is what proves it, and it is run by default
during provisioning (`--no-probe-unsigned` opts out). It is the check that stops
this file from being security theatre: a failure means the loop has drifted back
onto channel 0, and it should be treated as a transport bug rather than a
nuisance.

The rest of the vehicle-side behaviour, from the same file:

  * `handle_setup_signing` refuses while armed (line 74) — provision on the
    ground, and rotate on the ground.
  * The key lands in FRAM and activates on every channel immediately (94-100).
  * On load the vehicle sets its own timestamp to stored + 60 s (line 144) to
    close the replay window across a reboot, so its stream starts *ahead* of
    ours. Ahead is fine; only a stream more than a minute *behind* is refused.
  * An all-zero key with a zero timestamp disables signing (148-159). That is
    the documented recovery path and `disable_signing()` uses exactly it.

---------------------------------------------------------------------------
THE SECOND FINDING: OUR SIDE IS REPLAY-POISONABLE, THE VEHICLE'S IS NOT
---------------------------------------------------------------------------

The C and Python implementations check the same two things in opposite orders,
and the order is load-bearing.

`mavlink_helpers.h::mavlink_signature_check` (the vehicle) verifies the HMAC
first and returns before touching any stream state. A forged frame changes
nothing.

`pymavlink`'s `MAVLink.check_signature` (us) reads the timestamp, writes
`stream_timestamps[(link_id, sysid, compid)] = timestamp`, and only then
compares the signature. So one injected frame carrying a far-future timestamp
and a garbage signature is rejected — and permanently raises the bar on that
stream, after which every *legitimate* frame from that vehicle is dropped as an
old timestamp. Reproduced offline; pinned in `tests/test_reject_unsigned.py`.

That is a one-packet, Tier-1 denial of the estimator's ground-truth and
telemetry feed, which is worse than it sounds because `ground_truth_bridge`
going quiet stops the whole D4 loop. `harden_replay_window()` replaces the
bound method with one that mirrors the C order — signature first, stream state
only on success — and `enable_link_signing()` applies it by default.

---------------------------------------------------------------------------
WHAT A REJECTION LOOKS LIKE AT RUNTIME
---------------------------------------------------------------------------

`mavutil` sets `robust_parsing = True`, so a refused frame does not raise out
of `pump()`; it becomes a `BAD_DATA` message and the loop keeps running. Good
for availability, bad for observability — a rejected forgery is otherwise
indistinguishable from line noise. Hence `on_reject`: every unsigned frame that
the callback turns away is counted and handed to a sink, which is the hook
§6.1's security event log attaches to when stage H4 arrives.
"""

from __future__ import annotations

import hashlib
import hmac
import struct
import time
import types
from dataclasses import dataclass, field

from .keystore import KEY_BYTES, Keystore

# MAVLink epoch: signing timestamps are 10 microsecond units since 1/1/2015.
MAVLINK_EPOCH_UNIX = 1420070400
TIMESTAMP_UNITS_PER_SECOND = 100 * 1000

# Mirrors ArduPilot's `accept_list` (GCS_Signing.cpp:109-112). These two carry
# SiK radio link statistics, are injected by the radio itself rather than by
# the peer, and can never be signed because the radio holds no key. Refusing
# them costs the RSSI/noise telemetry that §6.1 wants for jamming detection,
# so they are allowed by id and by id only — nothing else rides in on this.
# Pinned against the pymavlink dialect in tests/test_reject_unsigned.py.
MAVLINK_MSG_ID_RADIO_STATUS = 109
MAVLINK_MSG_ID_RADIO = 166
DEFAULT_UNSIGNED_ALLOWLIST = frozenset({MAVLINK_MSG_ID_RADIO_STATUS,
                                        MAVLINK_MSG_ID_RADIO})

# A read-only command, used to ask the vehicle whether it will act on an
# unsigned frame. Deliberately one with no effect beyond an extra message:
# the probe must be safe to run against an airframe on the ground with props
# fitted, so arming or mode commands are not candidates.
MAV_CMD_REQUEST_MESSAGE = 512
MAVLINK_MSG_ID_SYSTEM_TIME = 2


def mavlink_timestamp(now_unix: float | None = None) -> int:
    """Wall clock -> the signing timestamp, in 10 us units since 2015.

    Wall clock is correct here and `supervisor_io.mono_ms()` is not: this
    number is compared against a peer's clock and against a value the vehicle
    persisted to FRAM across a reboot, so it has to be an absolute epoch that
    both sides agree on. It is never used to measure a duration.
    """
    now = time.time() if now_unix is None else now_unix
    return int(max(now, MAVLINK_EPOCH_UNIX) - MAVLINK_EPOCH_UNIX) * TIMESTAMP_UNITS_PER_SECOND


@dataclass
class RejectionLog:
    """Every frame the signing layer turned away, counted by reason.

    The counters that matter are `unsigned` and `bad_signature`: a burst of
    either is §6.3's "someone is trying", and a steady trickle of `unsigned`
    against a link you believe is provisioned means one end lost its key.
    """

    unsigned: int = 0
    bad_signature: int = 0
    allowed_unsigned: int = 0
    events: list = field(default_factory=list)
    max_events: int = 200

    def note_unsigned(self, msg_id: int, allowed: bool) -> None:
        if allowed:
            self.allowed_unsigned += 1
            return
        self.unsigned += 1
        if len(self.events) < self.max_events:
            self.events.append({"kind": "unsigned", "msg_id": int(msg_id),
                                "unix": time.time()})

    def to_json(self) -> dict:
        return {"unsigned": self.unsigned, "bad_signature": self.bad_signature,
                "allowed_unsigned": self.allowed_unsigned,
                "events": list(self.events)}


@dataclass
class SigningConfig:
    """How strict this link is, and what it is allowed to let through."""

    # Refuse unsigned frames. There is no reason to ever set this True on a
    # flight link: signing that still accepts unsigned frames authenticates
    # nothing, because the attacker simply omits the signature (plan §2.1).
    allow_unsigned: bool = False
    unsigned_allowlist: frozenset = DEFAULT_UNSIGNED_ALLOWLIST
    sign_outgoing: bool = True
    # See the second finding in the module docstring. Off only to reproduce
    # the stock behaviour in a test.
    harden_replay: bool = True
    # None lets pymavlink auto-increment one per link, which is what we want
    # for a fleet: distinct link ids keep each vehicle's replay window its own.
    link_id: int | None = None


# -- reject-unsigned ---------------------------------------------------------
def make_allow_unsigned_callback(cfg: SigningConfig, log: RejectionLog):
    """The callback pymavlink consults for every unsigned frame.

    Signature is `(mav, msg_id) -> bool`, matching `MAVLink.decode`. Returning
    False makes the frame BAD_DATA. Note that pymavlink only counts a rejection
    in `signing.reject_count` when a callback exists — with no callback at all
    the frame is still refused but invisibly, which is why one is always
    installed even in the strictest configuration.
    """

    def allow_unsigned(mav, msg_id: int) -> bool:
        allowed = cfg.allow_unsigned or int(msg_id) in cfg.unsigned_allowlist
        log.note_unsigned(msg_id, allowed)
        return allowed

    return allow_unsigned


def _strict_check_signature(mav, msgbuf, srcSystem: int, srcComponent: int) -> bool:
    """`check_signature` with the C implementation's ordering.

    Byte layout is pymavlink's and unchanged: the trailing 13 bytes are
    link_id (1) + timestamp (6) + signature (6), and the signature covers
    everything up to itself. The only difference from the stock method is
    that no stream timestamp is recorded until the HMAC has verified, so a
    forged frame cannot move the replay window. Mirrors
    `mavlink_helpers.h::mavlink_signature_check`.
    """
    if mav.signing.secret_key is None:
        return False

    h = hashlib.new("sha256")
    h.update(mav.signing.secret_key)
    h.update(msgbuf[:-6])
    # compare_digest rather than ==. The signature is only 6 bytes and an
    # attacker who can time our rejections can walk it a byte at a time.
    if not hmac.compare_digest(h.digest()[:6], bytes(msgbuf[-6:])):
        return False

    link_id = msgbuf[-13]
    tlow, thigh = struct.unpack("<IH", bytes(msgbuf[-12:-6]))
    timestamp = tlow + (thigh << 32)

    stream_key = (link_id, srcSystem, srcComponent)
    if stream_key in mav.signing.stream_timestamps:
        if timestamp <= mav.signing.stream_timestamps[stream_key]:
            return False                      # replay, or a reordered frame
    elif timestamp + 6000 * 1000 < mav.signing.timestamp:
        return False                          # new stream, more than a minute stale

    mav.signing.stream_timestamps[stream_key] = timestamp
    mav.signing.timestamp = max(mav.signing.timestamp, timestamp)
    return True


def harden_replay_window(link) -> None:
    """Install `_strict_check_signature` on this link's codec.

    Idempotent. Bound onto the instance rather than the class so that a test
    can hold one hardened and one stock codec at the same time and compare
    them, which is how the finding is pinned.
    """
    link.mav.check_signature = types.MethodType(_strict_check_signature, link.mav)


# -- enabling ---------------------------------------------------------------
def enable_link_signing(link, key: bytes, cfg: SigningConfig | None = None,
                        initial_timestamp: int | None = None,
                        log: RejectionLog | None = None) -> RejectionLog:
    """Turn on signing for OUR end of `link`. Does not touch the vehicle.

    Call `provision_vehicle` for the pair of operations; this half exists
    separately because the vehicle may already hold the key from a previous
    session (it is in FRAM), in which case re-sending `SETUP_SIGNING` is
    avoidable churn but our own codec still has to be told the key.
    """
    cfg = cfg or SigningConfig()
    log = log or RejectionLog()
    if len(key) != KEY_BYTES:
        raise ValueError(f"signing key must be {KEY_BYTES} bytes, got {len(key)}")

    link.setup_signing(
        key,
        sign_outgoing=cfg.sign_outgoing,
        allow_unsigned_callback=make_allow_unsigned_callback(cfg, log),
        initial_timestamp=initial_timestamp if initial_timestamp is not None
        else mavlink_timestamp(),
        link_id=cfg.link_id,
    )
    if cfg.harden_replay:
        harden_replay_window(link)
    return log


def provision_vehicle(link, key: bytes, cfg: SigningConfig | None = None,
                      repeats: int = 3, settle_s: float = 0.2,
                      armed: bool | None = None) -> RejectionLog:
    """Carry `key` to the vehicle, then enable signing on our side.

    Order matters and is not symmetric: the key goes out FIRST, and our side
    switches to it only afterwards. Reversing that signs the bootstrap message
    with a key the vehicle does not yet hold, and the frame carrying the key
    is dropped for being unverifiable — a deadlock that has to be broken by
    erasing the airframe's FRAM.

    `SETUP_SIGNING` is sent under whatever signing state the link has right
    now, which is exactly right in both cases. On a fresh link that is
    unsigned, because the vehicle has no key to check it against — the message
    IS the bootstrap, and it is why plan §3.3 insists provisioning happens
    over a trusted local channel and never over the air in the clear. On a
    rotation it is signed with the OLD key, which the vehicle still holds and
    can verify, so a re-key is authenticated end to end.

    Sent `repeats` times: there is no ACK for `SETUP_SIGNING`, and the link is
    UDP. A lost key looks exactly like a working link right up until the first
    signed frame is silently dropped.

    `armed` short-circuits the attempt when the caller already knows the
    vehicle's state — ArduPilot refuses the message while armed
    (`GCS_Signing.cpp:74`) and answers only with a STATUSTEXT, so a silent
    no-op is the default failure mode. Refuse it here where it is loud.
    """
    cfg = cfg or SigningConfig()
    if armed:
        raise RuntimeError(
            "refusing to provision a signing key while armed: ArduPilot "
            "discards SETUP_SIGNING when armed and only warns by STATUSTEXT, "
            "so this would look like success and leave the link unauthenticated")
    if len(key) != KEY_BYTES:
        raise ValueError(f"signing key must be {KEY_BYTES} bytes, got {len(key)}")

    stamp = mavlink_timestamp()
    for _ in range(repeats):
        link.mav.setup_signing_send(link.target_system, link.target_component,
                                    bytearray(key), stamp)
        time.sleep(settle_s)

    return enable_link_signing(link, key, cfg, initial_timestamp=stamp)


def provision_fleet(links, keystore: Keystore, cfg: SigningConfig | None = None,
                    armed=None, verbose: bool = True) -> dict:
    """One key per vehicle, generated if absent, pushed to each link.

    `links[i]` is vehicle i, matching every other index in Domain 4. Keys are
    identified in the log by fingerprint only — see `Keystore.fingerprint`.
    """
    cfg = cfg or SigningConfig()
    links = list(links)
    created = keystore.ensure(range(len(links)))
    logs: dict[int, RejectionLog] = {}

    for i, link in enumerate(links):
        is_armed = None if armed is None else bool(armed[i])
        logs[i] = provision_vehicle(link, keystore.key(i), cfg, armed=is_armed)
        if verbose:
            fresh = " (new)" if i in created else ""
            print(f"  vehicle {i}: signing key {keystore.fingerprint(i)}{fresh}, "
                  f"reject-unsigned={not cfg.allow_unsigned}")

    for warning in keystore.warnings:
        print(f"  WARNING: {warning}")

    return {"created": created, "logs": logs,
            "fingerprints": {i: keystore.fingerprint(i) for i in range(len(links))}}


def disable_signing(link, repeats: int = 3, settle_s: float = 0.2) -> None:
    """Clear the vehicle's key. The documented recovery path, not a shortcut.

    An all-zero key with a zero timestamp is what ArduPilot treats as "signing
    off" (`GCS_Signing.cpp:148-163`). Exists because provisioning a key you
    then lose leaves an airframe that will not talk to you, and the only other
    way back is erasing its FRAM.
    """
    zeros = bytearray(KEY_BYTES)
    for _ in range(repeats):
        link.mav.setup_signing_send(link.target_system, link.target_component,
                                    zeros, 0)
        time.sleep(settle_s)
    link.mav.signing.secret_key = None
    link.mav.signing.sign_outgoing = False
    link.mav.signing.allow_unsigned_callback = None


# -- verification ------------------------------------------------------------
def probe_unsigned_rejected(link, timeout_s: float = 3.0) -> dict:
    """Ask the vehicle to act on an UNSIGNED command, and hope it will not.

    This is the check that separates "signing is configured" from "signing
    does anything". It passes on the serial2 transport and FAILS on serial0 —
    see the first finding in the module docstring. Run it during provisioning
    and treat a failure as a transport bug, not a nuisance.

    Mechanism: send `MAV_CMD_REQUEST_MESSAGE` with our outgoing signature
    switched off, then wait for a `COMMAND_ACK`. An ack means the vehicle
    parsed, accepted and executed an unauthenticated command. Silence means it
    refused. The command only asks for a `SYSTEM_TIME` message, so a false
    negative costs one extra packet and nothing else.

    Must be called before `ExternalNavFanout` starts: it flips `sign_outgoing`
    on a codec that is not thread-safe, and the fanout thread transmits on the
    same link.
    """
    signing = link.mav.signing
    was_signing = signing.sign_outgoing
    # Drain first, or a COMMAND_ACK already in flight for someone else's
    # command is read as this probe's answer and the link is declared open.
    while link.recv_match(type="COMMAND_ACK", blocking=False) is not None:
        pass

    signing.sign_outgoing = False
    try:
        link.mav.command_long_send(
            link.target_system, link.target_component,
            MAV_CMD_REQUEST_MESSAGE, 0,
            float(MAVLINK_MSG_ID_SYSTEM_TIME), 0, 0, 0, 0, 0, 0)
    finally:
        signing.sign_outgoing = was_signing

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        ack = link.recv_match(type="COMMAND_ACK", blocking=True, timeout=0.25)
        if ack is not None and ack.command == MAV_CMD_REQUEST_MESSAGE:
            return {
                "rejected": False,
                "detail": (
                    "the vehicle ACKed an UNSIGNED command. This link reaches "
                    "MAVLINK_COMM_0, which ArduPilot accepts unsigned "
                    "unconditionally (GCS_Signing.cpp:116). Move the closed "
                    "loop off --serial0 before claiming an authenticated link."),
            }
    return {"rejected": True,
            "detail": "no ACK for an unsigned command within "
                      f"{timeout_s:.1f}s; the vehicle refused it"}


def signing_report(link, log: RejectionLog | None = None) -> dict:
    """Counters worth printing at the end of a run, and worth watching during.

    `bad_signature` rising on a link that is otherwise healthy is the signal
    §6.1 asks for: someone is putting frames on the wire that we will not act
    on. `goodsig` at zero after provisioning means the far end never got the
    key.
    """
    s = link.mav.signing
    out = {
        "enabled": s.secret_key is not None,
        "sign_outgoing": bool(s.sign_outgoing),
        "link_id": s.link_id,
        "goodsig": s.goodsig_count,
        "badsig": s.badsig_count,
        "unsigned_accepted": s.unsigned_count,
        "unsigned_rejected": s.reject_count,
        "streams": len(s.stream_timestamps),
        "replay_hardened": isinstance(getattr(link.mav, "check_signature", None),
                                      types.MethodType)
        and link.mav.check_signature.__func__ is _strict_check_signature,
    }
    if log is not None:
        out["rejections"] = log.to_json()
    return out
