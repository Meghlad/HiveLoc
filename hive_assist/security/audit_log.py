"""H5.1 — a security event log that cannot be quietly edited.

Stage H4 of `COMMS_HARDENING_PLAN.md` §6.2. The plan's requirement is
"append-only, ideally signed/hash-chained, so a post-incident review can trust
the log wasn't edited", and the reason it matters here is narrower than the
usual audit-log argument: this system's whole thesis is that a *confidently
wrong* input is the dangerous one. When the plausibility gate refuses a
position or the range monitor drops a link, that rejection is the only evidence
an attack was attempted at all — everything downstream carries on working
exactly as if nothing happened. A log that an attacker can edit after the fact
turns that evidence into nothing.

---------------------------------------------------------------------------
WHY THE CHAIN IS KEYED, AND WHY AN UNKEYED ONE WOULD BE DECORATION
---------------------------------------------------------------------------

The obvious construction is `h_i = SHA256(h_{i-1} || record_i)`. It is also
useless on its own here. An attacker who can rewrite the log file can rewrite
*every* hash after the record they edited, and the chain verifies perfectly.
An unkeyed hash chain only detects tampering by someone who can modify the file
but cannot recompute it — which describes disk corruption, not an adversary.

So the chain is an HMAC keyed with a secret the log writer holds:

    mac_i = HMAC-SHA256(k_audit, mac_{i-1} || seq || unix_ms || kind || body)

Now forging the chain requires `k_audit`. The threat model this actually closes
is the realistic one for a companion computer: an attacker who gains file
access (or who recovers the airframe later) can still *destroy* the log, but
cannot rewrite history into something innocuous. Destruction is loud; silent
revision is not. That asymmetry is the entire point.

The key comes from the same `Keystore` as the signing keys, under a reserved
id (`AUDIT_KEY_ID`), so it inherits scrypt-at-rest and the no-default-passphrase
rule. It is deliberately NOT any vehicle's signing key: one key, one purpose,
so that compromising a recovered airframe's FRAM does not also hand over the
ability to forge the record of that compromise.

---------------------------------------------------------------------------
THE ATTACK THIS DOES NOT STOP, STATED PLAINLY
---------------------------------------------------------------------------

**Truncation.** An attacker who deletes the last N records leaves a prefix that
is perfectly valid — every MAC still checks, the chain just ends earlier. No
append-only construction can detect this from the file alone, because the file
alone cannot know how long it was supposed to be.

Two partial answers are implemented and neither is a cure:

  * `Verification.count` is returned, so a caller that independently knows how
    many events it wrote (or that shipped a checkpoint elsewhere) can compare.
  * `checkpoint()` writes a record carrying the current sequence number, so
    truncation past the last checkpoint is detectable, and truncation *between*
    checkpoints is bounded to that window.

The real fix is off-box replication — ship records to somewhere the attacker
does not control — and that is an operational deployment decision, not
something this file can do on its own. It is named here so nobody reads
"tamper-evident" as "tamper-proof".

---------------------------------------------------------------------------
DURABILITY
---------------------------------------------------------------------------

Every append is flushed and `fsync`ed before returning. That costs real time —
a few hundred microseconds — which is why `AuditLog` is never called from the
20 Hz sender thread or the estimator's inner loop. It is called from the
supervisory path, where per-event latency is irrelevant and losing the tail of
the log to a crash during an incident is not.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import pathlib
import threading
import time
from dataclasses import dataclass, field

from .keystore import KEY_BYTES, Keystore

# Reserved keystore slot for the audit key. Negative so it can never collide
# with a vehicle id, which are indices into the fleet and always >= 0.
AUDIT_KEY_ID = -1

# The chain's fixed starting value. Included so that an empty log and a log
# truncated to zero records are distinguishable: a valid log always has at
# least the genesis record, and a zero-length file therefore fails to open as
# a log that was ever written to.
GENESIS = b"hive-audit-v1-genesis"

MAC_BYTES = 16          # SHA-256 truncated. 128 bits is far past forgery reach.
FORMAT_VERSION = 1


def audit_key(keystore: Keystore) -> bytes:
    """Fetch (creating if absent) the audit chain key from the keystore.

    Uses the same encrypted-at-rest store as the signing keys rather than a
    second mechanism, because a second mechanism is a second thing to get
    wrong. See `AUDIT_KEY_ID` for why the id is negative.
    """
    keystore.ensure([AUDIT_KEY_ID])
    return keystore.key(AUDIT_KEY_ID)


def _canonical(obj) -> bytes:
    """Deterministic JSON. The MAC is over bytes, so key order must be fixed.

    `sort_keys` plus the compact separators means a record re-serialised on a
    different Python build produces the same bytes and therefore the same MAC.
    Without this, verification would depend on dict iteration order.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      default=str).encode("utf-8")


def _mac(key: bytes, prev: str, seq: int, unix_ms: int, kind: str,
         body: dict) -> str:
    m = hmac.new(key, digestmod=hashlib.sha256)
    m.update(prev.encode("ascii"))
    m.update(str(int(seq)).encode("ascii"))
    m.update(str(int(unix_ms)).encode("ascii"))
    m.update(kind.encode("utf-8"))
    m.update(_canonical(body))
    return m.hexdigest()[:MAC_BYTES * 2]


@dataclass
class Verification:
    """The result of walking a chain end to end."""

    ok: bool
    count: int
    first_bad_seq: int | None = None
    reason: str = ""
    kinds: dict = field(default_factory=dict)

    def summary(self) -> str:
        if self.ok:
            return f"chain intact, {self.count} records, kinds={self.kinds}"
        return (f"CHAIN BROKEN at seq {self.first_bad_seq}: {self.reason} "
                f"({self.count} records read)")


class AuditLog:
    """Append-only, HMAC-chained security event log.

    Thread-safe: the estimator loop, the range monitor and the signing layer
    all write to one log from different threads, and interleaved appends would
    otherwise corrupt the chain (two writers reading the same `prev`).
    """

    def __init__(self, path: pathlib.Path | str, key: bytes,
                 sync: bool = True) -> None:
        if len(key) != KEY_BYTES:
            raise ValueError(f"audit key must be {KEY_BYTES} bytes, "
                             f"got {len(key)}")
        self.path = pathlib.Path(path).expanduser()
        self._key = bytes(key)
        self._sync = sync
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

        self._seq, self._prev = self._resume()

    # -- construction -------------------------------------------------------
    @classmethod
    def from_keystore(cls, keystore: Keystore,
                      path: pathlib.Path | str | None = None,
                      sync: bool = True) -> "AuditLog":
        default = pathlib.Path(os.environ.get(
            "HIVE_AUDIT_LOG", "~/.hive/logs/security.jsonl"))
        return cls(path if path is not None else default,
                   audit_key(keystore), sync=sync)

    def _resume(self) -> tuple[int, str]:
        """Pick up an existing chain, or start one.

        Deliberately does NOT verify the whole file on open. Verification is an
        explicit operation (`verify()`), because a log that refuses to open
        because of an old broken record is a log that stops recording the
        incident currently in progress. Availability of the *recorder* beats
        integrity of its history at open time; the history is checked when it
        is read, which is when the answer actually matters.
        """
        if not self.path.exists() or self.path.stat().st_size == 0:
            genesis_mac = _mac(self._key, GENESIS.decode("ascii"), 0, 0,
                               "genesis", {"v": FORMAT_VERSION})
            with open(self.path, "w", encoding="utf-8") as fh:
                fh.write(json.dumps({
                    "seq": 0, "unix_ms": 0, "kind": "genesis",
                    "body": {"v": FORMAT_VERSION},
                    "prev": GENESIS.decode("ascii"), "mac": genesis_mac,
                }, sort_keys=True, separators=(",", ":")) + "\n")
            try:
                os.chmod(self.path, 0o600)
            except OSError:
                pass
            return 0, genesis_mac

        last = None
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    last = line
        if last is None:
            raise ValueError(f"{self.path} has content but no records")
        rec = json.loads(last)
        return int(rec["seq"]), str(rec["mac"])

    # -- writing ------------------------------------------------------------
    def append(self, kind: str, /, **body) -> dict:
        """Record one event. Returns the record as written.

        `kind` is a short slug the monitor groups on — `unsigned_frame`,
        `bad_signature`, `estimate_rejected`, `range_rejected`, `key_rotated`.
        Everything else goes in `body` and is opaque to this module.

        `kind` is positional-ONLY (note the `/`). Event bodies routinely carry
        their own `kind` field — a plausibility violation's kind, a range
        violation's kind — and without the marker `append("x", kind="teleport")`
        raises `got multiple values for argument 'kind'`. A logging call that
        throws is a dropped security event, and it would throw only for the
        events most worth keeping.
        """
        with self._lock:
            seq = self._seq + 1
            unix_ms = int(time.time() * 1000)
            mac = _mac(self._key, self._prev, seq, unix_ms, kind, body)
            rec = {"seq": seq, "unix_ms": unix_ms, "kind": str(kind),
                   "body": body, "prev": self._prev, "mac": mac}
            line = json.dumps(rec, sort_keys=True, separators=(",", ":"),
                              default=str)
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
                if self._sync:
                    fh.flush()
                    os.fsync(fh.fileno())
            self._seq, self._prev = seq, mac
            return rec

    def checkpoint(self, note: str = "") -> dict:
        """Record the current sequence number, bounding truncation.

        See the module docstring: truncation past a checkpoint is detectable
        because the checkpoint asserts a length. Call it at run start and run
        end at minimum.
        """
        return self.append("checkpoint", at_seq=self._seq, note=note)

    @property
    def seq(self) -> int:
        return self._seq

    # -- reading ------------------------------------------------------------
    def records(self) -> list[dict]:
        out = []
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out

    def verify(self, expect_seq: int | None = None) -> Verification:
        """Walk the chain. Every MAC must check and every link must match.

        `expect_seq` is the caller's independent belief about the last sequence
        number — from a checkpoint shipped off-box, or from its own count of
        writes. Supplying it is the only way truncation of the tail is caught;
        without it a truncated log verifies clean, which is the honest limit
        described in the module docstring.
        """
        prev = GENESIS.decode("ascii")
        count = 0
        kinds: dict = {}
        last_seq = -1

        try:
            recs = self.records()
        except (OSError, json.JSONDecodeError) as exc:
            return Verification(False, 0, None, f"unreadable: {exc}")

        if not recs:
            return Verification(False, 0, None,
                                "empty log: a valid chain always has genesis")

        for rec in recs:
            try:
                seq = int(rec["seq"])
                unix_ms = int(rec["unix_ms"])
                kind = str(rec["kind"])
                body = rec["body"]
                claimed_prev = str(rec["prev"])
                claimed_mac = str(rec["mac"])
            except (KeyError, TypeError, ValueError) as exc:
                return Verification(False, count, last_seq + 1,
                                    f"malformed record: {exc}", kinds)

            if seq != last_seq + 1:
                return Verification(False, count, seq,
                                    f"sequence jump: expected {last_seq + 1}",
                                    kinds)
            if claimed_prev != prev:
                return Verification(False, count, seq,
                                    "prev does not match the previous mac "
                                    "(a record was inserted, removed or "
                                    "reordered)", kinds)

            expect = _mac(self._key, prev, seq, unix_ms, kind, body)
            # compare_digest: the MAC is the thing under attack, so a timing
            # oracle on its comparison is a real (if slow) forgery path.
            if not hmac.compare_digest(expect, claimed_mac):
                return Verification(False, count, seq,
                                    "MAC mismatch (record was edited, or the "
                                    "wrong audit key was supplied)", kinds)

            prev = claimed_mac
            last_seq = seq
            count += 1
            kinds[kind] = kinds.get(kind, 0) + 1

        if expect_seq is not None and last_seq != int(expect_seq):
            return Verification(
                False, count, last_seq,
                f"chain is internally valid but ends at seq {last_seq}, "
                f"caller expected {int(expect_seq)} — the tail was truncated",
                kinds)

        return Verification(True, count, None, "", kinds)


class NullAuditLog:
    """A log-shaped object that records nothing.

    So that every call site can write `self.audit.append(...)` unconditionally
    instead of guarding each one with `if self.audit is not None`. A forgotten
    guard is a dropped security event, and this makes the guard impossible to
    forget.
    """

    seq = 0

    def append(self, kind: str, /, **body) -> dict:   # noqa: D102
        return {"seq": 0, "kind": kind, "body": body}

    def checkpoint(self, note: str = "") -> dict:     # noqa: D102
        return {"seq": 0, "kind": "checkpoint", "body": {"note": note}}

    def verify(self, expect_seq: int | None = None) -> Verification:
        return Verification(True, 0, None, "null log records nothing")
