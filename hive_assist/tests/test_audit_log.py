"""H5.1 — the chain must catch edits, and must catch them for the right reason.

The test that matters is not "a valid chain verifies". It is that every way of
tampering with the file is *distinguishable*, because a log that says only
"broken" tells an incident reviewer nothing about what happened to it.
"""

from __future__ import annotations

import json

import pytest

from security.audit_log import (AUDIT_KEY_ID, AuditLog, NullAuditLog,
                                audit_key)
from security.keystore import KdfParams, Keystore


@pytest.fixture()
def log(tmp_path):
    return AuditLog(tmp_path / "security.jsonl", b"k" * 32, sync=False)


def test_fresh_log_starts_with_a_verifiable_genesis(log):
    v = log.verify()
    assert v.ok
    # Genesis exists so that truncation-to-empty is distinguishable from a log
    # that was simply never written to.
    assert v.count == 1
    assert v.kinds == {"genesis": 1}


def test_appends_chain_and_verify(log):
    for i in range(5):
        log.append("estimate_rejected", vehicle=i, kind="teleport")
    v = log.verify()
    assert v.ok
    assert v.count == 6                       # genesis + 5
    assert v.kinds["estimate_rejected"] == 5


def test_edited_body_breaks_the_mac(log, tmp_path):
    log.append("estimate_rejected", vehicle=0, kind="teleport", step_m=98.0)
    log.append("estimate_rejected", vehicle=1, kind="teleport", step_m=99.0)

    path = tmp_path / "security.jsonl"
    lines = path.read_text().strip().split("\n")
    rec = json.loads(lines[1])
    rec["body"]["step_m"] = 0.01              # "nothing happened here"
    lines[1] = json.dumps(rec, sort_keys=True, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n")

    v = log.verify()
    assert not v.ok
    assert v.first_bad_seq == 1
    assert "MAC mismatch" in v.reason


def test_deleted_middle_record_is_caught_as_a_chain_break(log, tmp_path):
    for i in range(4):
        log.append("range_rejected", vehicle=i)

    path = tmp_path / "security.jsonl"
    lines = path.read_text().strip().split("\n")
    del lines[2]                              # excise one event entirely
    path.write_text("\n".join(lines) + "\n")

    v = log.verify()
    assert not v.ok
    # Reported as a sequence jump rather than a MAC failure: the surviving
    # records are individually authentic, which is exactly the distinction an
    # incident review needs.
    assert "sequence jump" in v.reason


def test_reordered_records_are_caught(log, tmp_path):
    log.append("a", n=1)
    log.append("b", n=2)
    path = tmp_path / "security.jsonl"
    lines = path.read_text().strip().split("\n")
    lines[1], lines[2] = lines[2], lines[1]
    path.write_text("\n".join(lines) + "\n")
    assert not log.verify().ok


def test_truncation_is_invisible_without_an_expectation(log, tmp_path):
    """The honest limit, pinned so nobody claims more than it does."""
    for i in range(5):
        log.append("unsigned_frame", msg_id=i)
    path = tmp_path / "security.jsonl"
    lines = path.read_text().strip().split("\n")
    path.write_text("\n".join(lines[:3]) + "\n")

    # A truncated chain is internally perfect. This is the documented gap.
    assert log.verify().ok

    # ...and it is only closed by an independent belief about the length.
    v = log.verify(expect_seq=5)
    assert not v.ok
    assert "truncated" in v.reason


def test_wrong_key_cannot_verify(tmp_path):
    a = AuditLog(tmp_path / "log.jsonl", b"k" * 32, sync=False)
    a.append("estimate_rejected", vehicle=0)
    # An attacker who can read and rewrite the file but does not hold the audit
    # key: this is the whole reason the chain is keyed rather than a plain hash.
    b = AuditLog(tmp_path / "log.jsonl", b"j" * 32, sync=False)
    assert not b.verify().ok


def test_forged_record_without_the_key_fails(tmp_path):
    path = tmp_path / "log.jsonl"
    a = AuditLog(path, b"k" * 32, sync=False)
    a.append("estimate_rejected", vehicle=0)

    lines = path.read_text().strip().split("\n")
    last = json.loads(lines[-1])
    # Recompute a plausible-looking chain link with a guessed key. This is what
    # an unkeyed SHA-256 chain would have allowed outright.
    import hashlib
    forged = dict(last)
    forged["seq"] = last["seq"] + 1
    forged["kind"] = "checkpoint"
    forged["prev"] = last["mac"]
    forged["mac"] = hashlib.sha256(
        (last["mac"] + str(forged["seq"])).encode()).hexdigest()[:32]
    lines.append(json.dumps(forged, sort_keys=True, separators=(",", ":")))
    path.write_text("\n".join(lines) + "\n")

    assert not a.verify().ok


def test_resume_continues_an_existing_chain(tmp_path):
    p = tmp_path / "log.jsonl"
    a = AuditLog(p, b"k" * 32, sync=False)
    a.append("one", x=1)
    del a

    b = AuditLog(p, b"k" * 32, sync=False)
    b.append("two", x=2)
    v = b.verify()
    assert v.ok and v.count == 3
    assert b.seq == 2


def test_checkpoint_records_the_sequence(log):
    log.append("x")
    rec = log.checkpoint("run end")
    assert rec["body"]["at_seq"] == 1
    assert log.verify().ok


def test_audit_key_comes_from_the_keystore_and_is_not_a_vehicle_key(tmp_path):
    ks = Keystore.create(tmp_path / "ks.json", passphrase=b"pw",
                         kdf=KdfParams.weak_for_tests())
    ks.ensure([0, 1])
    k = audit_key(ks)
    assert len(k) == 32
    # One key, one purpose: recovering a vehicle's FRAM must not also hand over
    # the ability to rewrite the record of that recovery.
    assert k != ks.key(0) and k != ks.key(1)
    assert AUDIT_KEY_ID not in [0, 1]
    assert audit_key(ks) == k                 # stable across calls


def test_null_log_is_call_compatible():
    n = NullAuditLog()
    n.append("estimate_rejected", vehicle=0, kind="teleport")
    n.checkpoint("x")
    assert n.verify().ok


def test_rejects_a_short_key(tmp_path):
    with pytest.raises(ValueError):
        AuditLog(tmp_path / "l.jsonl", b"tooshort", sync=False)
