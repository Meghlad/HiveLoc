"""H2.2 — rotation must never write down a key the vehicle does not hold.

Every test here is about the *ordering*, because that is the only thing that
distinguishes a safe rotation from one that bricks an airframe. The happy path
is the least interesting case.

The fake link models the two behaviours that matter: ArduPilot only answers a
command whose signature it can verify, and `SETUP_SIGNING` is accepted only
while disarmed.
"""

from __future__ import annotations

import pytest

from security.audit_log import AuditLog
from security.keystore import KEY_BYTES, KdfParams, Keystore
from security.rotate_keys import (RotationPolicy, rotate_fleet, rotate_vehicle,
                                  verify_signed_link)


class Ack:
    def __init__(self, command):
        self.command = command


class FakeMav:
    def __init__(self, link):
        self._link = link
        self.signing = type("S", (), {"secret_key": None,
                                      "sign_outgoing": False,
                                      "link_id": 0,
                                      "timestamp": 0,
                                      "stream_timestamps": {},
                                      "goodsig_count": 0, "badsig_count": 0,
                                      "unsigned_count": 0, "reject_count": 0})()

    def setup_signing_send(self, sysid, compid, key, stamp):
        self._link.setup_signing_calls.append(bytes(key))
        if self._link.armed or self._link.deaf:
            return
        # The vehicle verifies the carrier frame under the key it currently
        # holds. A push signed with an unknown key is discarded — which is
        # exactly why the old key must carry the new one.
        if (self._link.vehicle_key is None
                or self.signing.secret_key == self._link.vehicle_key
                or self.signing.secret_key is None):
            self._link.vehicle_key = bytes(key)

    def command_long_send(self, *a, **k):
        self._link.commands += 1
        if self._link.deaf:
            return
        if self.signing.secret_key == self._link.vehicle_key:
            self._link.ack_queue.append(a[2])       # the command id


class FakeLink:
    """A vehicle that answers only what it can authenticate."""

    target_system = 1
    target_component = 1

    def __init__(self, key: bytes | None = None):
        self.vehicle_key = key
        self.armed = False
        self.deaf = False
        self.ack_queue: list[int] = []
        self.setup_signing_calls: list[bytes] = []
        self.commands = 0
        self.mav = FakeMav(self)

    def setup_signing(self, key, sign_outgoing=True,
                      allow_unsigned_callback=None, initial_timestamp=None,
                      link_id=None):
        self.mav.signing.secret_key = bytes(key)
        self.mav.signing.sign_outgoing = sign_outgoing

    def recv_match(self, type=None, blocking=False, timeout=None):
        # `type` shadows the builtin here because pymavlink's signature uses
        # that name; construct the reply from a real class rather than type().
        if self.ack_queue:
            return Ack(self.ack_queue.pop(0))
        return None


@pytest.fixture()
def ks(tmp_path):
    store = Keystore.create(tmp_path / "ks.json", passphrase=b"pw",
                            kdf=KdfParams.weak_for_tests())
    store.ensure([0, 1])
    return store


# --------------------------------------------------------------------------
def test_happy_path_rotates_and_commits(ks):
    old = ks.key(0)
    link = FakeLink(key=old)
    link.setup_signing(old)

    res = rotate_vehicle(link, ks, 0, timeout_s=0.2)

    assert res.ok
    assert ks.key(0) != old
    assert ks.key(0) == link.vehicle_key       # both ends agree
    assert len(ks.key(0)) == KEY_BYTES


def test_the_new_key_is_carried_by_the_old_one(ks):
    """The re-key is authenticated end to end.

    An attacker who does not hold the current key cannot rotate us onto one
    they choose, because the vehicle discards a SETUP_SIGNING it cannot verify.
    """
    old = ks.key(0)
    link = FakeLink(key=old)
    link.setup_signing(old)
    rotate_vehicle(link, ks, 0, timeout_s=0.2)

    # The push happened while our codec still held the old key.
    assert link.setup_signing_calls
    assert link.setup_signing_calls[0] != old      # it carried the candidate


def test_an_attacker_without_the_current_key_cannot_rotate_us(ks):
    """Same mechanism from the other side."""
    real = ks.key(0)
    link = FakeLink(key=real)
    link.setup_signing(b"\xAA" * KEY_BYTES)        # attacker's key, not ours
    link.mav.setup_signing_send(1, 1, bytearray(b"\xBB" * KEY_BYTES), 0)
    assert link.vehicle_key == real, "vehicle accepted an unauthenticated re-key"


def test_armed_vehicle_is_refused_loudly(ks):
    old = ks.key(0)
    link = FakeLink(key=old)
    link.setup_signing(old)

    res = rotate_vehicle(link, ks, 0, armed=True, timeout_s=0.2)

    assert not res.ok
    assert "armed" in res.reason
    assert ks.key(0) == old, "keystore changed on a rotation that never happened"
    assert not link.setup_signing_calls, "sent a message ArduPilot would discard"


def test_failed_rotation_rolls_back_and_writes_nothing(ks):
    """The case the two-phase commit exists for.

    The vehicle never receives the candidate. The keystore must still describe
    the key that actually works, or the next run is an unexplainable dead link.
    """
    old = ks.key(0)
    link = FakeLink(key=old)
    link.setup_signing(old)
    # Armed on the vehicle, but the caller does not know it — so the explicit
    # `armed` guard is bypassed and the push is silently discarded, which is
    # precisely ArduPilot's real failure mode.
    link.armed = True

    res = rotate_vehicle(link, ks, 0, armed=None, timeout_s=0.2)

    assert not res.ok
    assert res.rolled_back
    assert ks.key(0) == old, "committed a key the vehicle never acknowledged"
    assert link.mav.signing.secret_key == old, "left our codec on a dead key"


def test_link_dead_is_reported_and_still_writes_nothing(ks):
    old = ks.key(0)
    link = FakeLink(key=old)
    link.setup_signing(old)
    link.deaf = True                     # nothing answers, on any key

    res = rotate_vehicle(link, ks, 0, timeout_s=0.05)

    assert not res.ok and res.link_dead
    assert "FRAM" in res.reason          # the recovery path is stated
    assert ks.key(0) == old


def test_verify_signed_link_is_positive_proof(ks):
    key = ks.key(0)
    link = FakeLink(key=key)
    link.setup_signing(key)
    assert verify_signed_link(link, timeout_s=0.2)

    wrong = FakeLink(key=b"\x01" * KEY_BYTES)
    wrong.setup_signing(b"\x02" * KEY_BYTES)
    assert not verify_signed_link(wrong, timeout_s=0.05)


def test_rotation_is_recorded_in_the_audit_log(tmp_path, ks):
    log = AuditLog(tmp_path / "sec.jsonl", b"k" * 32, sync=False)
    old = ks.key(0)
    link = FakeLink(key=old)
    link.setup_signing(old)

    rotate_vehicle(link, ks, 0, timeout_s=0.2, audit=log)

    v = log.verify()
    assert v.ok
    assert v.kinds.get("key_rotation_started") == 1
    assert v.kinds.get("key_rotated") == 1
    # Fingerprints only — the log must never carry key material.
    body = str(log.records())
    assert old.hex() not in body and ks.key(0).hex() not in body


def test_fleet_rotation_stops_when_one_link_dies(ks):
    ks.ensure([0, 1, 2])
    links = []
    for i in range(3):
        lk = FakeLink(key=ks.key(i))
        lk.setup_signing(ks.key(i))
        links.append(lk)
    links[1].deaf = True

    out = rotate_fleet(links, ks, verbose=False)

    assert not out.ok
    assert out.dead_links == [1]
    # Sequential and interruptible: vehicle 2 was never touched, so a
    # systematic fault costs one airframe rather than the fleet.
    assert len(out.results) == 2
    assert links[2].setup_signing_calls == []


def test_policy_due():
    p = RotationPolicy(max_age_s=100.0)
    assert p.due(None)                       # never rotated
    assert p.due(0.0, now=200.0)
    assert not p.due(150.0, now=200.0)
