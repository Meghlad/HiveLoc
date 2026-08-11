"""H1.3 — the link refuses what it should refuse.

Four properties, all of which fail SILENTLY on a live link, which is why they
are pinned offline against the real pymavlink codec rather than left to a SITL
run to notice:

1. AN UNSIGNED COMMAND IS DROPPED. The whole point of stage H0. If this
   regresses, the loop is plaintext MAVLink over UDP again and anyone who can
   reach the port can fly the vehicle.

2. A FORGED SIGNATURE IS DROPPED. Signing with the wrong key must be no better
   than not signing at all.

3. A REPLAYED FRAME IS DROPPED. A captured, byte-identical, correctly-signed
   `SET_POSITION_TARGET_LOCAL_NED` must not fly the vehicle a second time. This
   is the property the 48-bit timestamp exists for and the one an attacker with
   an SDR gets for free if it breaks.

4. A FORGED FRAME CANNOT SHUT THE STREAM. `security/enable_signing.py`'s second
   finding: stock pymavlink writes the stream timestamp before it verifies the
   HMAC, so one junk frame with a far-future timestamp permanently blocks the
   legitimate stream. Both halves are asserted — the stock behaviour, so the
   bug stays visible, and the hardened behaviour, so the fix stays fixed.

No SITL, no sockets. `MAVLink` is instantiated over a `BytesIO`, which is
enough to sign and parse real frames, and the transmit side stands in for the
attacker as well as the operator — the difference is only which key it holds.
"""

from __future__ import annotations

import io
import time

import pytest

from security import enable_signing as es
from security.keystore import KEY_BYTES, KdfParams, Keystore

GOOD_KEY = bytes(range(32))
ATTACKER_KEY = b"\xa5" * 32


def _codec(key, *, sys_id=255, timestamp=None, allow_unsigned=None,
           harden=False, link_id=0):
    """A bare MAVLink codec with signing set up by hand.

    Deliberately does NOT go through `enable_link_signing`: that path needs a
    `mavutil` connection with a socket. The signing state it produces is the
    same handful of fields, set here directly, so the reject logic under test
    is the real one.
    """
    from pymavlink.dialects.v20 import ardupilotmega as mavlink2

    mav = mavlink2.MAVLink(io.BytesIO(), srcSystem=sys_id, srcComponent=1)
    mav.signing.secret_key = key
    mav.signing.sign_outgoing = key is not None
    mav.signing.link_id = link_id
    mav.signing.timestamp = (es.mavlink_timestamp() if timestamp is None
                             else timestamp)
    mav.signing.allow_unsigned_callback = allow_unsigned
    if harden:
        es.harden_replay_window(_FakeLink(mav))
    return mav


class _FakeLink:
    """The one attribute `harden_replay_window` touches."""

    def __init__(self, mav):
        self.mav = mav


def _setpoint(mav, north=8.0):
    """A frame that would actually move a vehicle if it were accepted."""
    return mav.set_position_target_local_ned_encode(
        0, 1, 1, 1, 0b0000111111111000,
        north, 0.0, -2.5, 0, 0, 0, 0, 0, 0, 0, 0).pack(mav)


def _parse(mav, frame):
    """Decode one frame, returning None if the signing layer refused it.

    `mavutil` sets `robust_parsing = True` on a live link, which converts the
    MAVError into a BAD_DATA message; here the exception is caught explicitly
    so the test asserts on the refusal rather than on the wrapper.

    MAVError comes from the same dialect module the codec was built from —
    every dialect defines its own class, so catching `common.MAVError` around
    an `ardupilotmega` codec catches nothing.
    """
    from pymavlink.dialects.v20.ardupilotmega import MAVError

    try:
        return mav.decode(bytearray(frame))
    except MAVError:
        return None


# -- 1. unsigned is dropped --------------------------------------------------
def test_unsigned_command_is_dropped():
    """No signature, no command. The floor the whole stage stands on."""
    log = es.RejectionLog()
    cfg = es.SigningConfig()
    rx = _codec(GOOD_KEY, sys_id=1,
                allow_unsigned=es.make_allow_unsigned_callback(cfg, log))

    attacker = _codec(None, sys_id=255)          # signs nothing
    assert _parse(rx, _setpoint(attacker)) is None
    assert log.unsigned == 1
    assert rx.signing.reject_count == 1


def test_correctly_signed_command_is_accepted():
    """The negative control. A reject-everything link is not a hardened one."""
    rx = _codec(GOOD_KEY, sys_id=1)
    tx = _codec(GOOD_KEY, sys_id=255)

    msg = _parse(rx, _setpoint(tx, north=8.0))
    assert msg is not None and msg.x == pytest.approx(8.0)
    assert rx.signing.goodsig_count == 1
    assert rx.signing.badsig_count == 0


def test_radio_status_is_the_only_thing_allowed_unsigned():
    """The allowlist mirrors ArduPilot's and must not grow by accident.

    A SiK radio injects RADIO_STATUS itself and holds no key, so refusing it
    costs the RSSI telemetry §6.1 wants for jamming detection. Every other
    message id is refused unsigned — including the two that matter most.
    """
    cfg = es.SigningConfig()
    log = es.RejectionLog()
    allow = es.make_allow_unsigned_callback(cfg, log)
    rx = _codec(GOOD_KEY, sys_id=1)

    assert allow(rx, es.MAVLINK_MSG_ID_RADIO_STATUS) is True
    assert allow(rx, es.MAVLINK_MSG_ID_RADIO) is True
    for msg_id in (75, 76, 84, 102, 11):        # COMMAND_INT/LONG, SETPOINT, VISION_POSITION_ESTIMATE, SET_MODE
        assert allow(rx, msg_id) is False, f"msg {msg_id} accepted unsigned"


def test_allowlist_ids_match_the_dialect():
    """Pin the two constants against pymavlink so a rename cannot drift them."""
    from pymavlink.dialects.v20 import ardupilotmega as mavlink2

    assert es.MAVLINK_MSG_ID_RADIO_STATUS == mavlink2.MAVLINK_MSG_ID_RADIO_STATUS
    assert es.MAVLINK_MSG_ID_RADIO == mavlink2.MAVLINK_MSG_ID_RADIO


# -- 2. a forged signature is dropped ---------------------------------------
def test_wrong_key_is_dropped():
    """An attacker who signs with a key they guessed gets nowhere."""
    rx = _codec(GOOD_KEY, sys_id=1)
    attacker = _codec(ATTACKER_KEY, sys_id=255)

    assert _parse(rx, _setpoint(attacker)) is None
    assert rx.signing.badsig_count == 1
    assert rx.signing.goodsig_count == 0


# -- 3. replay is dropped ----------------------------------------------------
def test_replayed_frame_is_dropped():
    """The identical bytes that worked once must not work twice.

    This is the capture-and-retransmit attack in §1.2's Tier 1: no key needed,
    no protocol knowledge needed, just an SDR and a recording.
    """
    rx = _codec(GOOD_KEY, sys_id=1)
    tx = _codec(GOOD_KEY, sys_id=255)
    frame = _setpoint(tx)

    assert _parse(rx, frame) is not None, "first transmission should be accepted"
    assert _parse(rx, frame) is None, "replay of the same frame was accepted"


def test_a_new_stream_more_than_a_minute_stale_is_dropped():
    """A recording played back later cannot open a fresh stream either.

    pymavlink accepts a previously-unseen (link_id, sysid, compid) whose
    timestamp is at most one minute behind ours. Beyond that the frame is a
    replay from an old session, and both implementations refuse it.
    """
    now = es.mavlink_timestamp()
    rx = _codec(GOOD_KEY, sys_id=1, timestamp=now)
    # 10 minutes stale, on a link id the receiver has never seen.
    stale = _codec(GOOD_KEY, sys_id=255, link_id=9,
                   timestamp=now - 600 * es.TIMESTAMP_UNITS_PER_SECOND)

    assert _parse(rx, _setpoint(stale)) is None


# -- 4. a forgery must not shut the stream ----------------------------------
def _far_future_forgery(link_id=0):
    """A frame with a valid-looking timestamp far ahead and a junk signature."""
    ahead = es.mavlink_timestamp() + 3600 * es.TIMESTAMP_UNITS_PER_SECOND
    attacker = _codec(ATTACKER_KEY, sys_id=255, timestamp=ahead, link_id=link_id)
    return _setpoint(attacker)


def test_stock_pymavlink_is_poisoned_by_one_forged_frame():
    """Documents the bug the hardening exists for — do not 'fix' this test.

    Stock `check_signature` records the stream timestamp BEFORE comparing the
    HMAC, so a single junk frame carrying a far-future timestamp raises the
    replay bar and every legitimate frame after it is refused as stale. One
    packet, no key, and the estimator's telemetry feed is dark.
    """
    rx = _codec(GOOD_KEY, sys_id=1, harden=False)
    tx = _codec(GOOD_KEY, sys_id=255)

    assert _parse(rx, _far_future_forgery()) is None      # forgery refused ...
    assert _parse(rx, _setpoint(tx)) is None              # ... and so is the real one


def test_hardened_check_survives_the_same_forgery():
    """Signature first, stream state only on success — the C implementation's
    order (`mavlink_helpers.h::mavlink_signature_check`)."""
    rx = _codec(GOOD_KEY, sys_id=1, harden=True)
    tx = _codec(GOOD_KEY, sys_id=255)

    assert _parse(rx, _far_future_forgery()) is None
    msg = _parse(rx, _setpoint(tx, north=8.0))
    assert msg is not None and msg.x == pytest.approx(8.0)


def test_hardening_still_rejects_a_genuine_replay():
    """The fix must not buy availability by giving up replay protection."""
    rx = _codec(GOOD_KEY, sys_id=1, harden=True)
    tx = _codec(GOOD_KEY, sys_id=255)
    frame = _setpoint(tx)

    assert _parse(rx, frame) is not None
    assert _parse(rx, frame) is None


# -- H2.1 — the keys themselves ---------------------------------------------
@pytest.fixture
def keystore(tmp_path):
    """A keystore with deliberately weak KDF parameters, so tests run fast."""
    return Keystore.create(tmp_path / "keystore.json", passphrase="test-pass",
                           kdf=KdfParams.weak_for_tests())


def test_keys_are_per_vehicle_and_distinct(keystore):
    """§3.1: one airframe's key must not be the fleet's key."""
    keystore.ensure(range(4))
    keys = [keystore.key(i) for i in range(4)]
    assert all(len(k) == KEY_BYTES for k in keys)
    assert len(set(keys)) == 4, "vehicles share a signing key"


def test_keystore_round_trips_through_disk(keystore, tmp_path):
    keystore.ensure(range(3))
    expected = {i: keystore.key(i) for i in range(3)}

    reopened = Keystore.open(tmp_path / "keystore.json", passphrase="test-pass")
    assert {i: reopened.key(i) for i in range(3)} == expected


def test_wrong_passphrase_is_refused(keystore, tmp_path):
    keystore.ensure([0])
    with pytest.raises(ValueError, match="tampered|passphrase"):
        Keystore.open(tmp_path / "keystore.json", passphrase="not-the-pass")


def test_tampered_keystore_is_refused(keystore, tmp_path):
    """Encrypt-then-MAC: a flipped ciphertext byte must not decrypt to
    something that merely looks wrong — it must not open at all."""
    import json

    keystore.ensure([0])
    path = tmp_path / "keystore.json"
    doc = json.loads(path.read_text())
    ct = bytearray(__import__("base64").b64decode(doc["ct"]))
    ct[0] ^= 0x01
    doc["ct"] = __import__("base64").b64encode(bytes(ct)).decode()
    path.write_text(json.dumps(doc))

    with pytest.raises(ValueError):
        Keystore.open(path, passphrase="test-pass")


def test_kdf_parameters_cannot_be_downgraded(keystore, tmp_path):
    """The tag covers the header, so an attacker cannot rewrite N down to 2
    and brute-force the passphrase against a file that still opens."""
    import json

    keystore.ensure([0])
    path = tmp_path / "keystore.json"
    doc = json.loads(path.read_text())
    doc["kdf"]["n"] = 2
    path.write_text(json.dumps(doc))

    with pytest.raises(ValueError):
        Keystore.open(path, passphrase="test-pass")


def test_ensure_is_idempotent(keystore):
    """Re-provisioning must not regenerate a key the vehicle already holds in
    FRAM — that takes the airframe off the air until it is re-provisioned."""
    keystore.ensure(range(2))
    before = {i: keystore.key(i) for i in range(2)}

    assert keystore.ensure(range(2)) == []
    assert {i: keystore.key(i) for i in range(2)} == before


def test_rotate_replaces_exactly_one_vehicles_key(keystore):
    keystore.ensure(range(3))
    others = {i: keystore.key(i) for i in (0, 2)}
    old = keystore.key(1)

    new = keystore.rotate(1)
    assert new != old and len(new) == KEY_BYTES
    assert {i: keystore.key(i) for i in (0, 2)} == others


def test_create_refuses_to_clobber_an_existing_keystore(keystore, tmp_path):
    with pytest.raises(FileExistsError):
        Keystore.create(tmp_path / "keystore.json", passphrase="test-pass",
                        kdf=KdfParams.weak_for_tests())


def test_no_passphrase_is_an_error_not_a_default(tmp_path, monkeypatch):
    """A keystore with a known passphrase would pass every other test here
    while protecting nothing."""
    monkeypatch.delenv("HIVE_KEYSTORE_PASSPHRASE", raising=False)
    with pytest.raises(ValueError, match="passphrase"):
        Keystore.create(tmp_path / "ks.json", kdf=KdfParams.weak_for_tests())


def test_keystore_is_written_owner_only(keystore, tmp_path):
    import stat as _stat

    mode = _stat.S_IMODE((tmp_path / "keystore.json").stat().st_mode)
    assert mode & 0o077 == 0, f"keystore is mode {mode:04o}, readable by others"


def test_fingerprint_identifies_without_revealing(keystore):
    """Run logs need to show both ends agree; they must never show the key."""
    keystore.ensure([0, 1])
    fp = keystore.fingerprint(0)

    assert len(fp) == 8
    assert fp != keystore.fingerprint(1)
    assert keystore.key(0).hex()[:8] != fp


# -- provisioning refuses the unsafe cases -----------------------------------
def test_provisioning_while_armed_is_refused():
    """ArduPilot discards SETUP_SIGNING when armed and answers only with a
    STATUSTEXT (GCS_Signing.cpp:74), so the silent failure mode is 'looks
    provisioned, is not'. Fail loudly on this side instead."""
    with pytest.raises(RuntimeError, match="armed"):
        es.provision_vehicle(object(), GOOD_KEY, armed=True)


def test_a_short_key_is_refused():
    """32 bytes exactly. A short key would be zero-padded on one side only."""
    with pytest.raises(ValueError, match="32 bytes"):
        es.provision_vehicle(object(), b"\x01" * 16, armed=False)


# -- the transport the whole stage depends on -------------------------------
def test_the_closed_loop_is_not_launched_on_serial0():
    """The security property lives in the launcher, so it is pinned here.

    ArduPilot accepts every unsigned frame on MAVLINK_COMM_0 unconditionally
    (`GCS_Signing.cpp:116`), and `--serial0` is that channel. Measured against
    live SITL: with a key provisioned and the loop on serial0, the vehicle
    still ACKed an unsigned command; moved to serial2, it refused it. Nothing
    else in this file would catch someone moving it back — every signing test
    would keep passing while the vehicle accepted anything.
    """
    import pathlib

    launcher = (pathlib.Path(__file__).resolve().parents[1]
                / "sim" / "run_fleet.sh").read_text()
    loop_line = [ln for ln in launcher.splitlines()
                 if "udpclient:127.0.0.1:${port}" in ln]

    assert loop_line, "could not find the loop's transport argument"
    for ln in loop_line:
        assert "--serial2" in ln, (
            f"the closed loop must not be launched on serial0/serial1: {ln.strip()}")

    # And the console must be passed explicitly, or SITL blocks in accept()
    # on its default `tcp:5760:wait` and the fleet never heartbeats.
    assert '--serial0 "tcp:0"' in launcher


def test_mavlink_timestamp_is_10us_units_since_2015():
    """The unit is not seconds and not microseconds; getting it wrong puts
    every frame outside the peer's replay window."""
    assert es.mavlink_timestamp(es.MAVLINK_EPOCH_UNIX) == 0
    assert es.mavlink_timestamp(es.MAVLINK_EPOCH_UNIX + 1) == 100_000
    # Clock before the epoch clamps rather than going negative.
    assert es.mavlink_timestamp(0) == 0
    assert es.mavlink_timestamp(time.time()) > 0
