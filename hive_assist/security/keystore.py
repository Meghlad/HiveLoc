"""H2.1 — one signing key per vehicle, generated here, encrypted at rest.

The plan's §3.1 rule is the whole reason this file exists: **never a shared
fleet secret**. A single key across four airframes means recovering one crashed
vehicle's FRAM hands the attacker command authority over the other three, and
"we rotated the fleet key" is then an outage, not a mitigation. One key per
vehicle makes a compromise bounded and a rotation local.

WHAT A MAVLINK SIGNING KEY IS. Exactly 32 bytes, used as the SHA-256 prefix in
the signature (`MAVLink_message.sign_packet`, and ArduPilot's
`SigningKey.secret_key[32]` in `GCS_Signing.cpp:36`). It is not a password and
must never be derived from one — it comes from `secrets.token_bytes`, i.e. the
OS CSPRNG, and nothing in this module ever prints it.

ENCRYPTION AT REST, AND ITS HONEST LIMITS (plan §3.2). The plan's bar for the
M1 dev box is "at minimum an encrypted file with a passphrase-derived key"; a
secure element is the target on real companion hardware and this file does not
pretend to be one. What is actually implemented:

    master        = scrypt(passphrase, salt, N, r, p, dklen=64)
    k_enc, k_mac  = master[:32], master[32:]
    keystream_i   = HMAC-SHA256(k_enc, nonce || be32(i))
    ciphertext    = plaintext XOR keystream
    tag           = HMAC-SHA256(k_mac, header || nonce || ciphertext)

Encrypt-then-MAC over the header as well as the body, so the KDF parameters
cannot be downgraded (an attacker rewriting N to 2 would otherwise get a
cheap-to-brute-force file that still opened). Verified with `compare_digest`.
No primitive here is novel — it is counter-mode key derivation plus EtM, both
standard — and it uses only the standard library, which is deliberate: this
repo has no crypto dependency and adding one to hold a key is a supply-chain
trade the plan's Tier-3 section explicitly declines to make. If `cryptography`
becomes acceptable, AES-GCM is a strictly better body cipher and the `"v"`
field exists so a v2 format can be added without ambiguity.

WHAT THIS DOES NOT PROTECT AGAINST. An attacker who is already running as you
on the companion computer, with the keystore open or the passphrase in the
environment. That is the secure-element gap, stated rather than papered over.

NO DEFAULT PASSPHRASE. `open()` raises rather than falling back to a constant.
A keystore with a known passphrase is a plaintext file wearing a hat, and it
would pass every test in `tests/test_reject_unsigned.py` while protecting
nothing.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import pathlib
import secrets
import stat
import struct
import time
from dataclasses import dataclass

# MAVLink 2 signing keys are exactly this long. Not a tunable: the field is
# uint8_t[32] on the wire and uint8_t secret_key[32] in ArduPilot's FRAM
# struct, and a short key would be silently zero-padded on one side only.
KEY_BYTES = 32

FORMAT_VERSION = 1

# scrypt work factors. N=2**15 with r=8 costs ~32 MB and a few hundred ms on
# the M1 — the standard interactive-login setting. Tests override it downward
# (see `KdfParams.weak_for_tests`), which is the only legitimate reason to.
SCRYPT_N = 2 ** 15
SCRYPT_R = 8
SCRYPT_P = 1
# hashlib.scrypt refuses to allocate past maxmem, and the OpenSSL default is
# right at the 32 MB this costs. Ask for headroom or the default parameters
# fail on some builds with a bare "memory limit exceeded".
SCRYPT_MAXMEM = 96 * 1024 * 1024

KEYSTORE_PATH = pathlib.Path(
    os.environ.get("HIVE_KEYSTORE", "~/.hive/keys/keystore.json")
).expanduser()

PASSPHRASE_ENV = "HIVE_KEYSTORE_PASSPHRASE"


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def _unb64(text: str) -> bytes:
    return base64.b64decode(text.encode("ascii"))


@dataclass(frozen=True)
class KdfParams:
    """scrypt cost parameters, stored in the clear and covered by the tag."""

    n: int = SCRYPT_N
    r: int = SCRYPT_R
    p: int = SCRYPT_P
    salt: bytes = b""

    @classmethod
    def fresh(cls) -> "KdfParams":
        return cls(salt=secrets.token_bytes(16))

    @classmethod
    def weak_for_tests(cls) -> "KdfParams":
        """N=2**10. Fast enough to run in a test, useless in the field.

        Exposed as a named constructor rather than a bare integer so that a
        weak keystore can never be created by accident — it has to be asked
        for by a name that says what it is.
        """
        return cls(n=2 ** 10, salt=secrets.token_bytes(16))

    def to_json(self) -> dict:
        return {"name": "scrypt", "n": self.n, "r": self.r, "p": self.p,
                "salt": _b64(self.salt)}

    @classmethod
    def from_json(cls, d: dict) -> "KdfParams":
        if d.get("name") != "scrypt":
            raise ValueError(f"unsupported kdf {d.get('name')!r}")
        return cls(n=int(d["n"]), r=int(d["r"]), p=int(d["p"]),
                   salt=_unb64(d["salt"]))

    def derive(self, passphrase: bytes) -> tuple[bytes, bytes]:
        """passphrase -> (k_enc, k_mac). Separate keys, never the same one."""
        master = hashlib.scrypt(passphrase, salt=self.salt, n=self.n,
                                r=self.r, p=self.p, dklen=64,
                                maxmem=SCRYPT_MAXMEM)
        return master[:32], master[32:]


def _keystream(k_enc: bytes, nonce: bytes, length: int) -> bytes:
    """HMAC-SHA256 in counter mode. One block per 32 bytes of plaintext."""
    out = bytearray()
    counter = 0
    while len(out) < length:
        out += hmac.new(k_enc, nonce + struct.pack(">I", counter),
                        hashlib.sha256).digest()
        counter += 1
    return bytes(out[:length])


def _xor(data: bytes, pad: bytes) -> bytes:
    return bytes(a ^ b for a, b in zip(data, pad))


def _mac_input(header: dict, nonce: bytes, ct: bytes) -> bytes:
    """Everything the tag commits to: version, KDF params, nonce, body.

    `sort_keys` + no whitespace makes this byte-identical on write and on
    read; a MAC over a re-serialised dict that formats differently verifies
    on the machine that wrote it and nowhere else.
    """
    canon = json.dumps(header, sort_keys=True, separators=(",", ":"))
    return canon.encode("utf-8") + nonce + ct


def _resolve_passphrase(passphrase: str | bytes | None) -> bytes:
    if passphrase is None:
        passphrase = os.environ.get(PASSPHRASE_ENV)
    if not passphrase:
        raise ValueError(
            f"no keystore passphrase. Pass one explicitly or set "
            f"${PASSPHRASE_ENV}. There is deliberately no default: a keystore "
            f"with a known passphrase protects nothing.")
    if isinstance(passphrase, str):
        passphrase = passphrase.encode("utf-8")
    return passphrase


class Keystore:
    """Per-vehicle 32-byte signing keys, held encrypted on disk.

    Open it, ask for a key, hand that key to `enable_signing`. The keys live
    in memory only for the life of the process and are never logged — use
    `fingerprint()` when a run needs to show that both ends agree.
    """

    def __init__(self, path: pathlib.Path, passphrase: bytes,
                 kdf: KdfParams, keys: dict[int, bytes],
                 created_unix: float) -> None:
        self.path = pathlib.Path(path)
        self._passphrase = passphrase
        self._kdf = kdf
        self._keys = dict(keys)
        self.created_unix = created_unix
        # Non-fatal things worth surfacing at the end of a provisioning run:
        # currently just "this file was readable by someone else".
        self.warnings: list[str] = []

    # -- lifecycle ---------------------------------------------------------
    @classmethod
    def create(cls, path: pathlib.Path | str | None = None,
               passphrase: str | bytes | None = None,
               kdf: KdfParams | None = None) -> "Keystore":
        """A new, empty keystore. Refuses to clobber an existing file.

        Overwriting is how a fleet loses its keys: the vehicles keep the old
        key in FRAM, the ground side generates new ones, and every link goes
        dark at once with a bad-signature count as the only clue.
        """
        path = pathlib.Path(path or KEYSTORE_PATH).expanduser()
        if path.exists():
            raise FileExistsError(
                f"{path} already exists; open() it or rotate(), do not create "
                f"over it — the vehicles still hold the keys it contains")
        ks = cls(path, _resolve_passphrase(passphrase),
                 kdf or KdfParams.fresh(), {}, time.time())
        ks.save()
        return ks

    @classmethod
    def open(cls, path: pathlib.Path | str | None = None,
             passphrase: str | bytes | None = None) -> "Keystore":
        path = pathlib.Path(path or KEYSTORE_PATH).expanduser()
        doc = json.loads(path.read_text())
        if int(doc.get("v", 0)) != FORMAT_VERSION:
            raise ValueError(f"keystore format v{doc.get('v')} not supported")

        kdf = KdfParams.from_json(doc["kdf"])
        k_enc, k_mac = kdf.derive(_resolve_passphrase(passphrase))
        nonce, ct, tag = _unb64(doc["nonce"]), _unb64(doc["ct"]), _unb64(doc["tag"])

        header = {"v": doc["v"], "kdf": doc["kdf"]}
        expected = hmac.new(k_mac, _mac_input(header, nonce, ct),
                            hashlib.sha256).digest()
        # compare_digest, not ==: a byte-at-a-time comparison leaks where the
        # tag first differs, which is enough to forge one given retries.
        if not hmac.compare_digest(expected, tag):
            raise ValueError(
                f"{path}: wrong passphrase, or the file has been tampered with. "
                f"These are indistinguishable by design and both are refused.")

        body = json.loads(_xor(ct, _keystream(k_enc, nonce, len(ct))))
        keys = {int(k): _unb64(v) for k, v in body["keys"].items()}
        for vid, key in keys.items():
            if len(key) != KEY_BYTES:
                raise ValueError(
                    f"vehicle {vid}: key is {len(key)} bytes, must be {KEY_BYTES}")

        ks = cls(path, _resolve_passphrase(passphrase), kdf, keys,
                 float(body.get("created_unix", 0.0)))
        ks._check_permissions()
        return ks

    @classmethod
    def open_or_create(cls, path: pathlib.Path | str | None = None,
                       passphrase: str | bytes | None = None,
                       kdf: KdfParams | None = None) -> "Keystore":
        path = pathlib.Path(path or KEYSTORE_PATH).expanduser()
        if path.exists():
            return cls.open(path, passphrase)
        return cls.create(path, passphrase, kdf)

    def _check_permissions(self) -> None:
        """0600 or it is not really at rest. Self-heals and says so."""
        mode = stat.S_IMODE(self.path.stat().st_mode)
        if mode & 0o077:
            self.path.chmod(0o600)
            self.warnings.append(
                f"{self.path} was mode {mode:04o} (readable beyond the owner); "
                f"reset to 0600. Assume the keys were exposed and rotate.")

    def save(self) -> None:
        body = json.dumps({
            "created_unix": self.created_unix,
            "keys": {str(vid): _b64(key) for vid, key in sorted(self._keys.items())},
        }).encode("utf-8")

        # Fresh nonce every write. The keystream is a function of (k_enc,
        # nonce), so reusing a nonce across two saves under the same
        # passphrase XORs two plaintexts together and leaks both.
        nonce = secrets.token_bytes(16)
        k_enc, k_mac = self._kdf.derive(self._passphrase)
        ct = _xor(body, _keystream(k_enc, nonce, len(body)))
        header = {"v": FORMAT_VERSION, "kdf": self._kdf.to_json()}
        tag = hmac.new(k_mac, _mac_input(header, nonce, ct),
                       hashlib.sha256).digest()

        doc = dict(header)
        doc.update({"nonce": _b64(nonce), "ct": _b64(ct), "tag": _b64(tag)})

        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        # Same tmp + os.replace discipline as hive/supervisor_io.py: a torn
        # keystore is unrecoverable, and os.replace makes it impossible rather
        # than unlikely. The temp file is created 0600 before it holds
        # anything, so the ciphertext is never briefly world-readable.
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(fd, json.dumps(doc, indent=2).encode("utf-8"))
        finally:
            os.close(fd)
        os.replace(tmp, self.path)

    # -- keys --------------------------------------------------------------
    def vehicle_ids(self) -> list[int]:
        return sorted(self._keys)

    def key(self, vehicle_id: int) -> bytes:
        try:
            return self._keys[int(vehicle_id)]
        except KeyError:
            raise KeyError(
                f"no signing key for vehicle {vehicle_id}; call ensure() "
                f"during provisioning") from None

    def ensure(self, vehicle_ids) -> list[int]:
        """Generate a key for any vehicle that lacks one. Returns the new ids.

        Idempotent on purpose — provisioning gets re-run, and regenerating a
        key that a vehicle already holds in FRAM would take that vehicle off
        the air until it is re-provisioned.
        """
        created = []
        for vid in vehicle_ids:
            vid = int(vid)
            if vid not in self._keys:
                self._keys[vid] = secrets.token_bytes(KEY_BYTES)
                created.append(vid)
        if created:
            self.save()
        return created

    def rotate(self, vehicle_id: int) -> bytes:
        """Replace one vehicle's key and return it, for re-provisioning.

        The new key is not live until `SETUP_SIGNING` has carried it to the
        vehicle. Between this call and that message the ground side and the
        airframe disagree, so rotate while disarmed — ArduPilot refuses
        `SETUP_SIGNING` when armed anyway (`GCS_Signing.cpp:74`).

        Prefer `security.rotate_keys.rotate_vehicle`, which does this as a
        two-phase commit (see `commit`). This method writes the new key to disk
        *before* the vehicle is known to hold it, so a failure between the two
        leaves the keystore describing a key the airframe never received.
        """
        vid = int(vehicle_id)
        self._keys[vid] = secrets.token_bytes(KEY_BYTES)
        self.save()
        return self._keys[vid]

    def commit(self, vehicle_id: int, key: bytes) -> None:
        """Store a specific key. The second phase of a verified rotation.

        Exists so that `rotate_keys` can generate a candidate, carry it to the
        vehicle, prove the vehicle answers on it, and only then write it down.
        The ordering matters more than it looks: a keystore that records a key
        the airframe does not hold is indistinguishable, on the next run, from
        an airframe that has been swapped or tampered with — you get a link
        that will not talk and no way to tell why. Committing last means the
        file on disk always describes a key that was observed working.

        Not a general-purpose setter. `ensure()` is how keys are created.
        """
        if len(key) != KEY_BYTES:
            raise ValueError(f"key must be {KEY_BYTES} bytes, got {len(key)}")
        self._keys[int(vehicle_id)] = bytes(key)
        self.save()

    def fingerprint(self, vehicle_id: int) -> str:
        """A short, safe-to-log identifier for a key.

        SHA-256 truncated to 8 hex characters. Enough to confirm both ends
        loaded the same key in a run log, nowhere near enough to recover it.
        Log this. Never log `key()`.
        """
        return hashlib.sha256(self.key(vehicle_id)).hexdigest()[:8]

    def report(self) -> dict:
        return {
            "path": str(self.path),
            "vehicles": self.vehicle_ids(),
            "fingerprints": {vid: self.fingerprint(vid) for vid in self.vehicle_ids()},
            "kdf_n": self._kdf.n,
            "warnings": list(self.warnings),
        }
