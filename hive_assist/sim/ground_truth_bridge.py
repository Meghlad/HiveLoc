"""D4.1 — per-vehicle TRUE position out of SITL, and nothing else.

This module exists to answer one question correctly: where is vehicle `i`,
really? Everything downstream — the range simulator, and therefore the
estimator, and therefore the setpoints — is built on the answer, so getting it
from the wrong field does not produce a slightly worse demo. It produces a demo
that proves nothing.

THE CORRECTNESS TRAP THE PLAN NAMES (§4.2). Do not read `LOCAL_POSITION_NED`.
That message is the EKF's own fused output. Feeding it into the range simulator
would generate the measurements that the estimator then uses to correct the EKF
that produced them — a loop closed on itself, which converges beautifully and
means nothing. `GLOBAL_POSITION_INT` is the same output wearing geodetic
clothes. Both are listed in `FORBIDDEN_SOURCES` below and this module refuses to
be configured with either.

WHICH FIELD IS ACTUALLY GROUND TRUTH — verified against ArduPilot 4.5.7 source,
`libraries/SITL/SITL.cpp`, not assumed:

    SIMSTATE.lat, .lng            int32, degE7.  `state.latitude * 1.0e7`
                                  into an int32 field. Exact to 1e-7 deg
                                  (~1.1 cm). THIS IS THE SOURCE.

    SIM_STATE.lat_int, .lon_int   int32, degE7. Identical value, identical
                                  precision. Valid fallback.

    SIM_STATE.lat, .lon           float32 holding a degE7 *value* (the C is
                                  `state.latitude*1.0e7` into a float field).
                                  A latitude near -35.36 becomes -3.5363261e8,
                                  where float32's ulp is 32 — so the value is
                                  quantised to 32 degE7 units and the
                                  worst-case error is 16, about 18 cm of
                                  latitude. Sixteen times SIMSTATE's 1.1 cm and
                                  twelve times the UWB sigma the ranges carry,
                                  which would make ground truth the dominant
                                  error term in the whole loop. Reading these
                                  looks completely reasonable and degrades
                                  everything silently. `_decode` refuses them.

So the ladder is SIMSTATE first, SIM_STATE.lat_int second, and there is no
third. `source(i)` reports which one a given link actually delivered, which is
the runtime discharge of the plan's "verify the exact ground-truth field per
instance".

RESIDUAL QUANTISATION, STATED HONESTLY. degE7 is ~1.11 cm of latitude and
~0.9 cm of longitude at the CMAC default home. The UWB sigma the range world
uses is 1.5 cm, so ground truth is quantised at roughly 3/4 of a sigma. That is
small enough not to matter and too large to ignore in a write-up: the estimator
cannot be shown to beat ~1 cm here no matter how good it is, because the truth
it is scored against does not resolve below that.

TRANSPORT. SITL is launched with `--serial2 udpclient:127.0.0.1:<port>`, so the
autopilot pushes to a port this process binds (`udpin:`). SITL's socket is
connected to that port, so replies sent from the same bound socket are accepted
and the link is bidirectional on one hop — no MAVProxy, no router, no DDS. The
same connection carries VISION_POSITION_ESTIMATE out (D4.4) and
SET_POSITION_TARGET_LOCAL_NED out (D4.5), which is why this class hands out its
`link()` rather than hiding it.

SERIAL2, NOT SERIAL0, FOR A SECURITY REASON — see `sim/run_fleet.sh`. ArduPilot
accepts every unsigned frame on MAVLINK_COMM_0 unconditionally
(`GCS_Signing.cpp:116`), and serial0 is that channel. Because this one link
carries the estimate, the setpoints and the arm commands, it is also the single
place worth authenticating, which is what `connect(keystore=...)` does: every
emitter downstream — the fanout, the commanders, the mission upload — transmits
through `link(i)` and inherits signing from it without knowing anything about
it. There is no second socket to remember.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from hive.frames import TacFrame

# Reading either of these into the range simulator closes a loop on itself.
# Named rather than merely avoided, so the mistake has to be made on purpose.
FORBIDDEN_SOURCES = frozenset({"LOCAL_POSITION_NED", "GLOBAL_POSITION_INT",
                               "ODOMETRY", "VISION_POSITION_ESTIMATE"})

# Ordered best-first. `source()` reports whichever one actually arrived.
#
# A LIST, not a tuple, and that is not style. pymavlink's `recv_match` does
# `if not isinstance(type, list): type = [type]`, so a tuple is wrapped instead
# of unpacked and the filter becomes `msg.get_type() in [('SIMSTATE', ...)]`,
# which is never true. The symptom is a live link, a good heartbeat, SIMSTATE
# visibly arriving at 20 Hz on the wire, and a bridge that reports no ground
# truth at all. Measured, not hypothesised.
TRUTH_SOURCES = ["SIMSTATE", "SIM_STATE"]

# The vehicle's half of the mission upload handshake. Routed to a mailbox for
# the same reason STATUSTEXT is: one socket, one reader, and a `recv_match`
# filtering for MISSION_REQUEST would silently eat the truth stream.
MISSION_PROTOCOL = ["MISSION_REQUEST", "MISSION_REQUEST_INT", "MISSION_ACK"]

MAVLINK_MSG_ID_SIMSTATE = 164
MAVLINK_MSG_ID_SIM_STATE = 108

DEG_E7 = 1.0e-7

# ArduCopter SITL's default home (CMAC, Canberra). The surveyed anchor sits at
# this coordinate by default so TacFrame's origin and SITL's origin coincide and
# any residual offset in a plot is a real error rather than a frame mismatch.
SITL_HOME_LAT = -35.363261
SITL_HOME_LON = 149.165230
SITL_HOME_ALT = 584.0

# Vehicle i's MAVLink endpoint. The stride matches ArduPilot's own -I convention
# (and the Rust supervisor's --base-port/--port-stride defaults), so a port
# number here means the same thing everywhere in the stack.
BASE_PORT = 14551
PORT_STRIDE = 10


def endpoint_url(i: int, base_port: int = BASE_PORT,
                 stride: int = PORT_STRIDE) -> str:
    return f"udpin:127.0.0.1:{base_port + i * stride}"


@dataclass
class VehicleTruth:
    """The last ground-truth fix from one vehicle."""

    index: int
    lat_deg: float = float("nan")
    lon_deg: float = float("nan")
    yaw_rad: float = float("nan")
    # True altitude AMSL. Only SIM_STATE carries it — SIMSTATE has no altitude
    # field at all — so it is captured from SIM_STATE independently of which
    # message wins the lat/lon race. Domain 1 is planar and the estimator never
    # sees this; it exists so a flight can be shown to have actually left the
    # ground, which is otherwise invisible in a 2D error number.
    alt_amsl_m: float = float("nan")
    xy_tac: np.ndarray = field(default_factory=lambda: np.full(2, np.nan))
    source: str = ""
    stamp_mono_s: float = 0.0
    count: int = 0

    @property
    def valid(self) -> bool:
        return self.count > 0 and bool(np.all(np.isfinite(self.xy_tac)))

    def age_s(self, now: float | None = None) -> float:
        if self.count == 0:
            return float("inf")
        now = time.monotonic() if now is None else now
        return now - self.stamp_mono_s


class GroundTruthBridge:
    """N MAVLink links to N headless SITL instances; true positions out.

    Owns the connections. `pump()` is non-blocking and drains whatever has
    arrived; the loop calls it every tick and then reads `truths()`. Nothing
    here blocks on a message, because a loop that blocks on vehicle 3 stops
    feeding vehicles 0-2 and the external-nav stream must never gap (§4.3).
    """

    def __init__(self, frame: TacFrame, n: int, base_port: int = BASE_PORT,
                 stride: int = PORT_STRIDE, hz: float = 25.0,
                 dialect: str = "ardupilotmega") -> None:
        self.frame = frame
        self.n = int(n)
        self.base_port = base_port
        self.stride = stride
        self.hz = float(hz)
        self.dialect = dialect
        self.links: list = []
        self.truth = [VehicleTruth(i) for i in range(self.n)]

        # Mailboxes. Everything the single reader sees but does not own itself
        # lands here for `VehicleCommander` to pick up. Bounded deques: a
        # chatty autopilot must not grow a list forever in a loop that is meant
        # to run for hours.
        self.status: list = [deque(maxlen=200) for _ in range(self.n)]
        self.acks: list = [deque(maxlen=64) for _ in range(self.n)]
        # The mission protocol is a REQUEST/RESPONSE handshake, so its replies
        # have to come back through the same single reader as everything else.
        # `sim/qgc_markers.py` drains this while uploading the map markers.
        self.mission: list = [deque(maxlen=64) for _ in range(self.n)]
        self.armed = [False] * self.n
        self.mode: list[int | None] = [None] * self.n
        # Set by connect() when a keystore is supplied: per-vehicle key
        # fingerprints and the rejection logs the links accumulate. None means
        # the links are unauthenticated, which is a fact worth being able to
        # read off the object rather than infer from how it was constructed.
        self.signing: dict | None = None

    # -- lifecycle --------------------------------------------------------
    def connect(self, timeout_s: float = 60.0, verbose: bool = True,
                keystore=None, signing_cfg=None) -> None:
        """Bind every endpoint, wait for each heartbeat, optionally sign.

        Heartbeat first, then the stream request: `SET_MESSAGE_INTERVAL` is
        addressed to a system id, and until a heartbeat arrives pymavlink does
        not know what that id is. Requesting before knowing sends the command to
        system 0 and ArduPilot ignores it, which looks exactly like "SIMSTATE is
        not available on this build".

        SIGNING SITS BETWEEN THOSE TWO STEPS, and the order is the reason this
        loop is now split in three passes instead of one. `SETUP_SIGNING` is
        addressed to a system id as well, so it cannot go out before the
        heartbeat; and once it has gone out, everything this process transmits
        must be signed, so it has to happen before the first stream request.
        Provisioning in the middle also makes the request the proof: if
        SIMSTATE starts flowing after signing is on, then a signed command was
        accepted end to end, and `wait_for_truth()` is already the assertion.
        No separate handshake check is needed and none is added.

        `keystore` is a `security.keystore.Keystore`; passing None leaves the
        link unsigned and every existing caller unchanged.
        """
        from pymavlink import mavutil

        self.links = []
        for i in range(self.n):
            url = endpoint_url(i, self.base_port, self.stride)
            link = mavutil.mavlink_connection(url, dialect=self.dialect)
            self.links.append(link)
            if verbose:
                print(f"  vehicle {i}: bound {url}, waiting for heartbeat ...")
            hb = link.wait_heartbeat(timeout=timeout_s)
            if hb is None:
                raise TimeoutError(
                    f"vehicle {i}: no heartbeat on {url} within {timeout_s:.0f}s. "
                    f"Is SITL running with --serial2 udpclient:127.0.0.1:"
                    f"{self.base_port + i * self.stride}?")
            # Read arm state from the heartbeat we already have. ArduPilot
            # discards SETUP_SIGNING while armed and only warns by STATUSTEXT,
            # so provisioning needs this and cannot wait for the first pump().
            self.armed[i] = bool(hb.base_mode
                                 & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
            if verbose:
                print(f"  vehicle {i}: heartbeat, system {link.target_system}")

        if keystore is not None:
            from security.enable_signing import provision_fleet

            self.signing = provision_fleet(self.links, keystore, signing_cfg,
                                           armed=self.armed, verbose=verbose)

        for i in range(self.n):
            self.request_truth_stream(i)

    def request_truth_stream(self, i: int, hz: float | None = None) -> None:
        """Ask for SIMSTATE (and SIM_STATE as the backup) at `hz`.

        Both are requested. They cost a few hundred bytes a second and having
        the fallback already flowing means `_decode` can fail over without a
        round trip at the moment the primary stops — the one moment when a
        round trip is least affordable.
        """
        from pymavlink import mavutil

        hz = self.hz if hz is None else hz
        interval_us = int(1e6 / max(hz, 1e-6))
        link = self.links[i]
        for msg_id in (MAVLINK_MSG_ID_SIMSTATE, MAVLINK_MSG_ID_SIM_STATE):
            link.mav.command_long_send(
                link.target_system, link.target_component,
                mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
                float(msg_id), float(interval_us), 0, 0, 0, 0, 0)

    def drain_status(self, i: int) -> list[str]:
        """Take and clear vehicle i's STATUSTEXT backlog.

        ArduPilot explains every arming refusal here — "PreArm: ...", "EKF3
        waiting for GPS config", "AHRS: waiting for home". Not surfacing it
        turns a one-line diagnosis into an afternoon.
        """
        out = list(self.status[i])
        self.status[i].clear()
        return out

    def drain_acks(self, i: int) -> list[tuple[int, int]]:
        out = list(self.acks[i])
        self.acks[i].clear()
        return out

    def drain_mission(self, i: int) -> list:
        """Take and clear vehicle i's mission-protocol backlog."""
        out = list(self.mission[i])
        self.mission[i].clear()
        return out

    def link(self, i: int):
        """The raw connection for vehicle i.

        Handed out on purpose. D4.4 and D4.5 transmit on the SAME socket the
        truth arrives on, because opening a second connection to the same SITL
        instance means two sockets, two sequence counters and two chances for
        the link to be considered dead by one half of the stack while the other
        half is happily using it.

        One socket is also one place to authenticate: whatever `connect()` did
        to this link's signing state, every caller here inherits.
        """
        return self.links[i]

    def signing_report(self) -> dict | None:
        """Per-vehicle signature counters, or None on an unsigned link.

        `goodsig` at zero after a run means the far end never got the key and
        nothing we sent was ever verified; `badsig` or `unsigned` climbing is
        the security event COMMS_HARDENING_PLAN.md §6.1 wants logged, and the
        only place in this stack it is currently visible.
        """
        if self.signing is None:
            return None
        from security.enable_signing import signing_report

        return {
            "fingerprints": self.signing["fingerprints"],
            "per_vehicle": {i: signing_report(link, self.signing["logs"].get(i))
                            for i, link in enumerate(self.links)},
        }

    def close(self) -> None:
        for link in self.links:
            try:
                link.close()
            except Exception:
                pass
        self.links = []

    # -- decoding ---------------------------------------------------------
    @staticmethod
    def _decode(msg) -> tuple[float, float, float] | None:
        """(lat_deg, lon_deg, yaw_rad) from a truth message, or None.

        The SIM_STATE branch reads `lat_int`/`lon_int` and NOTHING else. If a
        build ever ships SIM_STATE without those extension fields, this returns
        None and the caller reports the source as unavailable — which is the
        correct outcome, because the float `lat`/`lon` alternative is wrong by
        metres and would be accepted silently.
        """
        kind = msg.get_type()
        if kind == "SIMSTATE":
            return (msg.lat * DEG_E7, msg.lng * DEG_E7, float(msg.yaw))
        if kind == "SIM_STATE":
            lat_int = getattr(msg, "lat_int", None)
            lon_int = getattr(msg, "lon_int", None)
            if lat_int is None or lon_int is None:
                return None
            # A build that zero-fills the extensions is indistinguishable from
            # a vehicle at the Gulf of Guinea; refuse rather than guess.
            if lat_int == 0 and lon_int == 0:
                return None
            return (lat_int * DEG_E7, lon_int * DEG_E7, float(msg.yaw))
        return None

    # -- the tick ---------------------------------------------------------
    def pump(self, budget: int = 300) -> int:
        """Drain pending traffic on every link. Returns fixes updated.

        THIS IS THE ONLY READER, and it has to be. There is one UDP socket per
        vehicle and pymavlink's `recv_match(type=...)` DISCARDS every message
        that does not match its filter — it does not put them back. So a second
        consumer calling `recv_match(type="STATUSTEXT")` on the same link is not
        reading a second stream, it is racing this one, and whichever runs first
        silently eats the other's messages.

        That cost a live run: arming failed, and the STATUSTEXT explaining why
        had already been consumed and dropped by the truth pump, so the loop
        reported "nothing armed" with no reason attached. The fix is structural
        — this method reads EVERYTHING and routes it, and `VehicleCommander`
        reads from the mailboxes below instead of from the socket.

        `budget` bounds the work per link per tick. Without it a link that has
        backed up (the process was descheduled, the estimator took a long
        update) is drained in full while the others wait, and the loop's jitter
        — the D4.7 headline number — becomes a function of the worst backlog
        rather than of the loop itself.
        """
        from pymavlink import mavutil

        updated = 0
        now = time.monotonic()
        for i, link in enumerate(self.links):
            for _ in range(budget):
                msg = link.recv_match(blocking=False)
                if msg is None:
                    break
                kind = msg.get_type()

                if kind == "HEARTBEAT":
                    self.armed[i] = bool(
                        msg.base_mode
                        & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                    self.mode[i] = int(msg.custom_mode)
                    continue
                if kind == "STATUSTEXT":
                    text = (msg.text if isinstance(msg.text, str)
                            else msg.text.decode(errors="replace"))
                    self.status[i].append(text)
                    continue
                if kind == "COMMAND_ACK":
                    self.acks[i].append((int(msg.command), int(msg.result)))
                    continue
                if kind in MISSION_PROTOCOL:
                    self.mission[i].append(msg)
                    continue
                if kind not in TRUTH_SOURCES:
                    continue

                t = self.truth[i]
                if kind == "SIM_STATE":
                    t.alt_amsl_m = float(msg.alt)
                # Both messages are requested so the fallback is already
                # flowing if the primary ever stops. While SIMSTATE is alive it
                # WINS: without this, whichever of the two arrived last in the
                # drain sets the source, and the bridge reports SIM_STATE on a
                # vehicle that is streaming SIMSTATE perfectly well. Same
                # numbers, but the run's own provenance record would be wrong.
                if t.source == "SIMSTATE" and msg.get_type() != "SIMSTATE" \
                        and t.age_s(now) < 1.0:
                    continue
                decoded = self._decode(msg)
                if decoded is None:
                    continue
                lat, lon, yaw = decoded
                t.lat_deg, t.lon_deg, t.yaw_rad = lat, lon, yaw
                t.xy_tac = self.frame.from_geodetic_2d(lat, lon)
                t.source = msg.get_type()
                t.stamp_mono_s = now
                t.count += 1
                updated += 1
        return updated

    def truths(self) -> np.ndarray:
        """(N, 2) TacFrame ENU metres. NaN for any vehicle not yet heard from.

        NaN rather than zero. Zero is a legal position — it is the anchor — so a
        vehicle that has never reported would sit exactly on the origin,
        generate plausible ranges to everything, and be indistinguishable from a
        vehicle that is genuinely there. NaN propagates and the range world
        drops the pair.
        """
        return np.array([t.xy_tac for t in self.truth], dtype=float)

    def yaws(self) -> np.ndarray:
        return np.array([t.yaw_rad for t in self.truth], dtype=float)

    def altitudes_agl(self) -> np.ndarray:
        """True height above the surveyed anchor's altitude, metres."""
        return np.array([t.alt_amsl_m - self.frame.anchor_alt_m
                         for t in self.truth], dtype=float)

    def sources(self) -> list[str]:
        return [t.source for t in self.truth]

    def source(self, i: int) -> str:
        return self.truth[i].source

    def ages_s(self) -> np.ndarray:
        now = time.monotonic()
        return np.array([t.age_s(now) for t in self.truth], dtype=float)

    def all_valid(self) -> bool:
        return all(t.valid for t in self.truth)

    def wait_for_truth(self, timeout_s: float = 30.0,
                       poll_s: float = 0.02) -> bool:
        """Block until every vehicle has produced at least one true fix."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            self.pump()
            if self.all_valid():
                return True
            time.sleep(poll_s)
        return False

    def verify_sources(self, strict: bool = True) -> dict:
        """The D4.1 acceptance: name the field every vehicle is actually using.

        Returns a report rather than printing, so the loop can log it once and
        the test can assert on it.
        """
        report = {"per_vehicle": {}, "ok": True, "degraded": []}
        for i, t in enumerate(self.truth):
            report["per_vehicle"][i] = {
                "source": t.source or "NONE",
                "fixes": t.count,
                "age_s": t.age_s(),
            }
            if t.source in FORBIDDEN_SOURCES:      # cannot happen; asserted anyway
                raise RuntimeError(
                    f"vehicle {i}: ground truth read from {t.source}, which is "
                    f"the EKF's own output — see the module docstring")
            if t.source != "SIMSTATE":
                report["degraded"].append(i)
            if not t.valid:
                report["ok"] = False
        if strict and not report["ok"]:
            missing = [i for i, t in enumerate(self.truth) if not t.valid]
            raise RuntimeError(f"no ground truth from vehicles {missing}")
        return report


def default_frame() -> TacFrame:
    """TacFrame with its origin on SITL's default home.

    `survey_sigma_m = 0.02` is not decoration — Domain 1's anchor factor uses it
    to size Sigma_A, so a surveyed anchor claims 2 cm of accuracy and no more.
    """
    return TacFrame(anchor_lat_deg=SITL_HOME_LAT, anchor_lon_deg=SITL_HOME_LON,
                    anchor_alt_m=SITL_HOME_ALT, yaw_offset_deg=0.0,
                    survey_sigma_m=0.02)


def main() -> None:
    """Standalone check: connect, report the source, print live truth."""
    import argparse

    ap = argparse.ArgumentParser(description="D4.1 ground-truth bridge check")
    ap.add_argument("-n", "--vehicles", type=int, default=1)
    ap.add_argument("-s", "--seconds", type=float, default=5.0)
    args = ap.parse_args()

    bridge = GroundTruthBridge(default_frame(), args.vehicles)
    print(f"D4.1 ground-truth bridge — {args.vehicles} vehicle(s)")
    bridge.connect()
    if not bridge.wait_for_truth(timeout_s=30.0):
        raise SystemExit("no ground truth arrived — is SIMSTATE streaming?")

    report = bridge.verify_sources()
    print("\nsource verification")
    for i, row in report["per_vehicle"].items():
        print(f"  vehicle {i}: {row['source']:<10} ({row['fixes']} fixes)")
    if report["degraded"]:
        print(f"  NOTE: vehicles {report['degraded']} fell back off SIMSTATE")

    t_end = time.monotonic() + args.seconds
    while time.monotonic() < t_end:
        bridge.pump()
        xy = bridge.truths()
        print("  " + "  ".join(f"v{i}=({p[0]:+7.2f},{p[1]:+7.2f})"
                               for i, p in enumerate(xy)), end="\r")
        time.sleep(0.1)
    print("\ndone")
    bridge.close()


if __name__ == "__main__":
    main()
