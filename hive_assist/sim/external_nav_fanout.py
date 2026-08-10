"""D4.4 — the estimate becomes each autopilot's position source.

Until this module runs, the estimator is a bystander: it computes a very good
position and nothing acts on it. This is the hop that makes the loop a loop —
estimate -> EKF3 -> attitude controller -> motors -> true position -> ranges ->
estimate. Everything before it is a study.

THE STREAM MUST NOT GAP. This is the operational lesson that costs the most to
relearn. ArduPilot's EKF3 treats external nav as a live sensor; a stream that
dies mid-flight is a sensor failure, and the vehicle announces it as
"AHRS: waiting for home" or simply stops trusting the position and drifts. The
estimator, meanwhile, ticks at whatever rate the range world and iSAM2 allow,
which is neither constant nor fast enough. Those are two different clocks and
`close_the_loop.py` handled the mismatch by sleeping inside the estimator loop —
its own PRODUCTION NOTE says the real answer is a separate thread. This is that
thread. It re-sends the latest estimate at a fixed rate no matter what the
estimator is doing, including while the estimator is doing a 12 ms fixed-lag
rebuild, and it records the worst gap it ever produced so the claim
"continuous" is measured rather than asserted.

ORIGIN BEFORE ARM. External nav supplies a LOCAL position only. ArduPilot will
not set home — and therefore will not arm — without a GLOBAL origin, so
`SET_GPS_GLOBAL_ORIGIN` is sent after the stream is already flowing and before
anything tries to arm. Repeats are ignored once an origin exists, so it is sent
several times to survive packet loss.

THE COVARIANCE FIELD IS REAL AND ARDUPILOT READS IT. Verified in
`GCS_Common.cpp::handle_common_vision_position_estimate_data`: covariance[0],
[6] and [11] are the packed 6x6 diagonal (x, y, z variance), and ArduPilot forms

    posErr = cbrt(cov[0]^2 + cov[6]^2 + cov[11]^2)

then `constrain_float(posErr, VISO_POS_M_NSE, 100)`. Two consequences worth
knowing before reading a log. That expression is not dimensionally a sigma —
it is a cube root of a sum of squared variances — so the number EKF3 receives is
NOT our sigma and should not be compared to it. And VISO_POS_M_NSE is a FLOOR,
so the estimator can tell the autopilot it is doing worse than the floor but
never better. Both are ArduPilot's design, not ours; we feed honest variances
into it because a fixed noise value would throw away the one thing Domain 1
exists to produce — a covariance that means something.

WHAT WE DO NOT SEND. No yaw (EK3_SRC1_YAW is the compass; the `yaw` field is
ignored) and no useful Z (EK3_SRC1_POSZ is the barometer, per the plan's "feed
position only — don't fight the climb command"). Both fields are still populated
because the message requires them, and both are documented here so that nobody
later reads a constant in a log and concludes the estimator is asserting it.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np

# Packed indices of the 6x6 covariance diagonal in MAVLink's 21-float
# upper-triangular layout: x, y, z, roll, pitch, yaw.
COV_IDX_X, COV_IDX_Y, COV_IDX_Z = 0, 6, 11

# EKF3 external-nav configuration for a GPS-denied, range-localised ArduCopter.
# Set over MAVLink at startup so a fresh SITL instance needs no prepared eeprom.
#
# GPS_TYPE = 0 is the one that carries the thesis: §1.3's observability argument
# is only non-vacuous with GPS off. Everything else here is plumbing.
EXTNAV_PARAMS: dict[str, float] = {
    "AHRS_EKF_TYPE": 3,        # EKF3
    "EK3_ENABLE": 1,
    "EK2_ENABLE": 0,
    "GPS_TYPE": 0,             # <- GPS OFF. The whole point.
    "EK3_SRC1_POSXY": 6,       # ExternalNav  <- our estimate
    "EK3_SRC1_VELXY": 0,       # None: we publish position, not velocity
    "EK3_SRC1_POSZ": 1,        # Baro
    "EK3_SRC1_VELZ": 0,        # None
    "EK3_SRC1_YAW": 1,         # Compass. A magnetometer is not a GPS; keeping it
                               # keeps attitude out of scope, which is where the
                               # restructure put it.
    "VISO_TYPE": 1,            # MAVLink vision position
    "VISO_POS_M_NSE": 0.05,    # floor on posErr, metres
    "VISO_DELAY_MS": 20,       # our measured loop latency, not a guess
}


@dataclass
class FanoutConfig:
    hz: float = 20.0                 # EKF3 wants ~20 Hz and will take more
    alt_m: float = 2.5               # commanded hold altitude (z_ned = -alt)
    origin_lat_deg: float = -35.363261
    origin_lon_deg: float = 149.165230
    origin_alt_m: float = 584.0
    origin_repeats: int = 5
    prime_seconds: float = 2.0       # stream before the origin is set

    # Transmit the moment a new estimate lands, instead of waiting for the next
    # scheduled tick. Without it the estimate sits in the slot for up to a full
    # period before going out, and that shows up in the D4.7 number as a p50 of
    # half the period -- measured at 24.8 ms on a 20 Hz sender, which on its own
    # eats the entire 25 ms budget while the estimator itself took 1.4 ms.
    # The thread then only fills gaps, which is what it was for.
    send_on_publish: bool = True


@dataclass
class FanoutStats:
    sent: int = 0
    max_gap_ms: float = 0.0
    last_send_mono: float = 0.0
    stale_publishes: int = 0
    # range-in -> on-the-wire, per estimate. This is D4.7's headline path, and
    # it is measured HERE rather than at publish() because publish() only drops
    # a value into a slot — the estimate is not out until the sender picks it up.
    air_latency_ms: list = field(default_factory=list)

    def note(self, now: float) -> None:
        if self.last_send_mono:
            gap = (now - self.last_send_mono) * 1e3
            self.max_gap_ms = max(self.max_gap_ms, gap)
        self.last_send_mono = now
        self.sent += 1


class ExternalNavFanout:
    """One background sender for N vehicles, fed by the estimator.

    `publish()` is called by the loop whenever a new estimate exists; the thread
    transmits whatever the latest values are at its own fixed rate. The two are
    decoupled on purpose — see the module docstring.
    """

    def __init__(self, links, cfg: FanoutConfig | None = None) -> None:
        self.links = list(links)
        self.n = len(self.links)
        self.cfg = cfg or FanoutConfig()

        self._lock = threading.Lock()
        # pymavlink connections are not thread-safe and each one owns a
        # sequence counter. With `send_on_publish` the loop thread and the
        # sender thread both transmit, so every write goes through this.
        self._send_lock = threading.Lock()
        self._pos = np.zeros((self.n, 2))          # TacFrame ENU metres
        self._var = np.full(self.n, 0.05 ** 2)     # per-axis variance, m^2
        self._live = np.zeros(self.n, dtype=bool)
        self._boot = time.time()
        self._pending_origin: float | None = None

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.stats = FanoutStats()

    # -- setup ------------------------------------------------------------
    def configure(self, extra: dict | None = None, settle_s: float = 0.5) -> None:
        """Push the EKF3 external-nav parameter set to every vehicle."""
        from pymavlink import mavutil

        params = dict(EXTNAV_PARAMS)
        if extra:
            params.update(extra)
        for link in self.links:
            for name, value in params.items():
                link.mav.param_set_send(
                    link.target_system, link.target_component,
                    name.encode(), float(value),
                    mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
        time.sleep(settle_s)

    def set_origin(self) -> None:
        """SET_GPS_GLOBAL_ORIGIN on every vehicle. Safe to call more than once.

        ArduPilot ignores the message once an origin exists, so repeating it is
        how the origin survives UDP loss without needing an ack.
        """
        lat = int(self.cfg.origin_lat_deg * 1e7)
        lon = int(self.cfg.origin_lon_deg * 1e7)
        alt = int(self.cfg.origin_alt_m * 1e3)
        for _ in range(self.cfg.origin_repeats):
            with self._send_lock:
                for link in self.links:
                    link.mav.set_gps_global_origin_send(
                        link.target_system, lat, lon, alt)
            # The sender thread must already be running, or these sleeps are a
            # 0.75 s hole in the external-nav stream at the exact moment the EKF
            # is deciding whether it has a usable position source. Measured as
            # an 820 ms worst gap before `prime()` was reordered to start the
            # thread first.
            time.sleep(0.15)

    # -- the estimate feed -------------------------------------------------
    def publish(self, pos, cov_trace, trusted=None,
                origin_mono: float | None = None) -> None:
        """Hand the sender a new estimate. Non-blocking, never transmits.

        `trusted` marks vehicles whose covariance the supervisor would refuse.
        An untrusted vehicle keeps receiving its LAST good position rather than
        the bad one: stopping the stream would look like sensor death and
        provoke a failsafe, while forwarding a known-bad fix is the one thing
        this whole project argues against. Holding the last good value is the
        only option that neither lies nor gaps, and it is bounded — the
        orchestrator will not command a vehicle that stays untrusted.
        """
        p = np.asarray(pos, dtype=float).reshape(self.n, 2)
        c = np.asarray(cov_trace, dtype=float).reshape(self.n)
        ok = (np.ones(self.n, dtype=bool) if trusted is None
              else np.asarray(trusted, dtype=bool).reshape(self.n))
        ok &= np.all(np.isfinite(p), axis=1) & np.isfinite(c)

        with self._lock:
            for i in range(self.n):
                if ok[i]:
                    self._pos[i] = p[i]
                    self._var[i] = max(c[i] / 2.0, 1e-9)   # trace -> per-axis
                    self._live[i] = True
                elif self._live[i]:
                    self.stats.stale_publishes += 1
            if origin_mono is not None:
                self._pending_origin = origin_mono
        if self.cfg.send_on_publish:
            self.send_once()

    # -- transmission ------------------------------------------------------
    def _send_one(self, i: int, north: float, east: float, var: float) -> None:
        link = self.links[i]
        usec = int((time.time() - self._boot) * 1e6)
        cov = [0.0] * 21
        cov[COV_IDX_X] = float(var)
        cov[COV_IDX_Y] = float(var)
        # Z is barometric; declare it as such rather than claiming our own.
        cov[COV_IDX_Z] = float(max(var, 0.01))
        link.mav.vision_position_estimate_send(
            usec, float(north), float(east), float(-self.cfg.alt_m),
            0.0, 0.0, 0.0, cov, 0)

    def send_once(self) -> None:
        """Transmit the latest estimate to every vehicle, once.

        ENU -> NED happens here: TacFrame x is East and y is North, MAVLink
        wants (north, east, down). This is the same swap `hive/supervisor_io.py`
        applies on the plan path, and it is applied in exactly these two places
        so there are exactly two places to look when a coordinate is rotated.
        """
        with self._lock:
            pos = self._pos.copy()
            var = self._var.copy()
            live = self._live.copy()
            origin = self._pending_origin
            self._pending_origin = None
        with self._send_lock:
            for i in range(self.n):
                if not live[i]:
                    continue                   # nothing true to say yet
                self._send_one(i, north=pos[i][1], east=pos[i][0], var=var[i])
        now = time.monotonic()
        if origin is not None:
            self.stats.air_latency_ms.append((now - origin) * 1e3)
        self.stats.note(now)

    def prime(self) -> None:
        """Stream a placeholder at the origin, then set the global origin.

        Order matters and is the plan's: external nav must already be flowing
        when the origin arrives, or AHRS has a home with no position to attach
        it to. `_live` is forced on for the priming window because at this point
        the estimator has not produced anything yet and the autopilot needs to
        see a stream regardless.
        """
        with self._lock:
            self._live[:] = True
        self.start()                     # the thread carries the stream from here
        time.sleep(self.cfg.prime_seconds)
        self.set_origin()

    def _run(self) -> None:
        period = 1.0 / self.cfg.hz
        next_at = time.monotonic()
        while not self._stop.is_set():
            self.send_once()
            next_at += period
            # Absolute schedule, not `sleep(period)`. Sleeping a fixed period
            # after variable work accumulates drift, and the drift shows up as
            # a slowly falling external-nav rate that nobody notices until the
            # EKF starts complaining an hour in.
            delay = next_at - time.monotonic()
            if delay > 0:
                self._stop.wait(delay)
            else:
                next_at = time.monotonic()      # fell behind; resynchronise

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="extnav",
                                        daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    # -- context manager ---------------------------------------------------
    def __enter__(self) -> "ExternalNavFanout":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def report(self) -> dict:
        air = np.asarray(self.stats.air_latency_ms, dtype=float)
        return {
            "sent": self.stats.sent,
            "max_gap_ms": self.stats.max_gap_ms,
            "target_period_ms": 1000.0 / self.cfg.hz,
            "stale_publishes": self.stats.stale_publishes,
            "continuous": (self.stats.max_gap_ms
                           < 3.0 * 1000.0 / self.cfg.hz),
            "air_latency_p50_ms": float(np.percentile(air, 50)) if air.size else float("nan"),
            "air_latency_p99_ms": float(np.percentile(air, 99)) if air.size else float("nan"),
        }
