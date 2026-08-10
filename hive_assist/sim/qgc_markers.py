"""D4.9 — put the dispatch geometry on the QGroundControl map.

WHY THIS IS A MISSION UPLOAD AND NOT SOMETHING SIMPLER
------------------------------------------------------
QGC is attached to ArduPilot's `serial1` (udp 14550, see `run_fleet.sh --gcs`).
Python is on `serial0`. ArduPilot does NOT route GCS-originated traffic between
its serial ports, so a `STATUSTEXT` or `DEBUG_VECT` injected from here reaches
the autopilot and stops there — QGC never sees it. The only things that reach
QGC's map are items the autopilot itself STORES and re-serves on its other link.
A mission is one of those, which is why this module speaks the mission protocol
rather than inventing a lighter message.

The mission is never flown. The vehicles are in GUIDED the whole time and are
commanded by `SET_POSITION_TARGET_LOCAL_NED`; nothing here switches to AUTO.
These waypoints exist to be *looked at*.

WHAT GETS DRAWN
---------------
    seq 0   home            the surveyed anchor (ArduPilot reserves seq 0)
    seq 1   X_tac           the target — the coordinate the detector handed us
    seq 2.. standoff        one per elected agent, on the standoff perimeter

So a viewer sees the target and, ringed around it at `standoff_m`, the stations
the coalition actually flies to. The gap between them is the whole Domain 3
claim made visible: the swarm converges to the perimeter, never to the target.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

# ArduPilot reserves mission item 0 for home and renumbers anything put there,
# so the markers start at 1 and item 0 is written as the anchor deliberately.
HOME_SEQ = 0


@dataclass(frozen=True)
class Marker:
    """One point to draw, in TacFrame ENU metres."""

    name: str
    x: float            # east
    y: float            # north
    alt_m: float = 0.0  # relative to home; 0 keeps the icon on the ground


def markers_for_dispatch(x_tac, stations, agents, alt_m: float = 0.0
                         ) -> list[Marker]:
    """The target, then one marker per elected agent's station."""
    x = np.asarray(x_tac, dtype=float)
    out = [Marker("X_tac (target)", float(x[0]), float(x[1]), 0.0)]
    for agent, st in zip(agents, stations):
        s = np.asarray(st, dtype=float)
        out.append(Marker(f"standoff v{agent}", float(s[0]), float(s[1]), alt_m))
    return out


def _to_degE7(frame, m: Marker) -> tuple[int, int]:
    """TacFrame (x, y) -> (latE7, lonE7).

    `to_geodetic` wants a 3-vector; a bare (x, y) raises. z=0 means "at the
    anchor's altitude", which is what a map marker wants — the marker's own
    height is carried separately as a relative-alt mission field.
    """
    lat, lon, _ = frame.to_geodetic(np.array([m.x, m.y, 0.0]))
    return int(round(lat * 1e7)), int(round(lon * 1e7))


def upload_markers(link, bridge, index: int, frame, markers: list[Marker],
                   send_lock=None, timeout_s: float = 10.0,
                   verbose: bool = True) -> bool:
    """Run the mission upload handshake for one vehicle. True if ACKed.

    `send_lock` MUST be `ExternalNavFanout._send_lock` whenever the fanout
    thread is already running: pymavlink connections are not thread-safe and
    each owns a sequence counter, so two threads transmitting on one link
    corrupt each other's framing. This is the same discipline `set_origin()`
    follows.

    Failure is returned, never raised. A missing map marker is a cosmetic
    problem and must not be able to abort a flight that is otherwise healthy.
    """
    from pymavlink import mavutil

    n_items = len(markers) + 1          # +1 for the reserved home item

    def send(fn, *a, **kw):
        if send_lock is not None:
            with send_lock:
                fn(*a, **kw)
        else:
            fn(*a, **kw)

    def item(seq: int):
        """Build the MISSION_ITEM_INT for `seq`. seq 0 is home."""
        if seq == HOME_SEQ:
            lat, lon = (int(round(frame.anchor_lat_deg * 1e7)),
                        int(round(frame.anchor_lon_deg * 1e7)))
            return (lat, lon, 0.0, mavutil.mavlink.MAV_CMD_NAV_WAYPOINT)
        m = markers[seq - 1]
        lat, lon = _to_degE7(frame, m)
        return (lat, lon, float(m.alt_m), mavutil.mavlink.MAV_CMD_NAV_WAYPOINT)

    def send_item(seq: int) -> None:
        lat, lon, alt, cmd = item(seq)
        send(link.mav.mission_item_int_send,
             link.target_system, link.target_component, int(seq),
             mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT, cmd,
             0, 1, 0.0, 0.0, 0.0, 0.0, lat, lon, alt,
             mavutil.mavlink.MAV_MISSION_TYPE_MISSION)

    bridge.drain_mission(index)          # a stale reply would desynchronise us
    send(link.mav.mission_count_send,
         link.target_system, link.target_component, n_items,
         mavutil.mavlink.MAV_MISSION_TYPE_MISSION)

    deadline = time.monotonic() + timeout_s
    served: set[int] = set()
    while time.monotonic() < deadline:
        bridge.pump()                    # the ONLY reader; routes to mailboxes
        for msg in bridge.drain_mission(index):
            kind = msg.get_type()
            if kind in ("MISSION_REQUEST", "MISSION_REQUEST_INT"):
                seq = int(msg.seq)
                if seq >= n_items:
                    continue
                send_item(seq)
                served.add(seq)
            elif kind == "MISSION_ACK":
                ok = int(msg.type) == mavutil.mavlink.MAV_MISSION_ACCEPTED
                if verbose and not ok:
                    print(f"      vehicle {index}: mission REJECTED "
                          f"(type {int(msg.type)})")
                return ok
        time.sleep(0.01)

    if verbose:
        print(f"      vehicle {index}: mission upload timed out after "
              f"{timeout_s:.0f}s ({len(served)}/{n_items} items served)")
    return False


def publish(bridge, frame, markers: list[Marker], vehicles, send_lock=None,
            verbose: bool = True) -> int:
    """Upload the markers to every named vehicle. Returns how many ACKed.

    Every vehicle gets the same mission on purpose: QGC only draws the mission
    of the vehicle currently SELECTED in its UI, so uploading to one of four
    means the markers vanish when the viewer clicks another vehicle.
    """
    ok = 0
    for i in vehicles:
        if upload_markers(bridge.link(i), bridge, i, frame, markers,
                          send_lock=send_lock, verbose=verbose):
            ok += 1
    if verbose:
        names = ", ".join(m.name for m in markers)
        print(f"      QGC markers: {ok}/{len(list(vehicles))} vehicle(s) "
              f"accepted  [{names}]")
    return ok
