"""
hello_drone.py  -  the handshake: your Python hears the drone's heartbeat.

SITL is already broadcasting MAVLink to UDP 14550 (see the --out line in its launch log).
We open a second listener on that port, wait for the heartbeat, then stream a little
live telemetry so you can SEE the connection is real.

Run (with SITL running):  python hello_drone.py
"""

from pymavlink import mavutil

# 'udpin' = we LISTEN on this port for whatever SITL is broadcasting.
# 14550 is the conventional MAVLink ground-station port. Remember this number.
print("connecting to SITL on udp:127.0.0.1:14550 ...")
master = mavutil.mavlink_connection("udpin:127.0.0.1:14550")

# THE HANDSHAKE: block until we hear one heartbeat. This also teaches pymavlink
# which system/component it's talking to (master.target_system, target_component).
master.wait_heartbeat()
print(f"HEARTBEAT received!  vehicle system={master.target_system} "
      f"component={master.target_component}")
print("-> the walkie-talkie works. Now listening to live telemetry (Ctrl-C to stop)\n")

# Ask the autopilot to stream data at 4 Hz so we get regular position/attitude updates.
master.mav.request_data_stream_send(
    master.target_system, master.target_component,
    mavutil.mavlink.MAV_DATA_STREAM_ALL, 4, 1)   # stream_id, rate_hz, start=1

# Listen and print a few message types so you can watch the drone "talk".
while True:
    msg = master.recv_match(blocking=True)
    if msg is None:
        continue
    t = msg.get_type()

    if t == "HEARTBEAT":
        # decode the flight mode from the heartbeat
        mode = mavutil.mode_string_v10(msg)
        armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
        print(f"[HEARTBEAT] mode={mode:<10} armed={armed}")

    elif t == "GLOBAL_POSITION_INT":
        # the drone's fused position estimate (lat/lon in 1e7 deg, alt in mm)
        print(f"[POSITION ] lat={msg.lat/1e7:.6f}  lon={msg.lon/1e7:.6f}  "
              f"alt={msg.relative_alt/1000:.1f} m")

    elif t == "ATTITUDE":
        print(f"[ATTITUDE ] roll={msg.roll:+.2f}  pitch={msg.pitch:+.2f}  yaw={msg.yaw:+.2f} rad")
