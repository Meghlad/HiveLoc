"""
feed_position.py  -  your Python SPEAKS: feed a position into the drone's EKF.

Same UDP pipe as hello_drone.py, opposite direction: instead of only receiving
telemetry, we now TRANSMIT VISION_POSITION_ESTIMATE messages. ArduCopter fuses
these as an external navigation source (once VISO_TYPE / EK3_SRC1_* are set).

KEY IDEA - COORDINATE FRAMES:
  Your estimator thinks in its own XY. The autopilot thinks in NED:
     x = North, y = East, z = DOWN  (so altitude is NEGATIVE z!)
  We must hand it (north, east, down), not raw estimator coords. Getting this
  flip right is 90% of making estimator/vision integration work.

Run (with SITL running & params set):  python feed_position.py
"""

import time
from pymavlink import mavutil

master = mavutil.mavlink_connection("udpin:127.0.0.1:14550")   # or 14551 if you added it
master.wait_heartbeat()
print(f"connected: system={master.target_system} component={master.target_component}")

# A tiny background listener so we can SEE what the EKF thinks of our input.
master.mav.request_data_stream_send(
    master.target_system, master.target_component,
    mavutil.mavlink.MAV_DATA_STREAM_ALL, 4, 1)

boot = time.time()

def send_vision_position(north, east, down, roll=0.0, pitch=0.0, yaw=0.0):
    """north/east/down are METERS in the drone's local NED frame. down is +ve downward."""
    usec = int((time.time() - boot) * 1e6)          # timestamp since we started (microseconds)
    master.mav.vision_position_estimate_send(
        usec,                                        # usec timestamp
        north, east, down,                           # position in NED (meters)
        roll, pitch, yaw)                            # orientation (radians); 0 is fine for now

print("\nStreaming a HOVER-IN-PLACE position (N=0, E=0, alt=2m -> down=-2) ...")
print("Watch for the EKF to start trusting it. Ctrl-C to stop.\n")

# We'll pretend our estimator says: hovering at origin, 2 meters up.
target_north, target_east, target_alt = 0.0, 0.0, 2.0

last_print = 0
while True:
    # SPEAK: send our estimated position ~20 Hz (vision sources want a steady stream)
    send_vision_position(target_north, target_east, down=-target_alt)   # note the -alt flip!
    time.sleep(0.05)

    # LISTEN: peek at telemetry / EKF status so we can tell it's being accepted
    msg = master.recv_match(blocking=False)
    if msg is not None and time.time() - last_print > 1.0:
        t = msg.get_type()
        if t == "LOCAL_POSITION_NED":
            # the drone's OWN fused estimate in NED - if this tracks our input, fusion works
            print(f"[EKF fused NED] N={msg.x:+.2f} E={msg.y:+.2f} D={msg.z:+.2f}  "
                  f"(we're feeding N={target_north:+.2f} E={target_east:+.2f} D={-target_alt:+.2f})")
            last_print = time.time()
        elif t == "STATUSTEXT":
            # the drone often narrates EKF source switches here - very informative
            print(f"[AP says] {msg.text}")
            last_print = time.time()
