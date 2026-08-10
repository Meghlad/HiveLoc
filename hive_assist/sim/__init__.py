"""Domain 4 — the range-only, MAVLink-native closed loop.

Every module here is import-safe without a running SITL: the MAVLink-touching
classes only open sockets in `connect()`, and the pure geometry/estimation parts
(`range_world`, `live_estimator`) never touch a socket at all. That is what lets
`tests/` exercise the real loop code offline.
"""
