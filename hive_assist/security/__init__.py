"""Comms hardening — the link layer under Domain 4.

Scope is `COMMS_HARDENING_PLAN.md` stage **H0**: MAVLink 2 signing on every
link, one key per vehicle, unsigned frames refused. Nothing here is offensive
tooling; every attack named in a docstring is named so the code can refuse it.

Two modules, matching the plan's deliverables:

    keystore.py        H2.1 — key generation and encrypted-at-rest storage
    enable_signing.py  H1.1 — SETUP_SIGNING provisioning + reject-unsigned

Both are import-safe with no SITL running and no MAVLink socket open, the same
property `sim/` holds, so `tests/test_reject_unsigned.py` (H1.3) exercises the
real reject path offline.
"""
