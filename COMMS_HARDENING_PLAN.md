# Comms Hardening & Link Resilience Plan

> **Status:** defensive security plan for the anchored swarm's communication links.
> **Builds on:** the MAVLink pipeline (`close_the_loop.py`, external-nav fan-out),
> the Rust `swarm-supervisor` (the enforcement chokepoint), and the GPS-denied
> architecture (already a spoofing-resistance property).
> **Scope:** protecting *our own* links. This document contains no offensive tooling
> (no jammers, spoofers, or exploit code); those sections describe attacks only to
> defend against them.

---

## 0. Honest posture — what "hardened" means here

There is no such thing as an unhackable radio link, and any plan claiming one is a
red flag to anyone who reads it. Availability in particular cannot be guaranteed: a
resourced adversary with enough transmit power can always deny an RF channel — that
is physics, not an engineering gap. So the achievable goal is a link that is:

1. **Confidential & authenticated** — an eavesdropper learns nothing, and no one but
   the trusted ground/companion side can issue a command the vehicle will act on.
2. **Fail-safe under attack** — when the link is jammed, replayed, or fed a forged
   position, the vehicle degrades to a safe state instead of doing something dangerous.
3. **Observable** — every rejected signature, link drop, and implausible estimate is
   logged, so an attack attempt is *seen*, not silently absorbed.

Confidentiality and integrity against casual and opportunistic attackers (Tiers 0–1
below) are fully achievable with commodity tooling. Resisting a resourced adversary
(Tier 2) raises the bar meaningfully but is bounded by hardware and law. A
nation-state adversary (Tier 3) is explicitly out of scope. Naming these tiers is the
point of the threat model — it stops you from over-building one layer while another
stays open.

---

## 1. Threat model

### 1.1 Assets to protect (in priority order)

| # | Asset | Why it matters |
|---|---|---|
| A1 | **Command integrity** — only the trusted operator can command the vehicle | A forged command flies the drone. Highest value. |
| A2 | **Position-estimate integrity** — the `VISION_POSITION_ESTIMATE` fed to EKF3 | Your design's unique surface: a forged estimate *is* a confidently-wrong pose, and the vehicle flies on it. See §1.3. |
| A3 | **Range-mesh integrity** — inter-drone + anchor ranges | The trust root of the whole estimate. Corrupt the ranges, corrupt the position. |
| A4 | **Availability** — the link stays up under interference | Can't be guaranteed; the requirement is graceful, safe degradation. |
| A5 | **Telemetry confidentiality** — mission data, positions | Lower stakes than A1–A3, but leaks operational intent. |
| A6 | **Fail-safe integrity** — the supervisor can't be disabled or bypassed | If the safety chokepoint falls, every other property is moot. |

### 1.2 Adversary tiers

- **Tier 0 — accidental / ambient.** Co-channel interference, other 2.4/5 GHz traffic,
  a neighbour's video link. Not malicious. Defended by basic RF hygiene + failsafe.
- **Tier 1 — opportunistic.** A hobbyist with an SDR (RTL-SDR/HackRF) and public tools:
  passive sniffing, **replay** of captured MAVLink, injection of unauthenticated
  commands, naive de-auth of an open WiFi link. This is the realistic threat for most
  civilian operations and the one this plan fully closes.
- **Tier 2 — resourced.** Directional/narrowband jamming, GPS spoofing, protocol-aware
  MAVLink injection, distance-spoofing of UWB ranges, possibly physical proximity.
  Raised against, not eliminated — the honest deliverable here is *cost + detection +
  safe failure*.
- **Tier 3 — nation-state.** Wideband barrage jamming, supply-chain compromise,
  cryptanalytic resources. **Out of scope**, stated so no layer pretends otherwise.

### 1.3 Attack surfaces, mapped to *this* architecture

- **MAVLink C2 link (ground/companion ↔ vehicle).** Plaintext MAVLink is
  unauthenticated and unencrypted by default — trivially sniffed, replayed, injected
  by Tier 1. Primary surface for A1/A5.
- **External-nav injection path (`VISION_POSITION_ESTIMATE`).** *The highest-leverage
  attack on your specific design.* Your vehicles fly on an externally injected
  position. An attacker who can inject or tamper this stream doesn't need to touch the
  motors — they redefine where the drone thinks it is and let the autopilot fly it into
  the ground. This is A2, and it maps directly onto your existing thesis: a spoofed
  estimate is a confidently-wrong estimate, which is exactly what the supervisor and
  the error/σ discipline already exist to catch.
- **Range mesh (UWB / radio ranging).** A spoofed tag or distance-enlargement/reduction
  attack corrupts the factor graph's measurements (A3). Standard UWB ranging is not
  distance-authenticated unless secure-ranging (802.15.4z STS) is used.
- **GPS — largely mitigated by design.** Because the swarm is GPS-denied, GPS spoofing
  cannot directly move the vehicle; there is no GPS fix in the control loop to poison.
  This is a real, nameable security win — but it *shifts* the trust root onto the anchor
  and range mesh (A3), so those inherit the scrutiny GPS would otherwise get.
- **Companion computer (M1 / edge).** Holds keys and runs the estimator. Key material at
  rest and the integrity of the estimator process are surfaces (A6).
- **The supervisor itself (A6).** If it can be bypassed or its config forged, safe
  failure is gone. It must be the only path to a motor setpoint, and its inputs must be
  authenticated.

---

## 2. Layer 1 — Transport security (authenticate first, then encrypt)

**2.1 MAVLink 2 message signing — the non-negotiable floor.**
Enable MAVLink 2 signing on every link. Signing appends an HMAC-SHA256-based signature
plus a 48-bit timestamp to each frame, keyed by a per-link secret. Effect: the vehicle
rejects any command not signed with the shared key (kills Tier-1 injection), and the
timestamp + link-id monotonicity kills **replay** of captured frames. Configure the
autopilot to **reject unsigned messages** on the C2 and external-nav channels — signing
that still accepts unsigned frames buys nothing.

**2.2 Confidentiality tunnel — where the link carries IP.**
Signing authenticates but leaves payloads readable. Where both ends are under your
control and the link is IP-capable (WiFi, LTE, IP mesh radio), wrap MAVLink in an AEAD
tunnel — **WireGuard** (simplest, modern, low overhead) or **DTLS**. This gives
confidentiality + integrity + replay protection at the transport layer, and WireGuard's
static-key model fits a fleet cleanly.

**2.3 Latency budget — measure it against the loop.**
Your swarm plan's acceptance bar is a < 25 ms closed loop. Signing is cheap (µs-scale
HMAC). A tunnel adds handshake + per-packet overhead; on localhost/LAN it's small but
non-zero. **Requirement:** re-run `loop_latency.py` (swarm plan D4.7) with signing on,
then with the tunnel on, and confirm the loop still closes under budget. Security that
blows the latency budget silently breaks flight — treat it as a measured tradeoff.

**Deliverables H1.x**
- H1.1 `security/enable_signing.py` — set per-vehicle signing keys via `SETUP_SIGNING`,
  configure reject-unsigned on C2 + external-nav channels.
- H1.2 `security/wg_tunnel/` — WireGuard config generator per vehicle (IP links only).
- H1.3 `tests/test_reject_unsigned.py` — a forged/unsigned command is dropped; a
  replayed frame (stale timestamp) is dropped.

---

## 3. Layer 2 — Identity & key management

**3.1 Per-vehicle keys, never a shared fleet secret.** One key per vehicle for signing,
one tunnel keypair per vehicle. Compromising one airframe must not compromise the fleet.

**3.2 Keys at rest.** Do not store signing keys or tunnel private keys in plaintext on
the companion computer. Use the OS keychain / an encrypted keystore, and a hardware
secure element / TPM where the platform has one. On the M1 dev box, at minimum an
encrypted file with a passphrase-derived key; on real companion hardware, a secure
element is the target.

**3.3 Rotation & provisioning.** Define a rotation schedule and an on-compromise
rotation path (re-key via `SETUP_SIGNING`, re-issue tunnel keys). Provision keys during
a trusted setup phase over a wired/local channel, never over the air in the clear.

**Deliverables H2.x**
- H2.1 `security/keystore.py` — per-vehicle key generation, encrypted-at-rest storage,
  retrieval for the signing/tunnel setup.
- H2.2 `security/rotate_keys.py` — scheduled + on-demand rotation.
- H2.3 `docs/provisioning.md` — the trusted-setup procedure.

---

## 4. Layer 3 — Radio-layer resilience & the range mesh

**4.1 Spread spectrum / frequency hopping.** Narrowband jamming (Tier 1–2) is defeated
far more easily against a hopping link. Prefer C2 radios with FHSS/DSSS (e.g.
RFD900-class SiK radios, or LoRa for long-range low-rate telemetry). This is a
*hardware* property — the M1 and a software stack can't add it to a fixed-frequency
link. Note it as a hardware selection criterion, not a code task.

**4.2 Band/antenna diversity.** Where budget allows, a second link on a different band
(e.g. a primary 2.4/5 GHz data link plus a low-rate 900 MHz telemetry/failsafe link)
means a single-band jammer can't take the whole vehicle dark — the failsafe channel
survives to command a safe recovery.

**4.3 Secure ranging for the mesh (A3).** Standard UWB two-way ranging can be
distance-spoofed. Where the ranging hardware supports it, enable **802.15.4z secure
ranging (STS)**, which authenticates the timestamp sequence and resists
distance-enlargement/reduction. Where it doesn't, treat ranges as untrusted-but-gated:
the Huber kernels already on your inter-agent and anchor factors bound the influence of
a small number of spoofed ranges, and a sudden geometry-inconsistent range set should
raise a detection event (§6).

**4.4 GPS-denied is a feature — say so.** Because there is no GPS in the control loop,
GPS spoofing (a common, cheap Tier-1/2 attack on conventional drones) has no direct
purchase here. Document this as an intentional resilience property. The corollary: the
anchor and range mesh now carry the trust GPS would have, which is why §4.3 and the
detection layer matter.

**Deliverables H3.x** (mostly selection/config, not code)
- H3.1 `docs/radio_selection.md` — FHSS C2 + diversity failsafe link criteria.
- H3.2 `security/secure_ranging.md` — enable STS if supported; range-plausibility
  gating fallback if not.

---

## 5. Layer 4 — Fail-safe behaviour (enforced at the supervisor)

The security property that survives a partial compromise is: **nothing reaches the
motors except through the supervisor, and the supervisor fails safe.** Everything here
is enforced at that single chokepoint you already have.

**5.1 On authentication failure** (bad/absent signature): drop the command, log it
(§6), do not act. Repeated failures escalate to a link-untrusted state.

**5.2 On link loss** (jamming / range): the standard ladder, wired through ArduPilot
failsafe params and mirrored in the supervisor — geofence hold → loiter → return-to-
launch → land/disarm, chosen by how long the link has been down and where the vehicle
is. The low-rate diversity link (§4.2), if present, is what lets a human intervene
before the ladder reaches land.

**5.3 On implausible position estimate** (A2 — spoofed `VISION_POSITION_ESTIMATE`):
this is where your existing work pays off. EKF3's innovation gating rejects estimates
that disagree too hard with the vehicle's own dynamics; the supervisor adds a
plausibility gate — a position that jumps faster than physically possible, or whose
reported σ disagrees with its error against the range residuals, is refused and the
vehicle holds. A spoofed estimate is a confidently-wrong estimate, and the whole project
is already built to catch those. Make that gate explicit and test it with an injected
teleport.

**5.4 The supervisor's own integrity** (A6): its inputs (`EstimateSnapshot`, `Plan`)
must arrive over authenticated channels only; its config/invariants are loaded from a
signed source; it runs as a trusted native process, not something an over-the-air
message can reconfigure.

**Deliverables H4.x**
- H4.1 `security/failsafe_ladder.md` + ArduPilot param set — link-loss ladder.
- H4.2 supervisor plausibility gate (extends `swarm-supervisor`): reject
  physically-impossible position jumps and σ/residual mismatches.
- H4.3 `tests/test_spoofed_estimate.py` — an injected teleport is refused; the vehicle
  holds instead of chasing it.

---

## 6. Layer 5 — Detection & monitoring

A link you can't observe being attacked is quiet, not hardened.

**6.1 Log the security-relevant events:** rejected signatures, stale-timestamp
(replay) drops, MAVLink sequence-number anomalies, RSSI/link-quality collapses
(possible jamming), EKF innovation spikes and supervisor plausibility rejections
(possible estimate/range spoofing).

**6.2 Tamper-evident trail:** append-only, ideally signed/hash-chained, so a post-
incident review can trust the log wasn't edited.

**6.3 Live alerting:** surface anomaly patterns (a burst of signature failures, a
coordinated RSSI drop across vehicles) as operator alerts rather than buried log lines.

**Deliverables H5.x**
- H5.1 `security/audit_log.py` — hash-chained security event log.
- H5.2 `security/monitor.py` — anomaly signals + operator alerts; if Domain 5 (ROS 2)
  exists, publish on `/security_events`.

---

## 7. Regulatory & hardware constraints (verify before flying)

Radio hardening runs straight into spectrum law — design something you can't legally
transmit and the plan is worthless. In India this is governed by **WPC (Wireless
Planning & Coordination Wing, DoT)** for spectrum, **TEC** for equipment certification,
and the **DGCA Drone Rules 2021** for operations. De-licensed ISM bands exist (portions
of 2.4 GHz, 5 GHz, and the sub-GHz range) with **bounded transmit power and channel
rules**, and frequency-hopping/power configurations that are legal elsewhere may not be
here.

**Do not treat the numbers in any tutorial as current.** Confirm the present WPC band
allocations, permitted EIRP, and TEC certification requirements against the primary
sources before selecting radios or configuring hopping/power. This is a "verify against
the authority" item, not something to take from memory — mine or anyone's.

Hardware reality: FHSS, band diversity, and secure UWB ranging are **radio-hardware**
capabilities. The M1 + software stack implements signing, tunnelling, key management,
fail-safe logic, and monitoring; it cannot add spread-spectrum to a fixed-frequency
radio. Separate the two when budgeting.

---

## 8. Staged build order

| Stage | Delivers | Closes | Status |
|---|---|---|---|
| **H0** | MAVLink 2 signing everywhere, per-vehicle keys, reject-unsigned (H1.1, H1.3, H2.1) | Tier-1 injection & replay on C2 + external-nav | ✅ **done** — `--sign` |
| **H1** | Harden the `VISION_POSITION_ESTIMATE` path + plausibility gate (H4.2, H4.3) | A2, this design's unique surface | ✅ **done** — `--guard` |
| **H2** | Confidentiality tunnel on IP links, latency re-measured (H1.2, §2.3) | A5, and hardens A1 further | ▢ not built — see §8.1 |
| **H3** | Radio resilience: FHSS + diversity + secure ranging (H3.x) | Tier-2 jamming/range-spoof, A3/A4 | ◐ **H3.2 done in software** (`range_integrity`); FHSS/diversity/STS are hardware |
| **H4** | Fail-safe ladder + tamper-evident logging + monitoring (H4.1, H5.x) | A4/A6, and makes attacks visible | ◐ **H5.1 done** (`audit_log`); H4.1 ladder + H5.2 alerting not built |
| **H5** | Key rotation, provisioning procedure, regulatory sign-off (H2.2–H2.3, §7) | Operational durability | ◐ **H2.2 done** (`rotate_keys`); H2.3 + §7 outstanding |

### 8.1 What is built, and the honest shape of what is left

**Built and tested (53 tests, in a suite of 373 passing):**

| Module | Stage | What it refuses |
|---|---|---|
| `security/keystore.py` | H2.1 | per-vehicle keys, scrypt + encrypt-then-MAC at rest, no default passphrase |
| `security/enable_signing.py` | H1.1/H1.3 | unsigned and replayed frames; the serial0/channel-0 finding, resolved by moving the loop to serial2 |
| `security/plausibility.py` | H4.2 | positions that cannot be true — teleports, and **an estimate whose error disagrees with its own reported σ** |
| `security/range_integrity.py` | H3.2 | corrupted ranges, **asymmetrically** — a short reading has no benign explanation, a long one does |
| `security/rotate_keys.py` | H2.2 | a rotation that would write down a key the vehicle never received |
| `security/audit_log.py` | H5.1 | a silently edited record of any of the above |
| `security/guards.py` | — | wires H1/H3.2 into the D4 loop at the two seams that work |

**Deliberately not built, with the reason:**

- **H2 confidentiality tunnel.** WireGuard on a loopback SITL link would be
  theatre: it would encrypt traffic between two processes on one host and prove
  nothing about a radio. This is a deployment task for a real IP link, and the
  latency budget (§2.3) has to be re-measured there rather than here.
- **H3.1 FHSS / band diversity, H3.3 UWB STS.** Radio-hardware properties. No
  amount of software adds spread-spectrum to a fixed-frequency link, and this is
  stated in §4.1 rather than papered over. `range_integrity` is what you run
  *because* STS is usually unavailable, and alongside it when it is not.
- **H2.3 provisioning procedure and §7 regulatory sign-off.** Both are
  operational documents that must be written against a specific site and a
  current reading of WPC/TEC allocations — not from memory, per §7.
- **Secure element for keys.** Hardware. `keystore.py` says so in its own
  docstring rather than implying otherwise.
- **The netem / RF degradation sweep.** `hive/loss_model.py` runs the loss study
  offline; it has never been run against the live loop under real packet loss.

**First move (historical):** H0 was the right first move and it was taken. The
highest-value item after it was H1, because signing authenticates a *sender* and
says nothing about whether the payload can be true — and this project's own
forensic history is the proof that the estimate can be confidently wrong with no
attacker present at all. That is now closed, and the same gate catches both
cases because the safe response to each is identical.

**Next move:** the H4.1 fail-safe ladder. Every gate above refuses bad input;
none of them yet decides what the vehicle *does* about a link that has been down
for thirty seconds. `hive/loss_model.py` has the measurement (gate alone stalls
the mission to 9% progress at 30% loss; gate plus re-planning gives 0.90 m worst
jump at 100% progress) but the ladder itself — hold → loiter → RTL → land, keyed
on outage duration — is not wired to ArduPilot's failsafe parameters.

---

## 9. One-paragraph honest version

This plan makes the swarm's links confidential and authenticated against realistic
civilian attackers, fails the vehicle safe when a link is jammed or a position is
forged, and logs every attempt so nothing happens quietly — with the highest-leverage
surface (the injected `VISION_POSITION_ESTIMATE` your vehicles fly on) hardened
explicitly and enforced at the supervisor you already trust. It does not claim
unbreakability: a resourced adversary can still deny the RF channel, and the honest
answer to that is detection plus safe failure, not a promise it can't happen. Being
GPS-denied by design already removes one of the most common attacks for free; the cost
is that the anchor and range mesh inherit that trust, which is why they're gated and
monitored rather than assumed.
