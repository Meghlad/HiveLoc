# Sixteen things that went wrong, and what each one taught us

Every bug found while building `hive_assist`, in the order it surfaced, explained
in plain language.

Some were mistakes in the code. Some were mistakes in the **plan**. A few were
mistakes in how we were *measuring* — those were the sneakiest.

**How to read this.** Each entry has the same shape: what broke, why, and how it
was fixed. In between, an everyday comparison to make the idea stick.

Watch for a pattern: the most dangerous bugs were not the ones that made
something fail. They were the ones that made something **look fine** while being
wrong.

| | |
|---|---|
| bugs found and fixed | **16** |
| domains | **4** |
| were wrong beliefs, not wrong code | **6** |
| tests now holding them down | **218** |

---

## Domain 1 — Knowing where you are

The drones can't use GPS. They measure distances to each other and to one
surveyed post on the ground, then work out everyone's position from those
numbers.

### Bug 01 — "East" is not quite east

**What broke.** A test moved a point east along a line of latitude and checked it
landed exactly east. It missed by half a millimetre and the test failed.

The code was right. The *test* was wrong. The Earth is a ball, and lines of
longitude squeeze together as you go north. So walking along a latitude line is
not walking in a straight easterly line — it curves very slightly north.

> **Think of it like this.** Walk east around a globe near the top and you're
> really walking a small circle, not a straight line. Over 75 metres that curve
> is about half a millimetre. Real, just tiny.

**The fix.** Tested the rotation maths directly, where the answer is exact, and
loosened the map-based check to a tolerance that reflects real geometry instead
of pretending the Earth is flat.

*→ `tests/test_frames.py::test_yaw_offset_rotates_as_declared`*

### Bug 02 — Predicted three, measured one

**What broke.** We predicted that one post measuring distance to one drone at one
instant would leave three unknowns. The measurement said one.

The prediction forgot that the drones are all measuring distances to *each other*
too. They're locked into a rigid shape. Once you know the distance from the post
to six drones in a rigid shape, you've already pinned where the whole shape sits
— you just don't know which way it's turned.

> **Think of it like this.** Hold a coat hanger. Measure from your nose to six
> points on it and you know exactly how far away it is — but it could still be
> spun around to face any direction.

**The fix.** Rewrote the whole experiment to separate the two questions: *where*
is the swarm, and *which way round* is it. That split revealed something better —
a spread-out group of drones gives you the same information as one drone flying
around for a while, but instantly.

*→ `hive/nullspace.py`, the eight-row ladder*

### Bug 03 — Flying in circles forever won't tell you which way is north

**What broke.** The original plan said: if the post only measures distance, no
problem — let the drones fly around a bit and we'll work out the direction from
that. Measured across every configuration: it never works. Not after ten minutes,
not ever.

Take the entire solution — every drone, every moment, every heading — and spin it
around the post. Every single distance stays identical. Nothing you measured
changed. So nothing you measured can tell you it happened.

> **Think of it like this.** Someone spins the whole room while you're inside it,
> including you. Your distance to every wall is unchanged. You have no way to
> notice.

**The fix.** This was a bug in the *plan*, not the code, so the fix was to correct
the plan and prove the alternative. Two things actually work: survey which way the
post is **facing** (so it can report a direction, not just a distance), or plant a
second post.

In metres: distance-only pins the swarm to **2.7 cm** toward the post, and leaves
it free to swing sideways by **2 km**.

*→ `tests/test_nullspace.py::test_motion_baseline_does_not_rescue_yaw`*

### Bug 04 — A caption that described a graph we hadn't drawn

**What broke.** A chart label read "grows without bound." The line on the chart
flattened out and stayed flat. The label was describing what we expected, not what
happened.

Chasing down why led somewhere much more interesting. The old approach — declaring
one drone to be the origin — doesn't just drift away from the truth. It drifts away
while its own confidence report stays calm. It ended **1.16 m** from the real
position while reporting an uncertainty of **0.26 m**.

> **Think of it like this.** Someone lost in fog, walking confidently in the wrong
> direction, insisting they know exactly where they are. Being lost is bad. Not
> knowing you're lost is worse.

**The fix.** Replaced that chart panel with one that plots real error divided by
claimed confidence. A well-behaved system sits at 1. The old approach climbs to
**4.4**.

That number matters because the safety system reads exactly this confidence value
when it decides whether a drone is trustworthy enough to command.

*→ `tests/test_anchored_isam2.py::test_pinned_becomes_overconfident`*

---

## Domain 2 — Choosing who goes

The swarm holds a watching pattern. When a job arrives, the drones vote among
themselves on who should take it, and the ones left behind spread out to cover the
gap.

### Bug 05 — Measuring coverage with dice

**What broke.** A test said "with everything else equal, the closest drone should
win the vote." The closest drone lost.

Two problems stacked. First, everything else *wasn't* equal — part of each drone's
score is how badly its departure would tear a hole in the watching pattern, and
that differs per drone. Second, and worse: we were measuring that coverage by
scattering random sample points over the area. Random scattering is lumpy, so
drones that were identical by symmetry got scores that differed by 20% — pure luck
of the draw.

> **Think of it like this.** Judging which lamp lights a room best by dropping a
> handful of coins and seeing which ones landed in bright spots. Throw again, get
> a different winner.

**The fix.** Swapped random sampling for an evenly spaced grid, so the score is a
property of the geometry instead of the random seed. Then rewrote the test to
actually isolate distance. Added a new test asserting that drones which are
geometrically identical must score identically — the exact check that would have
caught this immediately.

*→ `tests/test_cbba.py::test_symmetric_ring_gives_symmetric_criticality`*

### Bug 06 — The safety check that watched only half the sky

**What broke.** The mission log proudly reported zero safety violations. It also
reported that two drones had come within **1.159 m** of each other, when the hard
floor was **1.2 m**. Both statements were in the same output, three lines apart.

The design splits the swarm into two teams that coordinate separately — that's
deliberate and good. But the safety check had quietly inherited the split. It was
checking the moving team against itself, and the waiting team against itself, and
never one against the other. So a re-positioning drone could drift right up to a
stationary teammate and nothing was looking.

```
   ┌── CHECKED ────────┐         ┌── CHECKED ────────┐
   │  ●      ●         │         │  ●        ●    ●  │
   │              ●────┼─────────┼──●                │
   └───────────────────┘         └───────────────────┘
                       └ 1.159 m ┘
                    nobody checking this one
```

> **Think of it like this.** Two teachers on playground duty. One watches the left
> side, one watches the right. Both truthfully report "no problems in my area"
> while two kids collide right on the line between them.

**The fix.** The safety check now always receives the positions of every drone
*not* in the plan it's checking, and treats them as obstacles. The path builder
steers around them too. On top of that, every drone is now given an instruction
every tick — the moving ones get a new destination, the waiting ones are told to
hold — so the real safety system sees the whole formation instead of a slice.

Clearance went to **1.273 m**.

*→ `tests/test_mission_fsm.py::test_static_agents_are_included_in_the_clearance_check`*

### Bug 07 — The fix broke a rule we'd written on purpose

**What broke.** Telling every drone what to do every tick — the fix for bug 06 —
immediately caused a wave of rejections. The safety system was refusing every
single plan.

It was right to. There's a rule that says you must never command a drone you can't
confidently locate. By instructing *everyone*, we were also instructing drones the
estimator had lost track of, and the safety system correctly slammed the door.

> **Think of it like this.** You wouldn't shout directions at someone you can't
> see. But you'd still walk around where you last saw them.

**The fix.** Two different lists, for two different purposes. Drones we trust go in
the plan and get told what to do. Drones we've lost track of are left out of the
plan entirely — but still handed to the safety check as things to avoid. Not being
steerable doesn't stop an aircraft taking up space.

*→ `tests/test_mission_fsm.py::test_untrusted_agent_cannot_be_elected`*

---

## Domain 3 — Flying there

One drone gets the go-signal, breaks away, and flies a curved path to a viewing
position a safe distance from the target. It never flies at the target itself.

### Bug 08 — A curve type that was never once chosen

**What broke.** Shortest curved paths come in six shapes. Testing across 3,000
random journeys, one shape was picked 142 times — and its mirror image was picked
**zero** times.

Left and right are symmetric. If one shape is useful sometimes, its mirror must be
too. Zero was impossible. The formula for that shape had a wrong term, so it
produced paths that ended in the wrong place — and a safety check I'd built was
quietly throwing them away every time.

> **Think of it like this.** Nobody in the whole school is left-handed. Something
> is wrong with how you're counting, not with the school.

**The fix.** Corrected the formula. Now the two mirror shapes get picked 97 and 91
times — the balance you'd expect. The check that had been hiding the bug was also
what made it findable, so it stayed, and a test now asserts all six shapes get used
at roughly mirror-equal rates.

*→ `tests/test_approach.py::test_all_six_dubins_words_get_used`*

### Bug 09 — Perfectly on course, two kilometres past the destination

**What broke.** Three of four mission types never arrived. The drone flew to the
viewing position, sailed straight past it, and kept going — **1,955 m** beyond —
while reporting a steering error of **0.000 m** the entire way.

The steering law's only job is to keep the drone *on the line*. It has no concept
of stopping. Once it reached the end of the planned path it simply carried on down
the same line, staying perfectly on course, forever. It was doing its job
flawlessly.

```
                        ╭─ stop here
   ─────────────╮      (◯)- - - - - - - - →
                ╰───────┘   still on the line
                            still going
```

> **Think of it like this.** A train perfectly centred on its rails is not the same
> as a train that stopped at the station. The rails don't end at the platform.

**The fix.** Added a landing phase. Within five metres of the destination, the
drone stops following the line and starts aiming straight at the spot, slowing down
as it closes. All four mission types now arrive.

The tests for "steers accurately" and "actually arrives" are now separate, because
they always were.

*→ `tests/test_approach.py::test_the_vehicle_does_not_fly_past_the_station`*

### Bug 10 — A route the aircraft physically cannot fly

**What broke.** The drone kept drifting wide on every turn. It looked like a badly
tuned steering gain. It wasn't.

The planned route asked for turns of 4-metre radius. At its cruising speed and
maximum turn rate, the tightest circle this aircraft can hold is 4.17 metres. The
route was asking for something impossible, and the steering was being blamed for
failing to deliver it.

> **Think of it like this.** Drawing a racing line that goes through the inside of
> a corner the car can't physically take, then complaining about the driver.

**The fix.** The turn radius is now calculated from the aircraft's own speed and
turn-rate limits, with 15% margin, instead of being typed in. Ask for something
tighter and you get a clear error at the moment you ask, rather than a mystery a
hundred metres later.

*→ `tests/test_approach.py::test_an_unflyable_turn_radius_is_refused`*

### Bug 11 — The simulation's own maths was drifting off the curve

**What broke.** Even after the last two fixes, the drone rode about a metre wide
through turns. The steering was fine. The *simulator* was wrong.

The simulation advanced the drone in straight hops, ten times a second. But the
drone is turning during each hop, so a straight hop cuts the corner. Do that a few
hundred times around a curve and the error piles up into something that looks like
a real control problem.

> **Think of it like this.** Drawing a circle using short straight pencil strokes.
> Each one is nearly right; the finished shape is visibly a polygon.

**The fix.** Replaced the straight hops with the exact curved motion, which has a
clean formula when speed and turn rate are steady between updates — which is also
what the real aircraft actually does. The phantom error disappeared.

*→ `hive/standoff.py`, the exact unicycle step in `follow()`*

### Bug 12 — A ruler that switched what it was measuring from

**What broke.** The error chart showed a sudden jump to **1.36 m** partway through
the flight. Digging into the raw numbers, the drone was within **4 cm** of its path
the whole time. The spike was invented by the measurement.

The landing phase from bug 09 changed which reference line the error was measured
against. The drone didn't move. The ruler did. And a chart that jumps when nothing
physical jumped will eventually be read by someone as a real event.

> **Think of it like this.** Measuring your height from the floor, then from the
> tabletop, and reporting that you shrank.

**The fix.** The reported error is now always the distance to the nearest point on
the planned path, in both phases — one quantity, measured the same way throughout.
Deliberately a different calculation from the one the steering uses internally,
because the steering needs a control signal and a human needs an honest number.

True peak: **17 cm**.

*→ `hive/standoff.py::cross_track_to_path`*

---

## Domain 4 — Surviving a bad radio link

Radio messages get delayed, dropped and reordered. The rule is that whenever
anything goes wrong, the drone must freeze in place — never make a sudden move.

### Bug 13 — The biggest one: freezing safely, then leaping

**What broke.** Under 30% packet loss, a drone that had been sitting still was
commanded to move **7.2 m** in one step. A legal step is **0.9 m**. Every existing
safety check approved it.

The freezing half worked perfectly — with no messages arriving, nothing was
commanded and the drone held. The danger was in the **recovery**. While the drone
sat frozen, the planner upstream had no idea and kept preparing instructions: step
40, step 60, step 80. When the radio came back, the first instruction to arrive was
step 80 — a destination far from where the drone had been parked.

And it passed every check, because it was a perfectly good instruction. It was
recent. It was inside the boundary. It was properly spaced. It named a trustworthy
drone.

**The safety system limits how *old* an instruction may be. Nothing limited how
*far* it may ask you to go.**

```
   ●·············································◯
   drone,          ·  ·  ·  (planner kept going)   planner got to here
   frozen
   └──────────────── 7.2 m in one step ────────────┘
                approved by every check
```

> **Think of it like this.** You freeze on the spot while your friend keeps walking
> and calling directions. The line drops. When it reconnects they say "meet me
> here" — and "here" is now eight streets away. Their instruction is fresh, polite
> and completely reasonable. Following it in one stride is not.

**The fix.** A new check that rejects any instruction asking a drone to travel
further than it could physically fly in one step. Worst commanded move dropped from
**7.2 m** to **0.90 m** at every loss level.

This is now a written recommendation for the main Rust safety system, which doesn't
have it yet — about ten lines in `brain/rust/swarm-supervisor/src/lib.rs`.

*→ `tests/test_safe_hold.py::test_stale_stream_without_a_gate_lunges`*

### Bug 14 — My first version of the fix would have allowed the exact leap

**What broke.** The new check gave a drone a travel allowance based on how long it
had been waiting. Wait two seconds, get two seconds' worth of distance. It sounded
generous and sensible.

It's exactly backwards. The drone was *frozen*. It banked no distance by waiting.
Paying it for the wait would have stamped the 7.2 m leap as affordable — the check
would have politely approved the very thing it existed to prevent.

> **Think of it like this.** Standing still for an hour doesn't mean you can now
> cover a mile in one stride. You didn't save up the steps.

**The fix.** The allowance is one step's worth, always, because the instruction is
for one step. There's a test named after this mistake so nobody re-derives the
clever-sounding version later.

*→ `tests/test_safe_hold.py::test_slew_gate_budget_is_per_tick_not_per_elapsed_second`*

### Bug 15 — Safe, and going nowhere

**What broke.** With the new check switched on, the drone stopped lunging — and
stopped arriving. Mission progress collapsed to 9%. It sat still, safely,
indefinitely.

Once frozen, the planner kept advancing away from the stationary drone, so every
instruction that arrived was too far and got rejected. Then the next one was
further still. The drone could never catch up to its own plan.

> **Think of it like this.** A friend who keeps walking ahead while telling you
> where to meet. Every new meeting point is further away than the last. You'll
> never make one.

**The fix.** The planner now throws away its stale plan after a pause and starts
fresh from where the drone actually *is*. With both pieces together: **0.90 m**
maximum step and **100%** of missions completed, at every loss level up to 30%.

Neither piece works alone — one gives you a leap, the other gives you a stall.

*→ `tests/test_safe_hold.py::test_replanning_plus_the_gate_gives_both`*

### Bug 16 — Two small ones worth keeping

**What broke.** A test claimed the leap gets worse as the radio gets worse. The
trend was obvious in the numbers, but the test failed anyway. And a script had a
quote-nesting mistake that no syntax checker could see.

The test used a maths tool that assumes a smooth straight-line relationship. But
leap size only comes in whole steps and stops growing past a certain point, so that
tool understated a trend that was plainly there — it read 0.57 where the right tool
read 0.80.

The script bug was a small Python snippet embedded in a shell script, with quote
marks nested one layer too deep. The shell checker saw a valid string. Python would
only have discovered it mid-experiment, on the other laptop.

> **Think of it like this.** Using a ruler to measure a staircase and concluding it
> isn't really going up.

**The fix.** Switched to a tool that ranks values rather than assuming a straight
line, and added a plain group comparison beside it. Rewrote the embedded snippet in
a language that doesn't need the nested quotes at all.

*→ `tests/test_safe_hold.py::test_lunge_magnitude_grows_with_loss`, `sim/netem_sweep.sh`*

---

## What all sixteen have in common

Only two of these were ordinary broken code. The rest fall into three groups, and
the grouping is the real lesson.

### Wrong belief, right code — bugs 02, 03, 05, 10

The code did exactly what it was told. What it was told was based on something we
assumed and had never measured.

### The measurement was lying — bugs 04, 11, 12, 16

The system was fine; the ruler, the caption or the statistic was wrong. These are
the worst kind — they send you fixing something that isn't broken.

### Fine on its own, broken together — bugs 06, 07, 09, 13, 15

Every piece behaved correctly in isolation. The fault lived in the gap between two
pieces that each assumed the other was handling it.

---

**The thread running through the dangerous ones:** none of them announced
themselves.

- The clearance bug reported zero violations.
- The fly-past reported zero steering error.
- The leap was approved by every safety check we had.
- The drifting estimator reported calm confidence.

That's why the fix for each one was a test that fails loudly — and why several of
those tests check a **property** rather than a value:

> "All six curve shapes must get used."
> "Identical drones must score identically."
> "Steering accuracy and arrival are separate claims."

Those are the checks that catch the bug you didn't think to look for.
