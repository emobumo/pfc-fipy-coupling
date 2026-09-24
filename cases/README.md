# Application cases

Engineering application cases built on the verified solver engine. These are
**not** verification-ladder tests (`tests/test_verification_ladder.py`): they do
not assert, they produce research results. Code is tracked here; the figures and
logs they generate go to `outputs/`, which stays untracked.

The numbers quoted below are **regression anchors**: the cases are deterministic
and a rerun reproduces them bit-for-bit (identical figures, byte for byte). If a
number here moves, the engine changed. Anchors are for the **v0.6 parameter set**
on the 60x60 m domain (`variables.grout_material_v06`), under the **scipy PCG**
linear solver that `run_local.ps1` has set as default since 2026-09-22. Anchors
for the previous parameter set survive only where a heading says so.

**Long runs on this machine die of segmentation faults and of Windows
bugchecks, and the cause is very likely the machine, not FiPy.** Two blue
screens on 2026-09-22/23 with *different* stop codes (0xFC
ATTEMPTED_EXECUTE_OF_NOEXECUTE_MEMORY and 0x20001 HYPERVISOR_ERROR), a
corrected WHEA hardware error, and repeated user-mode segfaults in the
PFC-bundled Python, all under sustained numerical load. An earlier note here
blamed concurrency for the scipy segfault; that is wrong -- the gradient case
segfaulted on 09-23 with nothing else running. Treat any crash as a crash to
retry, not as a result: the march is deterministic, so a completed rerun is the
same run.

Every case checkpoints itself (`src/analysis/run_checkpoint.py`), by default
every 200 steps, into `outputs/checkpoints/`. **After a crash, re-run the same
command**: the case finds its checkpoint, resumes bit-identically, and deletes
it when the run finishes cleanly. A checkpoint whose fingerprint does not match
the current geometry and parameters is ignored rather than resumed, so editing a
case invalidates its checkpoints instead of silently continuing a run that never
existed. `--ckpt-every 0` turns it off; `inclined_hole_gradient.py` also takes
`--only main|sweep|all` to run its sub-cases as separate processes.

Because silent corruption is possible and these cases are deterministic,
**every number quoted here should be reproduced by a second independent run and
compared bit-for-bit** before it goes into the thesis. Two identical runs make
silent corruption implausible; a crash is the lucky case, because it announces
itself.

Run with the PFC-bundled Python, like everything else:

    powershell -File scripts\run_local.ps1 cases\inclined_hole_toe.py

## The inclined-hole chain

Three cases, run in this order. Each answers one question, and the third's central
claim only holds *because* the second exists as a control — keep all three.

| case | source geometry | porosity field | what it establishes |
|---|---|---|---|
| `inclined_hole_grouting.py` | whole 17 m line | real `particles.csv` | the model chain is self-consistent |
| `inclined_hole_gradient.py` | whole 17 m line | analytic gradient | the gradient flips the gravity bias |
| `inclined_hole_toe.py` | toe segment only | analytic gradient | the flip is physics, not source geometry |

Shared setup: domain x[-30,30] y[0,60], 2.5 m cells (the REV floor for this pack).
Hole mouth (-30, 21) on the left boundary, dip 35 deg, length 17 m, toe
(-16.07, 30.75). The Phi90 mm bore is far below mesh resolution and is
represented as constant-pressure cells — acceptable because L_max = p0/lambda
barely depends on r0. Slurry parameters are the **v0.6 set**
(`variables.grout_material_v06`): tau0=30 Pa, mu_p=0.15 Pa.s, rho=1830 kg/m^3,
p0=5 MPa constant, gravity on. Permeability uses the `calibrated_power` law
k = 1.25e-7 * phi^3/(1-phi)^2, **not** Kozeny-Carman: KC overestimates the
flow-controlling permeability of decimetre waste rock by ~3 orders, which makes
the yield stall dynamically inactive. Saturation transport is enabled
case-locally; the engineering-case defaults in `src/` are untouched.

Local anchors under this law (v0.6): phi=0.12 -> lambda=4.40e5 Pa/m,
L_max=11.36 m; phi=0.18 -> 2.73e5, 18.29 m; phi=0.30 -> 1.40e5, 35.71 m. So a
0.12->0.30 porosity span is a 24.7x span in k and a 3.14x span in local L_max —
both ratios are parameter-set invariants, only the absolute scale moved.

**The domain was enlarged with the parameter set** (30x30 -> 60x60 m): v0.6
roughly doubles every L_max, and the old box could not hold the plume. The mouth
now sits ON the left boundary, which changes what that boundary means: it is the
exposed face the hole is collared on, not a far-field wall. A real face would
bleed; a no-flow face traps grout that reaches it, so every quota-consumption
time quoted below is **conservative** (see "Known limits" of each case).

## inclined_hole_grouting.py — baseline on the real pack

Reads `data/particles.csv` (36127 balls, vendored into the repo so the case is
self-contained; the original export lives at
`D:/PFC item/pfc_fipy_model1/particles.csv`, and `PARTICLES_CSV` overrides the
path). Bins it to areal porosity: **mean 0.1839, std 0.0136, range
0.1438–0.2118** over the 144 cells the pack covers — a near-homogeneous random
pack with no zoning, so no heterogeneity signature is expected or seen. The whole
17 m line is a 5 MPa source.

**The pack does not fill the enlarged domain.** It was exported on a 30x30 m
block, which is 144 of the 576 cells; the block is translated with the mouth so
hole and pack keep exactly the relationship the original case had, and the
remaining 432 cells are padded with the pack's own mean (0.1839). Padding adds no
structure, but it is not measured ground: of the 140 reachable cells, **100 (71%)
lie in the real pack and 40 (29%) in the padding**. Under v0.6 the plume outgrows
the block that was measured — read this case as "the chain is self-consistent on
real binned porosity", not as a prediction for 30 m of surveyed pile.

**Result: spread 19.36 / 17.92 m against an analytic L_max = p0/lambda = 18.79 m
at the actual phi = 0.184.** That agreement is the point of this case: it shows
the chain calibrated k -> per-cell lambda -> FiPy solve -> stall is
self-consistent. Stall at step 2298 (t=8119 s), 167 filled cells, conservation
drift 1.1e-3 relative. Unlike the previous parameter set, A is anchored
geometrically (d50^2/180) and never saw the case mine's 12 m design radius, so
this agreement is a check on the chain rather than on a fitted constant.

## inclined_hole_gradient.py — the gradient flips the gravity bias

Replaces the binned porosity with an analytic dense-bottom / loose-top gradient
phi(y) = phi_bottom + (phi_top - phi_bottom) * y/H, against a uniform phi=0.18
control. Source is still the whole line.

| | gradient 0.12->0.30 | uniform 0.18 |
|---|---|---|
| spread up (loose) / down (dense) | 22.63 m / 19.56 m | 16.48 m / 18.13 m |
| **up/down (filled)** | **1.16** | **0.91** |
| **up/down (analytic reach)** | **1.175** (1.179 at 1.25 m) | **0.888** (0.897) |
| filled cells / stall step | 199 / 2177 (t=6550 s) | 145 / 1481 (t=4320 s) |
| conservation drift (rel) | 9.1e-4 | 1.2e-3 |

The uniform pack sags downward (0.91 — gravity wins). Adding the gradient turns it
into a 1.16 upward bias: **the slurry would rather climb against gravity into the
loose, low-lambda region.** Gradient-strength sweep (phi_bottom=0.12 fixed):
phi_top 0.25/0.30/0.35 -> up/down 1.11/1.16/1.24, monotone.

**Two independent routes agree, and that agreement is the result to quote.**
The reachable domain is a pure path integral (`fill_diagnostics.reachable_domain`:
no solver, no time stepping). It gives 1.175 for the gradient field and 0.888
for the uniform one; the solver's filled front gives 1.16 and 0.91. Two methods
that share nothing but the porosity field land within about 0.02 of each other.

*Correction, 2026-09-24.* An earlier version of this section quoted the analytic
ratio as 1.26 (and 1.28 under the previous parameter set) and explained the gap
to the solver's 1.16 as "yield-edge creep diluting the asymmetry". Both the
number and the explanation were wrong. The reachable domain was then a shortest
path on the four face neighbours, which measures |dx|+|dy| and under-states
reach by 41% on diagonals -- a bias that does not shrink with refinement and
that inflated the ratio. With a 16-neighbour stencil (worst metric error 2.7%)
the ratio converges to 1.175/1.179 at 2.5/1.25 m, and the gap to the solver
closes. The direction of the result survives every stencil and both meshes;
only its magnitude came down.

The asymmetry is far milder than the local-L_max ratio (3.14) because spread is
a **path integral** of yield resistance along the way, not the endpoint local
lambda. This matters for reading every heterogeneous result that follows.

## inclined_hole_toe.py — a deliberate source-geometry control

**This case's source geometry is not the field process.** It restricts injection
to along-hole [12, 17] m (`BLEED_LEN`, tunable) and lets the first 12 m take part
in the seepage without injecting. Standard practice is a perforated grouting pipe
bleeding along its whole length, so the line-source cases above are the
engineering baseline; the orifice pipe seals only the collar (the reference
project used Phi108x1500 mm in a 2.0 m mouth), not twelve metres of hole.

An earlier version of this file claimed the toe bleed was "the physically correct
process". It is not, and the case does not need that claim. Its job is to be a
CONTROL: the upward asymmetry of the line source could in principle be an
artifact of injecting along a dipping line. Restricting the source to a compact
toe patch is about as different a source geometry as this geometry allows, and it
still comes out above 1 against a uniform control below 1. The less this case
resembles the field, the better it does that job — so keep it, and read it as a
control rather than as a process model.

Anchors below are for the v0.6 parameter set on the 60x60 m domain, scipy backend
with its PCG solver (`FIPY_SOLVERS=scipy` in `run_local.ps1`,
`linear_solver="pcg"`). The solver-sensitivity note at the end of this file
records what the previous parameter set did under PySparse, and why that made the
5 m point unreliable; under v0.6 the sweep is smooth and the issue does not recur.

| | gradient phi 0.12->0.30 | uniform phi=0.18 |
|---|---|---|
| spread up (loose) / down (dense) | 19.97 m / 17.51 m | 15.87 m / 16.08 m |
| **up/down** | **1.14** | **0.99** |
| filled cells / stall step | 159 / 1367 (t=3284 s) | 125 / 1235 (t=3516 s) |
| max radius from toe | 21.21 m | 19.81 m |
| conservation drift (rel) | 8.3e-4 | 1.1e-3 |

The spread collapses from a strip hugging the whole hole into a compact ellipse
around the toe. **The key result: the gradient turns a 0.99 downward bias into an
upward one with a completely different source geometry, which rules out source
geometry as the cause and leaves the porosity gradient as the physics.** The
uniform control landing on 0.99 rather than the line source's 0.91 is itself
informative: a compact source has no dipping line to bias it, so gravity and the
uniform yield threshold very nearly cancel. Magnitude here (1.14) is close to
the line source's (1.16 filled, 1.175 analytic).

**Under v0.6 this control discriminates less than it used to.** L_max (18-35 m)
now far exceeds the 17 m hole, so the reachable envelope is set by the far field
and barely depends on where along the hole the grout leaves: the toe-only reach
is 229 cells against 232 for the full line, with the same analytic up/down
(1.175). The solver fills the two differently (159 vs 199 cells), so the control
still says the flip is not an artefact of injecting along a dipping line; but
it no longer compares two genuinely different reach geometries.

Bleed-length sweep (gradient field): 3/5/8 m gives toe radius 20.02/21.21/23.30 m
and up/down 1.14/1.14/1.16 — the bleed length sets the ellipse SIZE, the porosity
gradient sets its ASYMMETRY. The two are decoupled, and under v0.6 the sweep is
monotone with no outlier.

### Known limits

At 2.5 m cells a 17-22 m spread spans 7-9 cells, so up/down ratios still carry
quantization noise: **treat these ratios as +-0.1, not precise numbers**, and
note that refining the mesh is safe: the "creep grows with refinement" finding
recorded earlier was the reach metric's own bias showing up on finer cells, not
solver creep (see CLAUDE.md, the metric correction under "充填诊断"). The
gradient field is
analytic, not a real PFC structure — it demonstrates the mechanism, it does not
reproduce a measured pile.

Under v0.6 the toe plume reaches the collar face (fill bbox starts at the leftmost
cell column, x=-28.8). That face is a no-flow boundary, so it confines the plume
instead of bleeding it. This is a property of the geometry — the hole is collared
on the face — not of the box size, and it cannot be removed by enlarging the
domain to the left, which would mean inventing rock behind the face.

## inclined_hole_channel.py — runaway and bypass voids from through-going structure

Same hole, parameters, mesh and stall rule as `inclined_hole_gradient.py`
(it borrows that file's `build_case` and swaps only the porosity field), with
fields that contain a connected high-porosity feature from
`src/structure/porosity_fields.py`: a vertical through-going band (the
backfill/host-rock contact, or a persistent gap), and anisotropic correlated
random fields. Diagnostics from `src/analysis/fill_diagnostics.py`.

**Stop rule is stall OR volume quota.** Run to full stall and the Bingham
model fills the reachable domain out to its rim -- in the continuum every
reachable cell; numerically about 83-86%, the rest a front shortfall still
connected to open ground -- so a sealed domain cannot show an enclosed void
that way. In the field the injection ends when the hole stops taking grout OR the
budgeted volume is spent — and runaway is the second ending arriving first.
Every structured run therefore gets the volume the uniform baseline needed to
stall (166.15 m³/m) and stops at min(stall, quota). The design target is the
baseline's own stalled footprint (155 cells).

    powershell -File scripts\run_local.ps1 cases\inclined_hole_channel.py --set band|position|random|all --plot

Band at φ=0.45, one cell wide, full height, swept over POSITION:

| band position | stop | t [s] | Q tail/peak | r_obs / r_equiv | target filled | reachable-unfilled |
|---|---|---|---|---|---|---|
| none (baseline) | stall | 4319 | 0.017 | 1.10 | — | 32 |
| past the toe (x=−11.25) | quota | 1736 | 0.084 | 1.91 | 76% | 122 |
| crossing the hole mid-length (x=−23.75) | quota | **434** | 0.059 | 1.94 | 69% | 111 |
| **at the collar (x=−28.75)** | quota | **554** | 0.045 | **2.24** | **67%** | 97 |

*Diagnostic columns re-computed 2026-09-24 with the corrected 16-neighbour
reach (see the metric note under the gradient case); every solver column --
stop, time, volume, fill count, Q, spread, wall -- reproduced the 09-22 runs
bit for bit, which doubles as the second independent run of these anchors.*

A band the hole ENTERS THROUGH — which is what a backfill/rock contact is —
drains the quota 8–10× faster than the baseline, spreads the grout to 2.2× what
that volume could fill uniformly, and leaves the hole's intended zone a third
short. That is the reference project's runaway mechanism ("充填体与原岩的交界面…
贯通空隙") reproduced from structure alone, with no open boundary: the grout has
nowhere to escape and it still starves the target.

**The whole of that missing third is ground the grout could have reached.** With
the corrected reach every band run has a design shortfall of 0%: the target is
short because the quota went down the band, not because 5 MPa could not get
there. Runaway here is a pure fill defect. (The old 4-neighbour reach booked
3-6% of it as "unreachable".)

Position still dominates: at fixed φ=0.45 it spans 434–1736 s (4×), while at
fixed position the φ sweep spans 1521–2469 s (1.6×). But **the "band porosity
barely matters" reading of the previous parameter set does not survive v0.6.**
That sweep (φ 0.30/0.45/0.60 past the toe) was 649/555/582 s and 94/90/92%
filled — flat and non-monotone. Under v0.6 it is 2469/1736/1521 s and
90/75/66% — monotone in both. Report it as "position matters several times more
than porosity", not as "porosity does not matter".

**The Q(t) criterion discriminates less well under v0.6.** The collar band's tail
ratio is 0.045, below the 0.05 stall line, even though it burned the quota in an
eighth of the baseline's time. With a lower yield stress the rate decays further
within the quota, so timing and spread ratio carry the runaway signal and Q(t)
alone would miss this case. In the field Q(t) is read against a pumping record,
not a stall threshold, so this is a limit of the automated flag, not of the
diagnostic.

No bypass voids in any band run: an unfilled reachable region stays connected
to open ground in these fields. Enclosed voids need structure that wraps —
see the random realisations.

Random correlated fields (mean 0.18, std 0.06, correlation 7.5 m × 2.5 m —
horizontal streaks), one per seed, same quota and target:

| seed | stop | t [s] | V_in | Q tail/peak | target filled | of which UNREACHABLE | reachable-unfilled |
|---|---|---|---|---|---|---|---|
| 1 | quota | 1457 | 166.2 | 0.014 | 77% | **15%** | 35 |
| 2 | **stall** | 1427 | **121.8** | 0.080 | 62% | 1% | **121** |
| 3 | quota | 1962 | 166.3 | 0.009 | 79% | **10%** | 60 |

Still no bypass voids, and the realisations still separate the two ways a target
can go unfilled -- but in mixed proportions, not the clean split recorded before.
Seed 1 is short 23 points of its target, 15 of them ground the path integral says
is **unreachable** at 5 MPa (a **design shortfall**: pressure or spacing) and the
rest reachable but not reached. Seed 3 is short 21 points, about half each. Only
the reachable domain can make that split, which is the point of the case.

*Correction, 2026-09-24:* this paragraph used to say the shortfall of seeds 1
and 3 was "overwhelmingly" unreachable (19% and 14%). That leaned on the old
4-neighbour reach, which under-stated what 5 MPa can reach.

**Seed 2 is stop-rule sensitive and should not be quoted as a result yet.** It
reports "stall" with a tail ratio of 0.080 -- above the 0.05 stall line -- and
leaves 121 of its 223 reachable cells (54%) unfilled, against 17% for the uniform
baseline's undisputed stall. (An earlier version called this self-contradictory
because "a stalled Bingham run fills every reachable cell"; with the corrected
reach even true stalls leave a 14-17% rim, so the argument rests on the rate and
on the size of the gap, not on the gap existing.) The stall detector is "60 consecutive steps with
no newly filled cell" (`STALL_PATIENCE`), and under v0.6 the front can take
longer than that to cross a cell while still advancing. Pending a re-run with a
larger patience, read seed 2 as a detector artifact, not as tight ground.

**Bypass voids did not appear in any single-stage run** — nine structured
fields, none. At this resolution and structure strength a single injection
leaves shortfalls that stay connected to open ground, not pockets sealed in
by grout. Enclosed voids need structure that wraps around a tight patch, or
the staged sequence: a pocket left unfilled by one stage, then walled off by
that stage's cement from the next stage's source (`stage_update.py`). That
is the 4.6 mechanism and the next case.

## inclined_hole_staged.py — staged advancing grouting versus a single pass

Same hole, parameters, mesh and per-stage stop rule as `inclined_hole_channel.py`.
The hole is drilled in passes; at stage k the perforated pipe bleeds along the
whole hole so far, [0, DEPTH_k]; when the stage stops the grout sets
(`stage_update.apply_stage_closure`) and the next pass drills deeper. Single
and staged runs inject the SAME total budget (the uniform baseline's stalled
166.15 m³/m), unused quota carried forward. Cumulative fill (what set,
`(n₀−n_final)/n₀`) is classified against the reachable domain of the VIRGIN
field from the FULL hole — what one pass could have reached.

    powershell -File scripts\run_local.ps1 cases\inclined_hole_staged.py --field base,collar,random --plot

| field / sequence | V used | t [s] | cells | of virgin reach | permanent residual |
|---|---|---|---|---|---|
| uniform / single | 166.15 | 4319 | 145 | **83%** | 32 |
| uniform / 7 → 10 → 17 m | **106.15** (stalled, 60 unspent) | 4046 | 87 | **52%** | 90 |
| uniform / 8.5 → 17 m | 106.15 | 3741 | 87 | 52% | 90 |
| collar band / single | 166.36 | 981 | 116 | 49% | 134 |
| collar band / 7 → 10 → 17 m | 130.36 | 4363 | 85 | **37%** | 165 |
| random seed 1 / single | 166.24 | 1457 | 131 | 80% | 35 |
| random seed 1 / 7 → 10 → 17 m | 166.19 | 2137 | 121 | 74% | 45 |

*The two right-hand columns were re-computed 2026-09-24 against the corrected
16-neighbour reach; all solver quantities, stage by stage, reproduced the 09-22
runs bit for bit. The true reachable zone is larger than the old reach said, so
every "of virgin reach" share dropped -- even a single pass run to stall fills
only 83% of it -- while every comparison between rows kept its sign and size.*

Per stage, the reference sequence 7 → 10 → 17 m on uniform ground:

| pass | source cells | of which buried in earlier cement | stop | V | newly filled |
|---|---|---|---|---|---|
| 1, to 7 m | 5 | 0 | stall 3440 s | 106.15 | 92 |
| **2, to 10 m** | 6 | **6** | stall 301 s | **0.00** | **0** |
| **3, to 17 m** | 10 | **10** | stall 305 s | **0.00** | **0** |

**Under v0.6 the whole sequence after the first pass is dead — on all three
fields.** L_max at phi=0.18 is 18.3 m, longer than the 17 m hole, so the first
pass to 7 m cements the ground along every metre the hole will ever occupy:
stage 2 finds 6 of 6 source cells buried, stage 3 finds 10 of 10, and both
inject 0.00 m³ while the crew pumps for 300 s into nothing. The previous
parameter set (L_max 7.9 m) buried only the 3 m increment and let the 8.5 m
advance still deliver; the same case now says a 17 m hole should not be staged
at all with this grout. The reference project's 7 → 10 → 17.5 m sequence has
exactly the increment that cannot work.

**Staging costs reach on uniform ground, and v0.6 makes the cost much larger.**
52% of the reachable zone instead of 83%, with 60 m³/m of the budget it cannot
place -- a loss of 31 points (the previous set lost 12). The 90 unfilled cells
are a **permanent residual** — after a full-depth sequence the hole is cemented
all round, so no further pass from it can reach them (the `shadow` column,
reachable in the virgin field and unreachable from the last pass in the cemented
field, equals the shortfall for every full-depth run).

**Staging cannot fix a collar runaway.** The interface band sits at the mouth,
so it is in every pass's source: the first pass stalls at 130 m³ and the later
passes get 0.00 m³. Staged fills 37% of the reachable zone against 49% for the
single pass. The remedy for a through-going contact is to seal it (the reference
project's double-fluid collar/bottom sealing), not to stage the injection.

**On the random field staging costs a little** (74% against 80%): the first pass
spends the whole quota before the deeper passes exist, so the sequence is
effectively a single shorter-hole pass.

**No enclosed bypass voids in any run, single or staged.** The sequence's
residual is a rim shadow, connected to open ground, not a sealed-in pocket.
Eighteen structured runs across the two cases under v0.6, on top of the
eighteen under the previous parameter set, and not one enclosed void. At 2.5 m
cells a fill footprint spans 7–9 cells and stays convex; an enclosed void would
need finer resolution or structure that wraps. Write this as "did not appear",
never as "cannot occur".

### Solver sensitivity of the toe control — previous parameter set (2026-09-22)

Kept as a record of how the yield edge can make a result solver-dependent. All
numbers here are for the **previous** parameter set (tau0=60, 30x30 m domain),
not for the tables above.

The toe case's gradient result was **linear-solver dependent**: 1.22 under FiPy's
default PySparse PCG, **1.06** under `FIPY_SOLVERS=scipy` (its PCG at tolerance
1e-15, or its LU — identical), which stalled later (364 vs 220 steps) and crept
further on both sides. The uniform control was bit-identical either way (0.97).
The **line-source** gradient and uniform results — the engineering baseline —
were identical under both solvers (623 / 2100 s / 44 cells; 492–493 /
1535–1540 s). The lesson that generalises: where a result sits on the yield
edge, check it against the reachable domain, which no solver touches -- and
make sure the reachable domain uses a metric that approaches the Euclidean one
(the "9 overshoot" once listed here was measured against the old 4-neighbour
reach and is not a solver property).
