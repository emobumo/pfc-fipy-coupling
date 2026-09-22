# Application cases

Engineering application cases built on the verified solver engine. These are
**not** verification-ladder tests (`tests/test_verification_ladder.py`): they do
not assert, they produce research results. Code is tracked here; the figures and
logs they generate go to `outputs/`, which stays untracked.

The numbers quoted below are **regression anchors**: the cases are deterministic
and a rerun reproduces them bit-for-bit (identical figures, byte for byte). If a
number here moves, the engine changed. Anchors are for the **scipy LU** linear
solver, the default `run_local.ps1` sets since 2026-09-22 (FiPy's own default,
PySparse PCG, crashed 3 of 5 long 576-cell runs); the line-source and real-pack
results are identical under both, the toe control is not — see its section.

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

Shared setup: domain x[-15,15] y[0,30], 2.5 m cells (the REV floor for this pack).
Hole mouth (-15, 1) on the left boundary near the bottom, dip 35 deg, length 17 m,
toe (-1.07, 10.75). The Phi90 mm bore is far below mesh resolution and is
represented as constant-pressure cells — acceptable because L_max = p0/lambda
barely depends on r0. Slurry (PO 42.5, w/c=0.5): tau0=60 Pa, mu_p=0.1 Pa.s,
rho=1820 kg/m^3, p0=5 MPa constant, gravity on. Permeability uses the
`calibrated_power` law k = 9.4e-8 * phi^3/(1-phi)^2, **not** Kozeny-Carman: KC
overestimates the flow-controlling permeability of decimetre waste rock by ~3
orders, which makes the yield stall dynamically inactive. Saturation transport is
enabled case-locally; the engineering-case defaults in `src/` are untouched.

Local anchors under this law (tau0=60): phi=0.12 -> lambda=1.02e6 Pa/m,
L_max=4.93 m; phi=0.18 -> 6.30e5, 7.93 m; phi=0.30 -> 3.23e5, 15.49 m. So a
0.12->0.30 porosity span is a 24.7x span in k and a 3.14x span in local L_max.

## inclined_hole_grouting.py — baseline on the real pack

Reads `data/particles.csv` (36127 balls, vendored into the repo so the case is
self-contained; the original export lives at
`D:/PFC item/pfc_fipy_model1/particles.csv`, and `PARTICLES_CSV` overrides the
path). Bins it to areal porosity: mean 0.184, std 0.014 —
a near-homogeneous random pack with no zoning, so no heterogeneity signature is
expected or seen. The whole 17 m line is a 5 MPa source.

**Result: spread 8.50 / 8.91 m against an analytic L_max = p0/lambda ~ 8.2 m at the
actual phi ~ 0.18.** That agreement is the point of this case: it shows the chain
calibrated k -> per-cell lambda -> FiPy solve -> stall is self-consistent. Note it
lands at 8 m, not the 12 m of the calibration anchor (phi=0.25) — the real pack is
denser, so lambda is larger and the spread shorter. Getting 12 m here would have
meant something was wrong.

## inclined_hole_gradient.py — the gradient flips the gravity bias

Replaces the binned porosity with an analytic dense-bottom / loose-top gradient
phi(y) = phi_bottom + (phi_top - phi_bottom) * y/30, against a uniform phi=0.18
control. Source is still the whole line.

| | gradient 0.12->0.30 | uniform 0.18 |
|---|---|---|
| spread up (loose) / down (dense) | 8.91 m / 7.07 m | 7.68 m / 8.50 m |
| **up/down** | **1.26** | **0.90** |
| conservation drift (rel) | 9.5e-4 | 1.3e-3 |

The uniform pack sags downward (0.90 — gravity wins). Adding the gradient turns it
into a 1.26 upward bias: **the slurry would rather climb against gravity into the
loose, low-lambda region.** Gradient-strength sweep (phi_bottom=0.12 fixed):
phi_top 0.25/0.30/0.35 -> up/down 1.22/1.26/1.35, monotone.

Asymmetry (1.26) is far milder than the local-L_max ratio (3.14) because spread is
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
CONTROL: the 1.26 upward asymmetry of the line source could in principle be an
artifact of injecting along a dipping line. Restricting the source to a compact
toe patch is about as different a source geometry as this geometry allows, and it
still comes out above 1 against a uniform control below 1. The less this case
resembles the field, the better it does that job — so keep it, and read it as a
control rather than as a process model.

Anchors below are for the scipy LU backend, the default since 2026-09-22
(`FIPY_SOLVERS=scipy` in `run_local.ps1`). Under the previous PySparse PCG
default the 5 m gradient run stalled earlier (step 220, t=406 s, 26 cells,
6.86 / 5.63 m, up/down 1.22); the 3 m and 8 m sweep points and the uniform
control were identical under both. See "Solver sensitivity" below.

| | gradient phi 0.12->0.30 | uniform phi=0.18 |
|---|---|---|
| spread up (loose) / down (dense) | 7.47 m / 7.07 m | 6.86 m / 7.07 m |
| **up/down** | **1.06** | **0.97** |
| filled cells / stall step | 35 / 364 (t=1006 s) | 31 / 278 (t=637 s) |
| max radius from toe | 11.13 m | 11.13 m |
| conservation drift (rel) | 8.1e-4 | 1.1e-3 |

The spread collapses from a strip hugging the whole hole into a compact ellipse
around the toe, well clear of the left and bottom boundaries. **The key result:
the gradient turns a 0.97 downward bias into an upward one with a completely
different source geometry, which rules out source geometry as the cause and
leaves the porosity gradient as the physics.** Its magnitude here (1.06–1.22
depending on solver, see below) is smaller than the line source's 1.26; the
solver-free number is the analytic reach ratio, 1.28.

Bleed-length sweep (gradient field): 3/5/8 m gives toe radius 8.90/11.13/13.45 m
and up/down 1.06/1.06/1.17 — the bleed length sets the ellipse SIZE, the porosity
gradient sets its ASYMMETRY. The two are decoupled.

### Known limits

At 2.5 m cells an 8-10 m spread spans only 3-4 cells, so up/down ratios carry
quantization noise: **treat these ratios as +-0.1, not precise numbers**, and
note that refining the mesh does NOT help under the current solver (creep grows
with step count; see CLAUDE.md "分辨率与求解器实测"). The gradient field is
analytic, not a real PFC structure — it demonstrates the mechanism, it does not
reproduce a measured pile.

## inclined_hole_channel.py — runaway and bypass voids from through-going structure

Same hole, parameters, mesh and stall rule as `inclined_hole_gradient.py`
(it borrows that file's `build_case` and swaps only the porosity field), with
fields that contain a connected high-porosity feature from
`src/structure/porosity_fields.py`: a vertical through-going band (the
backfill/host-rock contact, or a persistent gap), and anisotropic correlated
random fields. Diagnostics from `src/analysis/fill_diagnostics.py`.

**Stop rule is stall OR volume quota.** Run to full stall and the Bingham
model fills every reachable cell, so a sealed domain cannot show a void that
way. In the field the injection ends when the hole stops taking grout OR the
budgeted volume is spent — and runaway is the second ending arriving first.
Every structured run therefore gets the volume the uniform baseline needed to
stall (46.17 m³/m) and stops at min(stall, quota). The design target is the
baseline's own stalled footprint (50 cells).

    powershell -File scripts\run_local.ps1 cases\inclined_hole_channel.py --set band|position|random|all --plot

Band at φ=0.45, one cell wide, full height, swept over POSITION:

| band position | stop | t [s] | Q tail/peak | r_obs / r_equiv | target filled | reachable-unfilled |
|---|---|---|---|---|---|---|
| none (baseline) | stall | 1540 | 0.009 | 1.34 | — | 1 |
| past the toe (x=+3.75) | quota | 555 | 0.064 | 1.26 | 90% | 11 |
| crossing the hole mid-length (x=−8.75) | quota | **96** | 0.074 | 2.50 | 70% | 27 |
| **at the collar (x=−13.75)** | quota | **119** | **0.101** | **2.92** | **70%** | 21 |

Position matters far more than band porosity (the φ sweep at the toe,
0.30/0.45/0.60, gives 649/555/582 s and 94/90/92% — nearly flat). A band the
hole ENTERS THROUGH — which is what a backfill/rock contact is — drains the
quota 13× faster than the baseline, with the injection rate still an order of
magnitude above the stall line, spread ~3× what the volume could fill
uniformly, and the hole's intended zone 30% short. That is the reference
project's runaway mechanism ("充填体与原岩的交界面…贯通空隙") reproduced from
structure alone, with no open boundary: the grout has nowhere to escape and it
still starves the target.

No bypass voids in any band run: an unfilled reachable region stays connected
to open ground in these fields. Enclosed voids need structure that wraps —
see the random realisations.

Random correlated fields (mean 0.18, std 0.06, correlation 7.5 m × 2.5 m —
horizontal streaks), one per seed, same quota and target:

| seed | stop | t [s] | V_in | Q tail/peak | target filled | of which UNREACHABLE | reachable-unfilled |
|---|---|---|---|---|---|---|---|
| 1 | quota | 445 | 46.2 | **0.137** | 78% | 20% | 7 |
| 2 | **stall** | 1628 | **39.4** | 0.015 | 80% | 20% | 0 |
| 3 | quota | 411 | 46.2 | 0.043 | 86% | 12% | 6 |

Still no bypass voids — but the realisations separate the two ways a target
can go unfilled. Seed 1 hits a loose streak and drains the quota with the
rate still high (runaway signature), yet most of its 22% target shortfall is
ground the path integral says is unreachable at 5 MPa. Seed 2 is the mirror
image: tight ground, the hole stalls at 39 m³ before the quota is spent, and
the whole 20% shortfall is unreachable — a **design shortfall** (pressure or
spacing insufficient for this ground), with zero fill defect. The same
"80% filled" reads as runaway in one field and as under-pressure in the other,
and only the reachable domain tells them apart.

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
46.14 m³/m), unused quota carried forward. Cumulative fill (what set,
`(n₀−n_final)/n₀`) is classified against the reachable domain of the VIRGIN
field from the FULL hole — what one pass could have reached.

    powershell -File scripts\run_local.ps1 cases\inclined_hole_staged.py --field base,collar,random --plot

| field / sequence | V used | t [s] | cells | of virgin reach | permanent residual |
|---|---|---|---|---|---|
| uniform / single | 46.15 | 1535 | 40 | **98%** | 1 |
| uniform / 7 → 10 → 17 m | **41.66** (stalled, 4.5 unspent) | 1684 | 33 | **86%** | 7 |
| uniform / 8.5 → 17 m | 41.66 | 1379 | 33 | 86% | 7 |
| collar band / single | 46.68 | 119 | 30 | 66% | 21 |
| collar band / 7 → 10 → 17 m | 46.44 | 254 | 23 | **54%** | 28 |
| random seed 1 / single | 46.15 | 444 | 34 | 86% | 7 |
| random seed 1 / 7 → 10 → 17 m | 46.27 | 1431 | 34 | 86% | 7 |

Per stage, the reference sequence 7 → 10 → 17 m on uniform ground:

| pass | source cells | of which buried in earlier cement | stop | V | newly filled |
|---|---|---|---|---|---|
| 1, to 7 m | 5 | 0 | stall 683 s | 21.26 | 18 |
| **2, to 10 m** | 6 | **6** | stall 304 s | **0.00** | **0** |
| 3, to 17 m | 10 | 8 | stall 691 s | 20.37 | 18 |

**A drilling increment shorter than L_max is a wasted pass — measured in 2D on
all three fields.** The 3 m advance to 10 m puts every new source cell inside
the first pass's cemented zone (6 of 6 buried): the pass injects 0.00 m³,
Q collapses to 1e-11, and the crew pumps for 300 s into nothing. The reference
project's 7 → 10 → 17.5 m has exactly this increment. Removing that pass
(8.5 → 17 m) changes nothing — the two sequences are identical on this grid,
because 7.0 and 8.5 m select the same five 2.5 m source cells and the 10 m
pass delivered nothing anyway.

**Staging costs reach on uniform ground.** Two short sources each fill a
smaller stadium than the full hole does, and by the time the last pass runs,
the first pass's cement stands between its two fresh cells and the outer rim.
Same hole, same grout: 86% of the reachable zone instead of 98%, and the
sequence stalls with 4.5 m³ of the budget it cannot place. The seven lost rim
cells are a **permanent residual** — after a full-depth sequence the hole is
cemented all round, so no further pass from it can reach them (the `shadow`
column, reachable in the virgin field and unreachable from the last pass in
the cemented field, equals the shortfall for every full-depth run).

**Staging cannot fix a collar runaway.** The interface band sits at the mouth,
so it is in every pass's source: the first pass drains the whole quota in
254 s and the later passes get 0.00 and 0.07 m³. Staged fills 54% of the
reachable zone against 66% for the single pass. The remedy for a
through-going contact is to seal it (the reference project's double-fluid
collar/bottom sealing), not to stage the injection.

**On the random field staging is neutral** (86% either way): the first pass
stalls early in tight ground at 6.2 m³, the third hits the loose streak and
drains 40 m³ — the same volume ends up in the same streak, later.

**No enclosed bypass voids in any run, single or staged.** The sequence's
residual is a rim shadow, connected to open ground, not a sealed-in pocket.
At 2.5 m cells a fill footprint spans 3–4 cells and stays convex; an
enclosed void would need finer resolution or structure that wraps.

### Solver sensitivity of the toe control (2026-09-22)

The toe case's gradient result is **linear-solver dependent**: 1.22 under FiPy's
default PySparse PCG, **1.06** under `FIPY_SOLVERS=scipy` (LU), which stalls
later (364 vs 220 steps) and creeps further on both sides. The uniform control
is bit-identical either way (0.97). The **line-source** gradient and uniform
results — the engineering baseline and the 1.26 headline — are identical under
both solvers (623 / 2100 s / 44 cells / 9 overshoot; 492–493 / 1535–1540 s).
Read the toe control as "direction confirmed (>1 against 0.97), magnitude
between 1.06 and 1.22"; the analytic reach ratio, 1.28, is the solver-free
number. See CLAUDE.md "分辨率与求解器实测".
