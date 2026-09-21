# Application cases

Engineering application cases built on the verified solver engine. These are
**not** verification-ladder tests (`tests/test_verification_ladder.py`): they do
not assert, they produce research results. Code is tracked here; the figures and
logs they generate go to `outputs/`, which stays untracked.

The numbers quoted below are **regression anchors**: the cases are deterministic
and a rerun reproduces them bit-for-bit (identical figures, byte for byte). If a
number here moves, the engine changed.

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
still gives ~1.2. The less this case resembles the field, the better it does that
job — so keep it, and read it as a control rather than as a process model.

| | gradient phi 0.12->0.30 | uniform phi=0.18 |
|---|---|---|
| spread up (loose) / down (dense) | 6.86 m / 5.63 m | 6.86 m / 7.07 m |
| **up/down** | **1.22** | **0.97** |
| filled cells / stall step | 26 / 220 (t=406 s) | 31 / 278 (t=637 s) |
| max radius from toe | 10.37 m | 11.13 m |
| conservation drift (rel) | 8.0e-4 | 1.1e-3 |

The spread collapses from a strip hugging the whole hole into a compact ellipse
around the toe, well clear of the left and bottom boundaries. **The key result:
pure-gradient asymmetry is 1.22 here, against 1.26 for the line source. Two
completely different source geometries give the same ~1.2 upward bias, which rules
out source geometry as the cause and leaves the porosity gradient as the physics.**

Bleed-length sweep (gradient field): 3/5/8 m gives toe radius 8.90/10.37/13.45 m
but up/down 1.06/1.22/1.17 — the bleed length sets the ellipse SIZE, the porosity
gradient sets its ASYMMETRY. The two are decoupled.

### Known limits

At 2.5 m cells an 8-10 m spread spans only 3-4 cells, so up/down ratios carry
quantization noise: **treat 1.22 as +-0.1, not a precise number**. Refine to
1.0 / 0.5 m before publishing quantitative values. The gradient field is
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
stall (46.14 m³/m) and stops at min(stall, quota). The design target is the
baseline's own stalled footprint (50 cells).

    powershell -File scripts\run_local.ps1 cases\inclined_hole_channel.py --set band|position|random|all --plot

Band at φ=0.45, one cell wide, full height, swept over POSITION:

| band position | stop | t [s] | Q tail/peak | r_obs / r_equiv | target filled | reachable-unfilled |
|---|---|---|---|---|---|---|
| none (baseline) | stall | 1535 | 0.009 | 1.34 | — | 1 |
| past the toe (x=+3.75) | quota | 556 | 0.064 | 1.26 | 90% | 11 |
| crossing the hole mid-length (x=−8.75) | quota | **96** | 0.074 | 2.50 | 70% | 27 |
| **at the collar (x=−13.75)** | quota | **119** | **0.101** | **2.92** | **70%** | 21 |

Position matters far more than band porosity (the φ sweep at the toe,
0.30/0.45/0.60, gives 649/556/582 s and 94/90/92% — nearly flat). A band the
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
| 2 | **stall** | 1624 | **39.4** | 0.015 | 80% | 20% | 0 |
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
