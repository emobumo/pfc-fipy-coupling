# Application cases

Engineering application cases built on the verified solver engine. These are
**not** verification-ladder tests (`tests/test_verification_ladder.py`): they do
not assert, they produce research results. Code is tracked here; the figures and
logs they generate go to `outputs/`, which stays untracked.

Run with the PFC-bundled Python, like everything else:

    powershell -File scripts\run_local.ps1 cases\inclined_hole_toe.py

## inclined_hole_toe.py — inclined-hole grouting, toe-segment bleed

Constant-pressure Bingham grouting from an inclined borehole into a waste-rock
pile, comparing a gradient porosity field against a uniform control.

Setup: domain x[-15,15] y[0,30], 2.5 m cells (the REV floor for this pack).
Hole mouth (-15, 1) on the left boundary near the bottom, dip 35 deg, length
17 m, toe (-1.07, 10.75). The collar is cased (orifice pipe + plug, as in the
reference case), so only the toe segment bleeds: along-hole [12, 17] m
(`BLEED_LEN`, tunable). The Phi90 mm bore is far below mesh resolution and is
represented as constant-pressure cells — acceptable because L_max = p0/lambda
barely depends on r0.

Slurry (PO 42.5, w/c=0.5): tau0=60 Pa, mu_p=0.1 Pa.s, rho=1820 kg/m^3, p0=5 MPa
constant, gravity on. Permeability uses the `calibrated_power` law
k = 9.4e-8 * phi^3/(1-phi)^2, NOT Kozeny-Carman: KC overestimates the
flow-controlling permeability of decimetre waste rock by ~3 orders, which makes
the yield stall dynamically inactive. Saturation transport is enabled case-locally.

### Reference results (regression anchors)

Deterministic — a rerun reproduces these bit-for-bit. A drift here means the
engine changed.

| | gradient phi 0.12->0.30 | uniform phi=0.18 |
|---|---|---|
| spread up (loose) / down (dense) | 6.86 m / 5.63 m | 6.86 m / 7.07 m |
| **up/down** | **1.22** | **0.97** |
| filled cells / stall step | 26 / 220 (t=406 s) | 31 / 278 (t=637 s) |
| max radius from toe | 10.37 m | 11.13 m |
| conservation drift (rel) | 8.0e-4 | 1.1e-3 |

The finding: the porosity gradient **flips the gravity bias**. The uniform pack
sags slightly downward (0.97); adding a dense-bottom/loose-top gradient turns it
into a 1.22 upward bias — the slurry would rather climb against gravity into the
loose (low-lambda) region. The line-source predecessor gave 1.26 with a totally
different source geometry, which rules out source geometry as the cause.

Bleed-length sweep (gradient field): 3/5/8 m gives toe radius 8.90/10.37/13.45 m
but up/down 1.06/1.22/1.17 — the bleed length sets the ellipse SIZE, the porosity
gradient sets its ASYMMETRY. The two are decoupled.

### Known limits

At 2.5 m cells an 8-10 m spread spans only 3-4 cells, so up/down ratios carry
quantization noise: **treat 1.22 as +-0.1, not a precise number**. Refine to
1.0 / 0.5 m before publishing quantitative values. The gradient field is
analytic, not a real PFC structure — it demonstrates the mechanism, it does not
reproduce a measured pile.
