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

Reads `D:/PFC item/pfc_fipy_model1/particles.csv` (36127 balls; override with the
`PARTICLES_CSV` env var — the CSV lives outside the repo, so this case is the one
that is not self-contained). Bins it to areal porosity: mean 0.184, std 0.014 —
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

## inclined_hole_toe.py — inclined-hole grouting, toe-segment bleed

The real process cases the collar (orifice pipe + plug, as in the reference case),
so grout does **not** bleed from the whole hole — only from the deep toe section.
Geometry is unchanged from the two cases above, but the source is restricted to
along-hole [12, 17] m (`BLEED_LEN`, tunable); the collar side just participates in
the seepage without injecting. This is both the physically correct process and the
control that strips the line-source geometry out of the previous result.

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
