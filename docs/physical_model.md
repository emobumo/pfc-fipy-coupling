# Physical Model and Current Assumptions

## 1. Problem statement

This project studies slurry pouring and migration through a waste-rock pile in an underground mine backfilling / grouting context.

The final objective is not to directly reproduce a rainfall infiltration case, but to build a dedicated slurry-transport model on a fixed waste-rock skeleton:

- PFC2D provides waste-rock structure and porosity-related information.
- FiPy solves the continuum-scale fluid field.
- Fluid results may be mapped back to particle extra variables for visualization and statistics.

## 2. Modeling level

The current model is formulated at the continuum scale of a porous medium.

This means:

- flow is not solved at the explicit pore-channel scale,
- the waste-rock pile is represented as an equivalent porous medium,
- pressure / filling-related field variables are solved on a FiPy mesh.

Therefore, the current approach is a Darcy-type continuum formulation, not a direct pore-scale Navier–Stokes CFD model.

## 3. Saturation state

The slurry transport process is treated as a variable-saturation process rather than a fully saturated seepage problem.

Reason:

- slurry is poured locally from the top,
- the waste-rock void space is not fully occupied by slurry at the beginning,
- slurry progressively occupies pore volume during migration.

So the model should not be interpreted as a classical fully saturated Darcy seepage model.

## 4. Governing conceptual framework

The current working framework is:

- generalized Darcy-type continuum transport,
- variable saturation / pore occupation during slurry migration,
- nonlinear constitutive behavior,
- possible Bingham-type yield effect in slurry rheology.

In other words, this project does not assume a simple linear saturated Darcy law with constant mobility.

Instead, the effective mobility may depend on:

- porosity,
- local filling / saturation state,
- slurry rheology,
- local driving gradient.

## 5. Current physical assumptions

### 5.1 Fixed solid skeleton

The waste-rock skeleton is fixed.

Fluid flow does not induce particle displacement or skeleton deformation in the current version.

### 5.2 One-time porosity transfer

Porosity information is transferred from PFC to FiPy only once during initialization.

Later flow steps do not update porosity dynamically.

### 5.3 Variable-saturation slurry migration

The slurry migration process is treated as a variable-saturation process.

The slurry progressively occupies the available pore space, rather than assuming that the entire domain is fully saturated from the start.

### 5.4 Bingham-type slurry behavior

The first version should consider the slurry as a Bingham-type material at the conceptual level.

This means the constitutive relation should allow yield-controlled flow behavior rather than treating slurry as ordinary Newtonian water.

### 5.5 Clogging neglected in the first version

Because the waste-rock structure is relatively coarse, clogging is not treated as a dominant mechanism in the current first version.

This simplification may be revisited later if simulation results or experiments indicate otherwise.

## 6. What is intentionally excluded for now

The current first version does not include:

- skeleton deformation caused by fluid,
- dynamic porosity update,
- clogging-feedback-controlled permeability reduction,
- explicit pore-scale flow resolution,
- direct adoption of the standard Richards–van Genuchten soil-water formulation.

## 7. Implications for code design

The code should follow these constraints:

- keep the PFC side focused on geometry / porosity export,
- keep the FiPy side focused on continuum field solving,
- avoid introducing assumptions specific to classical saturated water infiltration without explicit justification,
- preserve extension points for later upgrades such as clogging, dynamic porosity, or more advanced rheology.

## 8. Terminology used in the repository

- **porosity**: pore volume fraction provided by PFC at initialization
- **filling**: fraction of pore space occupied by slurry in a control volume
- **mobility**: effective flow capacity in the generalized Darcy-type model
- **pressure**: continuum driving variable used in FiPy solving
- **clogging**: currently disabled first-version mechanism, reserved for future extension

## 9. Status of this document

This document describes the current working assumptions of the repository.

It is a design baseline for implementation, not a final claim that the physical model is complete.

The assumptions may be revised after numerical tests, benchmark comparisons, or experimental validation.