# PFC2D + FiPy Coupling for Waste-Rock Slurry Transport

This repository develops a coupled PFC2D + FiPy workflow for engineering-scale slurry transport in a fixed waste-rock pile, with application to underground mine grouting/backfilling scenarios.

## Project goal

The target problem is local slurry pouring on the top of a waste-rock pile and the subsequent migration of slurry through the pile.

The final objective is **not** to directly copy a rainfall infiltration case. Instead, this project builds a dedicated slurry-transport model with the following role split:

- **PFC2D** provides waste-rock structure and porosity-related information.
- **FiPy** solves the continuum-scale fluid field.
- Fluid results may be written back to particle extra variables for visualization and statistics.

## Current first-version modeling assumptions

The current baseline model uses the following assumptions:

- The waste-rock skeleton is fixed and does not deform under fluid action.
- Porosity is transferred from PFC to FiPy only once at initialization.
- The process is treated as **variable-saturation slurry migration**, not as a fully saturated seepage problem.
- The continuum formulation is a **generalized nonlinear Darcy-type model** rather than a simple saturated linear Darcy law.
- Slurry rheology should account for **Bingham-type yield behavior** in the first version.
- **Clogging is neglected** in the current baseline because the waste-rock structure is assumed to be relatively coarse.
- The model does **not** directly adopt the standard Richards–van Genuchten soil-water formulation unless explicitly re-evaluated later.

## Repository intent

This repository prioritizes a **minimal, verifiable first version** before introducing more complex mechanisms such as clogging, dynamic porosity update, skeleton deformation, or more advanced rheology.

## Documentation

Detailed physical assumptions and modeling scope are described in:

- `docs/physical_model.md`

## Current design principles

Implementation work in this repository should follow these principles:

- keep the PFC side focused on geometry / porosity export,
- keep the FiPy side focused on continuum field solving,
- avoid silently switching back to classical saturated water-infiltration assumptions,
- preserve clean extension points for future upgrades.

## Status

This repository is under iterative development. The current model description is a working baseline for implementation and validation, not a final claim that the physical model is complete.