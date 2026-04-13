# Coupling Contract

## General principle
PFC2D represents the particle-scale structure of the underground mine waste-rock pile.
FiPy represents the engineering-scale continuum field used to describe slurry transport.

## Core idea
The rainfall infiltration 2D reference case is not reused for its rainfall physics.
It is reused for its PFC2D-FiPy data-exchange workflow.

## What is preserved vs. what is replaced

### Preserved from the reference case
- PFC2D -> FiPy -> PFC2D exchange order
- continuum-field interpolation back to particles
- explicit coupling-cycle structure
- separation between reference workflow and new development

### Replaced for the new project
- VG-model interpretation
- Richards-equation soil-water meaning
- rainfall-infiltration boundary condition
- soil-column physical setting
- reference hydraulic parameters

## Coupling workflow
One coupling cycle is defined as:

1. Build or restore the waste-rock structure in PFC2D.
2. Extract required structural or geometric information from PFC2D.
3. Build or update the FiPy mesh and field variables.
4. Solve one or more FiPy continuum transport steps.
5. Map FiPy field values back to PFC2D particles.
6. Update particle-side state if required.
7. Continue to the next coupling cycle.

## PFC2D -> FiPy
Potential quantities exchanged from PFC2D to FiPy include:
- domain geometry
- waste-rock pile shape
- porosity or void-related information
- particle spatial distribution
- boundary location
- internal structural state if needed later

## FiPy -> PFC2D
Potential quantities exchanged from FiPy to PFC2D include:
- pressure-like field
- hydraulic head-like field if that form is still used
- flux / velocity field
- local slurry occupancy / filling indicator
- local deposition / clogging indicator if introduced later

## Current placeholder pouring boundary
For the current validated coupling stage, the FiPy inlet boundary is kept simple:
- top boundary inlet patch centered at `inlet_center_x`
- inlet half-width in x as `inlet_half_width_x`
- fixed inlet pressure value as `inlet_pressure_value`

This boundary is an engineering placeholder only and does not represent a final constitutive inflow model.

## Mapping rule
Continuum field variables computed on the FiPy mesh are mapped back to particles in PFC2D by interpolation.
The current reference implementation follows the rainfall infiltration case, where field values are interpolated and written back to particle extra variables.

## Execution rule
The exchange sequence should remain consistent with the reference case until the new slurry-transport framework is stable.
The physical interpretation of variables may change, but the data-exchange order should remain explicit and traceable.

## Development rule
Reference-case code under `reference_cases/` is for workflow reference only.
New maintainable implementation should be developed under `src/`.

## Near-term development target
The first development target is to reproduce the same exchange structure as the rainfall infiltration reference case for the new waste-rock slurry-filling scenario.

## Important note
At the current stage, priority is given to:
- preserving data-exchange logic
- keeping execution order clear
- separating reference cases from new development
- replacing physics gradually rather than rewriting everything at once

## Current fixed coupling assumptions

The current coupling workflow uses the following fixed assumptions:

- PFC provides structural information only at initialization.
- Porosity is read once and used to initialize FiPy permeability / mobility.
- No later coupling step updates porosity from PFC.
- The particle skeleton is fixed during flow and does not respond mechanically to fluid action.
- Flow-related results are mapped back to particle extra values only.
- No fluid-force feedback is applied to the waste-rock skeleton.
- The particle model is treated as fixed, so no automatic geometry adaptation is included.