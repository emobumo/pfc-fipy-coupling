# Problem Definition

## Project
Engineering-scale simulation of slurry grouting / filling in underground mine waste rock using PFC2D + FiPy coupling.

## Physical scenario

A grouting borehole is arranged at the top of the waste-rock pile in an underground mine.

Slurry is injected into the waste-rock mass through this borehole under a constant-pressure boundary condition.

The slurry then migrates, infiltrates, and redistributes within the waste-rock mass.

## Current inlet boundary (placeholder)

At the current stage, FiPy uses a simple engineering placeholder for borehole injection:

- represent the grouting borehole by a localized inlet zone near the top of the waste-rock pile
- define the inlet zone by `inlet_zone_center_x`, `inlet_core_width_x`, and `inlet_spread_width_x`
- apply a constant-pressure injection boundary through `inlet_pressure_core_value`
- use `inlet_pressure_spread_factor` to describe the spread-zone pressure level relative to the core injection pressure

This is intentionally a temporary engineering representation of top-borehole constant-pressure injection, used to keep coupling validation and early model development simple.
## Engineering meaning
This model is intended to represent slurry transport behavior in a waste-rock filling process at engineering scale, rather than a small laboratory-scale soil-column rainfall test.

## Modeling goal
The goal is to simulate:
- slurry migration path
- penetration depth
- spatial distribution in the waste-rock pile
- pressure / flux / filling-related field evolution

## Relation to reference case
The rainfall infiltration 2D reference case is used only as a coupling-workflow reference.
Its data-exchange logic is reused, but the physical meaning, constitutive model, material parameters, and boundary conditions will be changed for slurry transport in waste rock.

## Modeling note
The rainfall infiltration reference case uses a VG-model-based Richards-equation interpretation for unsaturated soil-water flow.
That interpretation is not assumed to be directly valid for slurry transport in underground waste rock.

In this project, the reference case is reused only for:
- coupling workflow
- data-exchange order
- mesh-field-to-particle mapping logic

It is not reused directly for:
- constitutive law
- hydraulic meaning of variables
- rainfall boundary interpretation

## Planned changes relative to the reference case
Compared with the rainfall infiltration reference case, the new project will gradually replace:
- rainfall boundary with slurry pouring / inflow boundary
- unsaturated soil hydraulic interpretation with slurry transport interpretation
- soil-column test geometry with engineering-scale waste-rock geometry
- reference material parameters with waste-rock and slurry parameters

## Expected outputs
Expected outputs may include:
- slurry pressure-like field
- velocity / flux field
- penetration depth
- local filling degree / saturation-like indicator
- deposition or clogging-related indicator if needed later

## Current strategy
At the current stage, the priority is:
1. preserve the original PFC2D + FiPy data-exchange logic from the reference case
2. reuse the coupling workflow
3. gradually replace the physical model and engineering geometry
4. avoid premature large-scale refactoring

## Fixed structural assumptions

For the current project stage, the following assumptions are fixed:

1. The waste-rock particle skeleton is rigid and does not deform under fluid action.
2. Porosity is transferred from PFC to FiPy only once at initialization.
3. Porosity is not updated during later flow steps.
4. The particle model is treated as fixed/specific, so no automatic particle-model adaptation is required.

## Modeling interpretation

Under these assumptions:
- geometric porosity is treated as constant during the flow simulation
- permeability / intrinsic mobility is initialized once from the initial structure
- later flow evolution is represented on the FiPy side only
- fluid results are written back to particle extra values for diagnostics / visualization only

## Current first-version constitutive interpretation

For the current baseline, slurry transport is interpreted as variable-saturation continuum transport in a generalized nonlinear Darcy-type framework.

The first-version baseline should conceptually account for Bingham-type yield behavior rather than treating slurry as an ordinary Newtonian fluid.

Clogging is not treated as a dominant mechanism in the current first-version baseline unless it is explicitly re-enabled later.
