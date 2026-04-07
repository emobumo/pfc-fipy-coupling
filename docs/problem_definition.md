# Problem Definition

## Project
Engineering-scale simulation of slurry grouting / filling in underground mine waste rock using PFC2D + FiPy coupling.

## Physical scenario
Slurry is poured onto the top surface of a waste-rock pile in an underground mine.
The slurry then migrates, infiltrates, and redistributes within the waste-rock mass.

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