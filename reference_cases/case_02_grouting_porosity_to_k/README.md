# case_02_grouting_porosity_to_k

## Case role
This case is stored as a reference case for one specific mechanism:

**PFC local porosity / structure information -> FiPy cell permeability or mobility initialization**

It is not treated as the final solver architecture for the current waste-rock slurry-filling project.

## Why this case is useful
This case is useful because it shows how to:
- build a mapping between PFC local regions and FiPy cells
- obtain local porosity / structure-related information from PFC
- use that information to initialize a FiPy permeability / mobility-like field
- write FiPy flow-related results back to particle extra values

## What is intended to be reused
For the current main project, the intended reusable part is:

1. PFC local structure / porosity acquisition
2. cell-wise mapping from PFC to FiPy
3. one-time initialization of FiPy mobility / permeability
4. flow-result writeback to particle extra values

## What is NOT intended to be reused directly
The following parts are not intended to be copied directly into the current main project:

- dynamic fracture-driven permeability updating
- repeated in-loop structure-to-permeability updates
- direct fluid-force application to the waste-rock skeleton
- full loading sequence and mechanical response logic of the original case

## Current project interpretation
In the current waste-rock slurry-filling project:

- PFC provides the waste-rock geometry and initial porosity/structure information
- FiPy solves the continuum slurry-flow field
- porosity is transferred once at initialization
- permeability / mobility is initialized once
- the waste-rock skeleton is kept fixed during flow
- fluid results are written back to particle extra values only

## File structure
- `raw_word/`: original Word source
- `pfc/`: PFC-side case files
- `py/`: Python / FiPy-side case files
- `notes.md`: overall notes for this case
- `extracted_logic/porosity_to_k_notes.md`: extracted reusable mechanism notes

## Status
This case is kept as a mechanism reference only.
Main development should continue under `src/`, not inside this reference-case folder.