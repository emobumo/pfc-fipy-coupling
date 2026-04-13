# Porosity-to-Permeability Notes

## Purpose
This note extracts only the reusable "PFC porosity / structure -> FiPy permeability" logic from case_02.

The goal is not to copy the full grouting case, but to isolate a mechanism that can be re-implemented in the main project.

## Core reusable mechanism

### Step 1: build FiPy cells and a PFC-side spatial mapping
The reference case creates FiPy cells and then generates corresponding PFC-side geometry / grouping information so that each FiPy cell can be associated with a local particle region.

This is the key prerequisite for transferring local PFC information into FiPy cell variables.

### Step 2: obtain local structural information from PFC
The reference case shows that local porosity can be queried from PFC measurement objects.

For the current main project, this idea is useful even if the exact implementation details later change.

The main reusable concept is:
- each FiPy cell should have access to one local porosity / void-ratio-like value derived from the waste-rock structure

### Step 3: convert porosity into permeability / mobility
The reference case suggests that a FiPy cell permeability field `kk` can be derived from local structure information.

For the current main project, this should be simplified into:

- read local porosity once from PFC at initialization
- compute a cell-wise permeability / mobility field once
- store it in FiPy as an initial structural property field

This means the current main project does NOT need:
- repeated porosity updates during flow
- repeated fracture-driven `kk` updates
- dynamic permeability recomputation every cycle

### Step 4: use the initialized permeability / mobility in FiPy
After the initial `kk` / mobility field is created, FiPy uses it in the placeholder slurry-transport solve.

For the current project, this fits the intended simplification:
- PFC provides static structure
- FiPy solves the continuum flow field
- PFC is not mechanically updated by the fluid

### Step 5: write flow results back to particles
The reference case also writes flow-related quantities back to particle extra values.

This part is directly relevant to the current project, because the current repository already uses particle extra values for:
- pressure-like field
- flux vector
- filling indicator
- clogging indicator

## Recommended interpretation for the current main project

### Reuse
Reuse the case only for:
- local porosity acquisition logic
- spatial mapping from PFC regions to FiPy cells
- initialization of cell-wise permeability / mobility
- writeback pattern from FiPy to particle extra values

### Do not reuse directly
Do not reuse directly:
- dynamic fracture-to-permeability logic
- repeated in-loop `kk` modification from evolving PFC structure
- fluid-force application to particles
- the full loading sequence of the original case

## Proposed implementation target in src/

A suitable main-project implementation path is:

- `src/pfc_adapter/porosity_reader.py`
  - read local porosity / structure information from PFC

- `src/coupling/porosity_to_permeability.py`
  - convert local porosity into cell permeability / mobility

- `src/models/slurry_transport/variables.py`
  - store the initialized mobility field

- `src/models/slurry_transport/equations.py`
  - use that mobility field in the FiPy solve without repeated structural updating

## One-time initialization rule
For the current project, the intended rule is:

1. build or restore the waste-rock pile in PFC
2. read local porosity once
3. initialize FiPy permeability / mobility once
4. run the slurry-flow solve
5. write fluid-related results back to particle extra values
6. do not update the waste-rock skeleton or porosity during subsequent flow steps

This is the key simplification relative to the original reference case.