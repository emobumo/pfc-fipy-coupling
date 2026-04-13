# case_02_grouting_porosity_to_k - notes

## Case role in this repository
This case is stored as a mechanism reference, not as a production model to be copied directly.

Its main value is to show how PFC-side structural information can be transferred to FiPy-side cell properties, especially:
- local cell/group mapping between PFC and FiPy
- porosity-related information acquisition on the PFC side
- updating FiPy permeability / mobility-like fields from PFC-side structure
- writing FiPy results back to particle extra values

## What is reused from this case
For the current waste-rock slurry-filling project, the reusable part is:

1. establish a mapping between FiPy cells and PFC local regions / particle groups
2. obtain local porosity or structure-related information from PFC
3. convert that information into an initial FiPy permeability / mobility field
4. solve the FiPy continuum field
5. write flow-related results back to particle extra variables for visualization / diagnostics

## What is NOT reused directly
The following parts of this reference case are not adopted directly into the current main project:

- repeated permeability updates driven by fracture evolution
- dynamic fracture-based permeability increase logic
- direct fluid-force application back onto the particle skeleton
- coupled mechanical response of the waste-rock skeleton to fluid loading
- using this case as the final physical model of slurry transport

## Important project-specific interpretation
In the current main project, PFC is used mainly to provide the waste-rock geometry and an initial porosity-based structural field.

The intended workflow is:
- transfer porosity / structure information from PFC to FiPy once at initialization
- build an initial permeability / mobility field in FiPy
- keep the waste-rock skeleton fixed afterward
- do not update porosity continuously during flow
- do not let fluid forces drive particle motion
- only map flow-related quantities back to particle extra values

## Practical takeaway
This case should be treated as a reference for:
- porosity-to-permeability initialization logic
- PFC-to-FiPy structural data transfer
- FiPy-to-PFC result writeback style

It should not be treated as the direct final solver architecture for the current project.