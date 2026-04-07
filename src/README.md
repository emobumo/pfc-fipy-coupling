# src

This directory contains new maintainable development code for PFC2D + FiPy coupling.

## Structure
- `pfc_adapter/`: PFC-side state extraction and interaction
- `fipy_adapter/`: FiPy-side mesh, variables, and solver logic
- `coupling/`: variable exchange, timestep control, and coupling driver
- `utils/`: shared helper functions

## Rule
Do not place raw reference-case code here.
Reference cases remain under `reference_cases/`.