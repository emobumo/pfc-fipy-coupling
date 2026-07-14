# src

This directory contains new maintainable development code for PFC2D + FiPy coupling.

## Structure
- `models/slurry_transport/`: the physics core — governing equations, rheology,
  saturation transport, and parameter sets (`equations.py`, `variables.py`)
- `pfc_adapter/`: PFC-side porosity read and field write-back (`itasca` is
  imported lazily, so the FiPy side runs and is tested without PFC)
- `fipy_adapter/`: FiPy mesh construction
- `coupling/`: porosity -> permeability mapping and the coupling driver

## Rule
Do not place raw reference-case code here.
Reference cases remain under `reference_cases/`.
Application cases (research runs built on this engine) live under `cases/`.
