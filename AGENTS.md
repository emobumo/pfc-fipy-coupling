# Repository rules

## Scope
- This repository is for PFC2D + FiPy coupling development.
- Files under `reference_cases/` are reference-only unless explicitly asked to modify them.
- Prefer minimal diffs over broad refactors.
- Preserve original execution order for reference cases.

## Reference cases
- Treat Word-derived cases as archival references first, not production code.
- Do not rewrite reference cases into a new architecture unless explicitly requested.
- When analyzing a reference case, explain the execution chain before proposing changes.

## Development rules
- New maintainable code should go under `src/`.
- Do not mix new development code into `reference_cases/`.
- Separate PFC-side scripts, FiPy-side scripts, and coupling logic whenever possible.
- Keep variable mapping and time-stepping logic explicit.

## High-risk changes
Ask before changing:
- time stepping logic
- convergence criteria
- boundary conditions
- particle-mesh mapping rules
- unit conversion assumptions

## Validation
- After edits, explain what changed.
- Prefer checking one small runnable path before larger refactors.
- When something cannot be validated, state exactly what is missing.

## Notes for this repository
- The rainfall infiltration 2D case is currently a faithful split from Word.
- Python files in that case may share runtime context and should not be modularized prematurely.

- The particle skeleton is rigid; do not add fluid-driven particle deformation.
- Porosity is transferred only once at initialization; do not add later porosity updates.
- The particle model is fixed; do not add automatic particle-model adaptation unless explicitly requested.