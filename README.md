# Exact 1-D Riemann Solver for the JWL Equation of State

A Python implementation of an exact one-dimensional Riemann solver for compressible flow using the Jones–Wilkins–Lee (JWL) equation of state.

## Features

- JWL thermodynamic functions
- Shock-wave relations and post-shock state calculations
- Rarefaction-wave integration using a fourth-order Runge–Kutta (RK4) method
- Iterative star-state solution
- Sampling of the solution profile
- Convergence-history and solution-profile exports
- Two configurable test cases: Shyue-type and Lee-type

## Repository structure

```text
JWL-Riemann-Solver/
├── jwl_riemann_solver.py   # Shared solver functions
├── run_case.py             # Shared runner and test-case parameters
├── requirements.txt        # Python dependencies
├── README.md
└── .gitignore
```

## Requirements

- Python 3.9 or newer recommended
- NumPy
- pandas
- Matplotlib

Install dependencies with:

```bash
python -m pip install -r requirements.txt
```

## Run a test case

Run the Shyue-type case:

```bash
python run_case.py shyue
```

Run the Lee-type case:

```bash
python run_case.py lee
```

An optional second argument selects the output directory:

```bash
python run_case.py shyue outputs/shyue
python run_case.py lee outputs/lee
```

The runner writes convergence data, star-state data, sampled solution profiles, and profile plots to the selected output directory. Generated CSV and PNG files are excluded from version control by `.gitignore` by default.

## Test-case parameters

The test-case configurations currently reproduce the input values in the author's original notebooks.

| Parameter | Shyue-type | Lee-type |
|---|---:|---:|
| JWL `A` | 8.545 | 632.1 |
| JWL `B` | 0.205 | -0.04472 |
| JWL `R1` | 4.6 | 11.3 |
| JWL `R2` | 1.35 | 1.13 |
| JWL `Gamma` | 0.25 | 0.8938 |
| Reference density `rho0` | 1.84 | 1.905 |
| Left density | 1.7 | 0.9525 |
| Left velocity | 0.0 | 0.0 |
| Left pressure | 10.0 | 1.0 |
| Right density | 1.0 | 3.810 |
| Right velocity | 0.0 | 0.0 |
| Right pressure | 0.5 | 2.0 |
| Output time | 12.0 | 20.0 |

## Current verification

Both cases ran successfully with the refactored shared solver in the development environment and reached the solver's convergence criterion. This verifies execution, but it is not by itself an independent validation against published reference data. Reference comparisons and numerical-analysis details should be documented once the relevant reference values and plots are checked.

## Notes

- The solver functions were extracted from the original notebooks with their numerical algorithms preserved in this initial refactor.
- This repository is an academic project. Review equations, units, and reference comparisons before relying on results for research or publication.
