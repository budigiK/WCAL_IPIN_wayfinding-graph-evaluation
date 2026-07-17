# Reproducibility Package — IPIN 2026 WCAL Submission

This repository contains the full non-proprietary artifacts required to reproduce the experiments and figures reported in:

**Are Automatically-Generated Indoor Wayfinding Graphs Routable? A Reproducible No-Reference Evaluation Framework**
Harita Kanuri, Ramamurthy Rallabandi — IPIN 2026 WCAL Workshop, Rome, October 5–8, 2026.

## Contents

- `synthetic_layout_generator.py` — synthetic indoor layout generator and the two graph-construction method classes (Method A: corridor-template; Method B: centroid-proximity).
- `m1_to_m4_evaluation.py` — evaluation harness implementing metrics M1–M4 and writing the full CSV results table.
- `make_plots.py` — generates the four-panel aggregate figure from the results CSV.
- `m1_to_m4_results.csv` — reference output used in the manuscript.
- `m1_to_m4_results_plots.png` — aggregate plot panel used in the paper.

## What the code does

The release evaluates two independently plausible graph-construction strategies over the same synthetic layouts:

- **Method A** — corridor-template construction (strong structural prior).
- **Method B** — centroid-proximity construction (local proximity heuristic).

The harness computes four metrics:

- **M1** — Entrance Reachability Ratio.
- **M2** — Reachability Agreement between methods.
- **M3** — Path-Length Consistency over jointly reachable zones (undefined when the jointly reachable set is empty).
- **M4** — Edge Soundness Validity Rate, reported as undefined when wall geometry is absent.

## Requirements

- Python 3.9+ (the generator and harness use only the standard library).
- `pandas` and `matplotlib` are required only for `make_plots.py`:

```
pip install pandas matplotlib
```

## How to run

Place the scripts in the same directory and run:

```
python m1_to_m4_evaluation.py
```

This regenerates the full results CSV (`m1_to_m4_results.csv`) and prints summary statistics to the console. To regenerate the figure:

```
python make_plots.py
```

## Expected output

The harness sweeps:

- `n_rows ∈ {2, 3, 4, 6}`
- `n_cols ∈ {2, 4, 6}`
- `with_walls ∈ {False, True}`
- Method A corridor rows = 3 (fixed)
- Method B `k_neighbors ∈ {2, 3, 4}`

This corresponds to **72 unique configurations**. The code includes three repeated seed executions per configuration, but the generator is deterministic for the tested parameter settings, so outputs are identical across repeats; these are retained as a determinism consistency check rather than treated as stochastic variation (the CSV therefore contains 216 rows — 72 unique configurations × 3 identical repeats).

Headline aggregates (matching Table 1 of the paper): under `with_walls=False`, both methods achieve M1 = 1.000 and M2 = 1.000 with M4 non-verifiable; under `with_walls=True`, mean M1 is 0.292 (Method A) vs. 0.095 (Method B), M2 falls to 0.676, and mean M4 is 0.471 (Method A) vs. 0.955 (Method B).

## Reproducibility scope

This repository reproduces the synthetic experiments, results table, and figures reported in the paper. The originating production system remains proprietary; the released generator, method abstractions, and evaluation harness are sufficient to reproduce all non-proprietary claims and all reported numbers in the manuscript.

## License

Released under the MIT License (see `LICENSE`).
