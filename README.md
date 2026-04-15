# Stochastic Manpower Allocation Benchmark

## Overview

This project is a practical benchmark for stochastic manpower planning in manufacturing. It compares two related multi-objective formulations, CCGC and CCCP, under the same synthetic operating conditions, then reports how they behave in terms of speed, iteration count, and solution similarity.

The intent is not to model a specific real factory with proprietary data. The intent is to create a transparent and repeatable stress test where the optimization structure is realistic, the data generation is deterministic, and the results are easy to interpret.

In short, this repository helps answer a useful question:

"When the problem gets larger and more constrained, do CCGC and CCCP still produce near-identical manpower plans, and what is the runtime cost of each formulation?"

## Why synthetic data is the right choice here

For benchmarking optimization formulations, synthetic data is an advantage, not a limitation.

- It guarantees reproducibility with a fixed random seed.
- It isolates solver behavior from external noise in historical operational logs.
- It lets us scale problem size and constraints deliberately.
- It provides a fair side-by-side environment where both models face the exact same inputs.

The notebook also exports the generated dataset to CSV so runs can be audited, shared, and reused.

## Optimization design at a glance

The benchmark includes three objective components:

1. Robust output objective.
2. Wage cost objective.
3. Robust idle-time objective.

Robust terms combine expectation and variance-based penalty to reflect uncertainty. Utopia values are computed first by solving three single-objective subproblems. Those utopia values are then reused by both multi-objective formulations to keep the comparison aligned.

## Project contents and file guide

### `stochastic_manpower_benchmark.ipynb`

Primary research notebook and the canonical benchmark workflow.

What it does:

- Generates synthetic data for a large manufacturing scenario.
- Defines objective terms and common feasibility constraints.
- Solves utopia subproblems with SLSQP.
- Solves CCGC and CCCP with the same parameter set.
- Prints benchmark tables (time, iterations, objective values).
- Visualizes runtime and allocation comparison.
- Persists synthetic data to CSV.

When to use it:

- You want full transparency into equations and solver setup.
- You want plots and step-by-step analysis.
- You are modifying the benchmark methodology.

### `optimization_web_app.py`

Interactive Streamlit frontend for quick experimentation.

What it does:

- Exposes model selection (`CCGC`, `CCCP`, or `Both`).
- Lets you choose scenario size (`N`), manpower total (`X`), and random seed.
- Executes the solver pipeline.
- Shows runtime, iterations, objective components, and first 10 allocations.
- Performs feasibility validation before solving.

When to use it:

- You want quick what-if testing without editing notebook cells.
- You want a simple user-facing interface.

### `requirements.txt`

Runtime dependency list for both the notebook logic and Streamlit app.

### `synthetic_manpower_data.csv`

Generated synthetic dataset from the notebook run.

Typical columns include:

- work unit index,
- manpower lower and upper bounds,
- expected output and output variance,
- wage cost,
- expected idle time and idle variance.

### `s00170-012-4159-3.pdf`

Reference reading and modeling context used to ground formulation choices.

## Setup instructions (Windows, PowerShell)

From the project root:

```powershell
git clone 
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run optimization_web_app.py
```

Expected behavior:

- Streamlit serves locally (typically `http://localhost:8501`).
- You select model and input parameters.
- Clicking **Run Optimization** executes the selected formulation(s).
- Results table and allocation snippets are displayed in-page.

## Run the notebook benchmark

1. Open the notebook in VS Code or Jupyter.
2. Select the project virtual environment kernel.
3. Run cells top to bottom.

Expected outputs:

- synthetic dataset summary,
- CSV export path,
- utopia values,
- CCGC/CCCP benchmark comparison table,
- runtime chart,
- first-10 allocation comparison chart.

## Feasibility rules and common pitfalls

The per-unit manpower bounds are `[2, 12]`. For `N` units, feasible total manpower `X` must satisfy:

```text
2N <= X <= 12N
```

Examples:

- If `N = 300`, valid `X` range is `600` to `3600`.
- If `N = 500`, valid `X` range is `1000` to `6000`.

If `X` is outside the feasible range, optimization is rejected by validation logic.

## Reading benchmark results correctly

When comparing CCGC and CCCP, focus on two dimensions:

1. Computational overhead:
   - Runtime and iteration count indicate solver effort.
2. Practical solution equivalence:
   - Allocation delta and objective values indicate whether both formulations lead to materially similar plans.

A common outcome is that both models produce near-identical allocations while one formulation is computationally cheaper. That is exactly the type of tradeoff this benchmark is designed to expose.

## Suggested experimentation workflow

1. Start with a baseline run and save metrics.
2. Increase `N` while keeping feasibility of `X`.
3. Change random seed to test sensitivity.
4. Run `Both` model mode and compare allocation deltas.
5. Export or snapshot results for trend tracking.

## Reproducibility notes

- Keep the random seed fixed when comparing formulations.
- Keep weights (`w1`, `w2`, `w3`) unchanged between runs intended for comparison.
- Avoid mixing outputs from different dataset generations in the same analysis table.

## Troubleshooting

### Streamlit command not found

Use:

```powershell
python -m streamlit run optimization_web_app.py
```

### Solver appears slow at larger sizes

This is expected as dimensionality and active constraints increase. Compare both models under the same inputs before drawing conclusions.

### Notebook kernel mismatch

Make sure the notebook kernel points to the `.venv` interpreter used to install dependencies.

## License and usage

This repository is intended for benchmarking, research, and educational use. Adapt parameter ranges and constraint structures to fit your operational assumptions before using results for production planning.
