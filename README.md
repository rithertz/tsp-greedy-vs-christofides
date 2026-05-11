# TSP Heuristics Comparison

This project compares two approaches for the Euclidean Traveling Salesman
Problem (TSP):

- **Greedy nearest neighbor**: a fast `O(n^2)` heuristic.
- **Christofides**: a polynomial-time approximation algorithm with a `1.5`
  guarantee for metric TSP instances.

The experiment generates random and clustered point sets, runs both
algorithms, records tour length and runtime metrics, and produces CSV summaries,
plots, and a text report.

## Project Structure

```text
.
|-- main.py                         # Runs the full benchmark pipeline
|-- requirements.txt                # Python dependencies
|-- src/                            # TSP algorithms, generators, metrics, tests
|-- results/                        # Generated CSV benchmark results
|-- reports/                        # Generated report and plots
`-- algorithm executions/           # Live demo and C++ reference experiment files
```

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS or Linux, activate the environment with:

```bash
source .venv/bin/activate
```

## Run the Experiment

```bash
python main.py
```

The full benchmark runs 500 instances:

- city counts: `10`, `20`, `50`, `100`, `200`
- instance types: `random`, `clustered`
- repetitions per configuration: `50`

Outputs are written to:

- `results/raw_results.csv`
- `results/summary_stats.csv`
- `reports/final_report.txt`
- `reports/plots/*.png`

## Run Tests

```bash
python -m unittest discover -s src -v
```

## Live Demo

The `algorithm executions` folder contains a presentation-friendly visual demo.
Run it from the project root with:

```bash
python "algorithm executions/live_tsp_demo.py"
```

On Windows, you can also run:

```bash
"algorithm executions\run_live_demo.bat"
```

The demo shows the greedy construction, the Christofides pipeline, and a live
quality comparison against the exact optimum for small instances or the MST
lower bound for larger ones.

## Notes for GitHub

Generated Python caches, local editor settings, logs, virtual environments, and
compiled binaries are ignored. CSV results, plots, and the final report are kept
in the repository so the benchmark output is visible without rerunning the full
experiment.
