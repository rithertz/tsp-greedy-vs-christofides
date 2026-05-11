# Live Demo: Greedy vs Christofides

This folder contains a presentation-ready live demo for the TSP project report.

## What the demo shows
- The same Euclidean TSP instance on both sides
- Greedy construction step by step
- Christofides broken into MST, odd-degree matching, Euler walk, and final shortcut tour
- A live quality comparison using:
  - exact optimum when `n <= 12`
  - MST lower bound when `n > 12`
- The complexity story from the report:
  - Greedy is fast: `O(n^2)`
  - Christofides is slower but usually produces shorter tours: about `O(n^3)`

## How to run

From Windows Explorer:
- Double-click `run_live_demo.bat`

From a terminal in the project root:
```bash
python "algorithm executions/live_tsp_demo.py"
```

## Presenter flow
1. Start on `Overview` and explain that both algorithms see the same points.
2. Click `Next Stage` to move to `Greedy Build` and drag the `Greedy Progress` slider to show the local-choice behavior.
3. Move to `MST`, then `Odd Nodes + Matching`, then `Euler Tour` to explain the Christofides pipeline.
4. End on `Final Comparison` and read out the ratios shown at the bottom.
5. Use `New Random`, `New Clustered`, and the `Cities` slider to generate fresh examples live.

## Demo tips
- For a clean exact-optimum comparison, keep cities at `12` or below.
- For a more dramatic quality-vs-speed story, try clustered instances with `14` to `18` cities.
- The red star marks city `0`, which is used as the fixed starting city in both tours.
