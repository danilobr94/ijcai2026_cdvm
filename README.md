# Constraint Data Value Maximization (CDVM)

Code for the IJCAI 2026 paper:

> **Constraint-Data-Value-Maximization: Utilizing Data Attribution for Effective Data Pruning in Low-Data Environments**
> Danilo Brajovic, David A. Kreplin, Marco F. Huber
> [arXiv:2605.11312](https://arxiv.org/abs/2605.11312)

```bibtex
@misc{brajovic2026constraintdatavaluemaximizationutilizingdataattribution,
      title={Constraint-Data-Value-Maximization: Utilizing Data Attribution for Effective Data Pruning in Low-Data Environments},
      author={Danilo Brajovic and David A. Kreplin and Marco F. Huber},
      year={2026},
      eprint={2605.11312},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2605.11312},
}
```

**Want to use CDVM?** Start with [`example_fashion_mnist.ipynb`](example_fashion_mnist.ipynb), it is self-contained and runs on Google Colab. [`example_synthetic.ipynb`](example_synthetic.ipynb) is similarly standalone. The rest of this repository reproduces the experiments and plots from the paper.

---

## Environment Setup

### Option 1: Docker (recommended)

```bash
docker build -t cdvm .
docker compose up -d
```

The container includes CUDA support, all Python dependencies, and JupyterLab.

### Option 2: Local Python environment

```bash
pip install -r requirements.txt
```

Key dependencies:

| Package | Version | Purpose |
|---|---|---|
| `opendataval` | 1.2.1 | Data valuation framework |
| `mlflow` | 2.15.1 | Experiment tracking |
| `cvxpy` | latest | Convex optimization (CDVM) |
| `gurobipy` | latest | LP solver (CDVM + pruning) |
| `jupyterlab` | latest | Notebooks |

### Gurobi license

CDVM and pruning optimization require a valid Gurobi license.
Place `gurobi.lic` in the runtime environment before running any experiment or notebook.
Academic licenses are available free of charge from [gurobi.com](https://www.gurobi.com/academia/academic-program-and-licenses/).

### MLflow tracking server

Start the local MLflow UI before running notebooks or scripts:

```bash
mlflow ui --host 0.0.0.0 --port 5000
```

The code reads from and writes to `mlruns/` in the repo root.

---

## Data

MLflow experiments (`mlruns/`, `mlartifacts/`) and intermediate CDVM artifacts
(`data/attr_datavals/`, `data/attr_selected_vals/`) are included in this repository.

Large precomputed files (attribution matrices ~7.8 GB, influence score matrices ~200 MB)
are hosted separately. Download them with:

```bash
python download_data.py
```

This fetches `data/attr_matr/` and the `influence_score_matrix-*.npy` files required
to run CDVM and pruning optimization without recomputing them from scratch.
See [`data/README.md`](data/README.md) for a full breakdown of all data files.

---

## Reproducing Plots (Path A)

After downloading the data, run the notebooks in any order:

| # | Notebook | Output |
|---|---|---|
| 1 | `plot_performance.ipynb` | Main performance curves → `data/output/` |
| 2 | `plot_appendix.ipynb` | Appendix plots → `data/output/` |
| 3 | `plot_result_table.ipynb` | LaTeX result tables → `data/results_*.txt` |
| 4 | `plot_retention_overlap.ipynb` | Set overlap heatmaps → `output/` |
| 5 | `plot_hyperparams.ipynb` | Hyperparameter distributions → `data/output/` |
| 6 | `plot_runtime.ipynb` | Runtime vs. performance → `data/output/` |
| 7 | `example_synthetic.ipynb` | Toy 2D examples → `output/` |
| 8 | `example_fashion_mnist.ipynb` | Standalone Fashion-MNIST experiment |

Create output directories if they do not exist:

```bash
mkdir -p data/output output data/results
```

### MLflow run name prefixes expected by the notebooks

| Prefix | Method |
|---|---|
| `DataAttrOpt(5000, 0.03, 15, best, alpha=best)` | CDVM |
| `PruningOptimization` | Influence-based pruning |
| `DataOob(1000` | DataOOB baseline |
| `DataBanzhaf(num` | Banzhaf baseline |
| `RandomEvaluator()` | Random baseline |

Run names are matched by prefix — any change to a method's `__repr__` will break notebook lookups.

---

## Running Experiments from Scratch (Path B)

### Step 1 — Baselines

```bash
python run_baselines.py
```

Runs Random, DataOOB, and Banzhaf across all 6 datasets with 25 repetitions each.
Creates MLflow experiments `final-3-{dataset}-25`.
Expected runtime: several hours on a GPU machine.

### Step 2 — Pruning optimization

```bash
python run_pruning.py
```

Runs influence-based pruning for all datasets and budget sizes [50, 100, 150, 200, 250, 300].
Adds `PruningOptimization` runs to the `final-3-{dataset}-25` experiments.
Requires Gurobi and the precomputed influence score matrices (or `rerun=True` to recompute).

### Step 3 — Attribution matrix precomputation (optional)

```bash
python run_attr_computation.py
```

Precomputes the train–test attribution matrix and stores it under `data/attr_matr/`.
If skipped, `run_cdvm.py` computes attribution matrices on the fly (much slower).

### Step 4 — CDVM optimization

```bash
python run_cdvm.py
```

Runs CDVM for all datasets, budget sizes, and hyperparameter configurations.
Adds `DataAttrOpt(...)` runs to `final-3-{dataset}-25` and writes:
- `data/attr_datavals/` — binary selection vectors per run
- `data/attr_selected_vals/` — best `max_val` / `alpha` per run

Requires Gurobi and either precomputed attribution matrices (Step 3) or sufficient compute.

### Step 5 — Generate plots

Follow Path A once all MLflow runs are present.

---

## Key Source Files

| File | Purpose |
|---|---|
| `methods/cdvm.py` | CDVM algorithm (primary contribution) |
| `methods/pruning_optimization.py` | Influence-based pruning (baseline) |
| `methods/dataoob.py` | DataOOB evaluator (baseline) |
| `run_baselines.py` | Entry point: baseline experiments |
| `run_cdvm.py` | Entry point: CDVM optimization |
| `run_pruning.py` | Entry point: pruning optimization |
| `run_attr_computation.py` | Precompute attribution matrices |
| `utils_run.py` | MLflow logging and result export |
| `utils_plot.py` | Plot aggregation helpers |

---


