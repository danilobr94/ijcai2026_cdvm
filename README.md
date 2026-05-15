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

**Want to use CDVM?** Start with [`example_fashion_mnist.ipynb`](example_fashion_mnist.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/danilobr94/ijcai2026_cdvm/blob/main/example_fashion_mnist.ipynb), it is self-contained and runs on Google Colab. [`example_synthetic.ipynb`](example_synthetic.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/danilobr94/ijcai2026_cdvm/blob/main/example_synthetic.ipynb) is similarly standalone. The rest of this repository reproduces the experiments and plots from the paper.

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

Additionally, the patches in `odv_fix/` must be applied manually to the installed `opendataval` package (see `Dockerfile` for reference).

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
are hosted separately and can be downloaded optionally to speed up Path B:

```bash
python download_data.py
```

This fetches `data/attr_matr/` and the `influence_score_matrix-*.npy` files.
If not downloaded, both files are recomputed automatically when running Path B (slower).
See [`data/README.md`](data/README.md) for a full breakdown of all data files.

---

## Reproducing Plots (Path A)

The MLflow results included in this repository are sufficient — run the notebooks in any order:

| # | Notebook | Output |
|---|---|---|
| 1 | `plot_performance.ipynb` | Main performance curves → `data/output/` |
| 2 | `plot_appendix.ipynb` | Appendix plots → `data/output/` |
| 3 | `plot_result_table.ipynb` | LaTeX result tables → `data/results_*.txt` |
| 4 | `plot_retention_overlap.ipynb` | Set overlap heatmaps → `output/` |
| 5 | `plot_hyperparams.ipynb` | Hyperparameter distributions → `data/output/` |
| 6 | `plot_runtime.ipynb` | Runtime vs. performance → `data/output/` |
| 7 | [`example_synthetic.ipynb`](example_synthetic.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/danilobr94/ijcai2026_cdvm/blob/main/example_synthetic.ipynb) | Toy 2D examples → `output/` |
| 8 | [`example_fashion_mnist.ipynb`](example_fashion_mnist.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/danilobr94/ijcai2026_cdvm/blob/main/example_fashion_mnist.ipynb) | Standalone Fashion-MNIST experiment |

Create output directories if they do not exist:

```bash
mkdir -p data/output output data/results
```

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
Requires Gurobi. Influence score matrices are recomputed automatically if not present (slower).

### Step 3 — CDVM optimization

```bash
python run_cdvm.py
```

Runs CDVM for all datasets, budget sizes, and hyperparameter configurations.
Adds `DataAttrOpt(...)` runs to `final-3-{dataset}-25` and writes:
- `data/attr_datavals/` — binary selection vectors per run
- `data/attr_selected_vals/` — best `max_val` / `alpha` per run

Requires Gurobi. Attribution matrices are precomputed automatically if not present (slower).

### Step 4 — Generate plots

Follow Path A once all MLflow runs are present.





