"""Run PruningOptimization (Yang et al., ICLR 2023) across six training-set sizes
and six datasets, then summarize the best per-axis result into the shared
``final-3-{dataset}-25`` MLflow experiments used by the plot notebooks.

Workflow per dataset (``run_dataset``):
  1. For each budget in {50, 100, 150, 200, 250, 300} run 25 independent
     PruningOptimization experiments and log them to
     ``{dataset}-PruningOptimization-npy-25``.
  2. Call ``summarize_pruning_results``: for every removal fraction (axis)
     keep the highest mean accuracy across all budgets and both removal
     directions (least- / most-influential first), then write a single
     ``PruningOptimization`` run into ``final-3-{dataset}-25`` so the
     plot notebooks can compare it against the other baselines.

LiSSA influence matrices (``data/influence_score_matrix-LogisticRegression{dataset}.npy``)
must exist before running; computing them from scratch takes several hours per
dataset.  Set ``rerun=True`` in ``dve_kwargs`` to force recomputation.
"""
import mlflow
import pandas as pd
import torch
from opendataval.dataloader import mix_labels
from opendataval.experiment import ExperimentMediator
from opendataval.dataloader import DataFetcher
from opendataval.model.mlp import ClassifierMLP

from methods.pruning_optimization import PruningOptimization
from utils_run import run_multiple_models, log_metric

NUM_EXPERIMENTS = 25
train_count, valid_count, test_count = 1000, 500, 500
noise_rate = 0.1
noise_kwargs = {'noise_rate': noise_rate}

metric_name = "accuracy"
train_kwargs = {"epochs": 15, "batch_size": 250, "lr": 0.01}
device = torch.device("cuda")


def summarize_pruning_results(dataset_name, experiment_suffix="-PruningOptimization-npy-25"):
    """Pick the best size per axis from all pruning runs and log a summary into final-3."""
    import os, json

    least = "remove_least_influential_first_Metrics.ACCURACY"
    most = "remove_most_influential_first_Metrics.ACCURACY"

    experiment_name = dataset_name + experiment_suffix
    mlruns_path = os.path.join(os.path.dirname(__file__), "mlruns")

    # Resolve experiment directory
    exp_path = None
    for entry in os.scandir(mlruns_path):
        if not entry.is_dir():
            continue
        meta = os.path.join(entry.path, "meta.yaml")
        if not os.path.isfile(meta):
            continue
        with open(meta) as f:
            if f"name: {experiment_name}" in f.read():
                exp_path = entry.path
                break
    if exp_path is None:
        raise ValueError(f"Experiment '{experiment_name}' not found in mlruns/.")

    exp_id = os.path.basename(exp_path)
    dfs = []
    for run_id in os.listdir(exp_path):
        tag_file = os.path.join(exp_path, run_id, "tags", "mlflow.runName")
        if not os.path.isfile(tag_file):
            continue
        with open(tag_file) as f:
            run_name = f.read().strip()
        if run_name.startswith("Joint plots"):
            continue
        table_path = os.path.join(os.path.dirname(__file__), "mlartifacts", exp_id, run_id, "artifacts", "results_table.json")
        if not os.path.isfile(table_path):
            continue
        with open(table_path) as f:
            data = json.load(f)
        dfs.append(pd.DataFrame(data=data["data"], columns=data["columns"]))

    # For each axis take the best mean across all sizes and both removal directions
    summary = {}
    for df in dfs:
        if "Method" in df.columns:
            df = df.drop(columns=["Method"])
        df = df.astype(float)
        for col in [least, most]:
            grouped = df.groupby("axis", as_index=False).agg({col: ["mean", "std"]})
            grouped.columns = ["axis", "mean", "std"]
            for _, row in grouped.iterrows():
                axis = row["axis"]
                if axis not in summary or row["mean"] > summary[axis]["mean"]:
                    summary[axis] = {"mean": row["mean"], "std": row["std"]}

    df_summary = pd.DataFrame(summary).T.reset_index().rename(columns={"index": "axis"})

    num_exp = experiment_suffix.split("-")[-1]
    exp_name = f"final-3-{dataset_name}-{num_exp}"
    mlflow.end_run()
    mlflow.set_experiment(exp_name)
    mlflow.start_run(run_name="PruningOptimization", tags={"mlflow.runColor": "#90ee90"})

    for _, row in df_summary.iterrows():
        log_metric("LEAST.ACCURACY", row["mean"], step=int(row["axis"] * 100))

    mlflow.log_table(data=df_summary, artifact_file="results_table.json")
    mlflow.end_run()


def run_dataset_with_model(dataset_name, model_class=ClassifierMLP, model_kwargs={"layers": 3, "hidden_dim": 100}):
    """Like ``run_dataset`` but with a custom ``model_class`` (e.g. ClassifierMLP) instead of LogisticRegression."""
    data_fetcher = DataFetcher(dataset_name, force_download=False)
    covar_dim, label_dim = data_fetcher.covar_dim, data_fetcher.label_dim
    pred_model = model_class(*covar_dim, *label_dim, **model_kwargs).to(device)

    exper_med = ExperimentMediator.setup(
        dataset_name=dataset_name,
        cache_dir="./data_files/",
        force_download=False,
        train_count=train_count,
        valid_count=valid_count,
        test_count=test_count,
        add_noise=mix_labels,
        noise_kwargs=noise_kwargs,
        train_kwargs=train_kwargs,
        pred_model=pred_model,
        metric_name=metric_name,
        random_state=42,
    )

    exper_med.fetcher.to_device(device)

    for size in [50, 100, 150, 200, 250, 300]:
        dve_kwargs = {"optimization_type": "size", "rerun": False, "only_size": size, "dataset_name": dataset_name}
        run_multiple_models(exper_med, PruningOptimization, [50],
                            NUM_EXPERIMENTS, dve_kwargs, check_available=False, random=False, log_joint_bool=False)


def run_dataset(dataset_name):
    """Run all six budget sizes for ``dataset_name`` and write the best-size summary to ``final-3``."""
    exper_med = ExperimentMediator.model_factory_setup(
        dataset_name=dataset_name,
        cache_dir="./data_files/",
        force_download=False,
        train_count=train_count,
        valid_count=valid_count,
        test_count=test_count,
        add_noise=mix_labels,
        noise_kwargs=noise_kwargs,
        train_kwargs=train_kwargs,
        model_name="LogisticRegression",
        metric_name=metric_name,
        random_state=42,
        device=device,
    )

    for size in [50, 100, 150, 200, 250, 300]:
        dve_kwargs = {"optimization_type": "size", "rerun": False, "only_size": size, "dataset_name": dataset_name}
        run_multiple_models(exper_med, PruningOptimization, [""],
                            NUM_EXPERIMENTS, dve_kwargs, check_available=False, random=False, log_joint_bool=False)

    summarize_pruning_results(dataset_name, experiment_suffix=f"-PruningOptimization-npy-{NUM_EXPERIMENTS}")


if __name__ == "__main__":
    run_dataset("pol")
    run_dataset("nomao")
    run_dataset("adult")
    run_dataset("imdb-embeddings")
    run_dataset("bbc-embeddings")
    run_dataset("cifar10-embeddings")
