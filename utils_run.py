import math
import json
import re
import pandas as pd
import numpy as np
from datetime import datetime

from opendataval.experiment.exper_methods import remove_high_low
from opendataval.experiment import ExperimentMediator
from opendataval.experiment.util import filter_kwargs
from opendataval.dataval import RandomEvaluator
from opendataval.metrics import Metrics

from mlflow import log_metric, log_param, log_params, log_table

from methods.cdvm import CDVM

import mlflow


def _configure_mlflow_tracking():
    import os
    uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(uri)
    print("Connected to", uri)

_configure_mlflow_tracking()


def make_data_evaluator(base_class, num_evaluators, **kwargs):
    """Create a list of ``num_evaluators`` instances of ``base_class(**kwargs)``."""
    return [base_class(**kwargs) for _ in range(num_evaluators)]


def find_run(experiment_name, method_start):
    """Return the result table for the first run whose name starts with `method_start`.

    Reads entirely from the local mlruns/ directory — no HTTP calls — to avoid
    truncated responses (large param arrays) and artifact-download 500 errors.
    Skips runs that have no results_table.json (partial/failed runs).
    """
    import os

    mlruns_path = os.path.join(os.path.dirname(__file__), "mlruns")

    # Resolve experiment directory by scanning meta.yaml files
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

    for run_id in os.listdir(exp_path):
        meta_file = os.path.join(exp_path, run_id, "meta.yaml")
        if os.path.isfile(meta_file):
            with open(meta_file) as f:
                if "lifecycle_stage: deleted" in f.read():
                    continue
        tag_file = os.path.join(exp_path, run_id, "tags", "mlflow.runName")
        if not os.path.isfile(tag_file):
            continue
        with open(tag_file) as f:
            name = f.read().strip()
        if name.startswith("Joint plots") or not name.startswith(method_start):
            continue
        # Read results_table.json directly from the local filesystem
        table_path = os.path.join(os.path.dirname(__file__), "mlartifacts", os.path.basename(exp_path), run_id, "artifacts", "results_table.json")
        if not os.path.isfile(table_path):
            continue  # partial/failed run — try next match
        with open(table_path) as f:
            data = json.load(f)
        return pd.DataFrame(data=data["data"], columns=data["columns"])

    raise ValueError(f"No run starting with '{method_start}' found in experiment '{experiment_name}'.")


def log_data_vals(exper_med: ExperimentMediator):
    """Log raw data values and top-100 indices to the active MLflow run for each evaluator."""
    for i, dve in enumerate(exper_med.data_evaluators):
        log_param(f"z_data_values_experiment_{i}", dve.data_values)
        log_param(f"z_sorted_top_100_idcs_{i}", np.argsort(dve.data_values)[-100:])


def get_expr_med_log_params(exper_med: ExperimentMediator):
    """Extract loggable metadata from an ExperimentMediator (dataset, split sizes, model, metric)."""
    params = {
        "dataset_name": exper_med.fetcher.dataset.dataset_name,
        "noise_idcs": exper_med.fetcher.noisy_train_indices,
        "train_count": len(exper_med.fetcher.train_indices),
        "valid_count": len(exper_med.fetcher.valid_indices),
        "test_count": len(exper_med.fetcher.test_indices),
        "train_kwargs": exper_med.train_kwargs,
        "model_name": exper_med.pred_model,
        "metric_name": exper_med.metric,
    }

    return params


def log_single_method_df(df_resp,
                         least="remove_least_influential_first_Metrics.ACCURACY",
                         most="remove_most_influential_first_Metrics.ACCURACY",
                         start_run=True, color=None
                         ):
    """Group ``df_resp`` by retention fraction and log mean LEAST/MOST accuracy to MLflow.

    Optionally starts a new run named after the method and colour-tags it.
    ``df_resp`` must contain results for exactly one method.
    """

    assert len(df_resp["Method"].unique()) == 1, "Data frame must contain only one method."

    # Group data frame and log average across all runs to mlflow
    df_grouped = df_resp.groupby("axis", as_index=False).agg({least: ['mean', 'std'],
                                                              most: ['mean', 'std']})

    df_grouped.columns = ["axis", "mean_least", "std_least", "mean_most", "std_most"]

    if start_run:
        kwargs = {"run_name": df_resp["Method"].unique()[0]}
        if color is not None:
            kwargs["tags"] = {"mlflow.runColor": color}

        mlflow.start_run(**kwargs)

    key_least = "LEAST." + least.split(".")[-1]
    key_most = "MOST." + most.split(".")[-1]
    for i, row in df_grouped.iterrows():
        log_metric(key_least, row["mean_least"], step=int(row["axis"] * 100))
        log_metric(key_most, row["mean_most"], step=int(row["axis"] * 100))


def log_df_to_metric(df, methodwise=False, params=None):
    """Log mean LEAST/MOST accuracy per method to MLflow.

    If ``methodwise=True``, each method gets its own run with high/low plots;
    otherwise all methods are logged into the current run under prefixed metric keys.
    Method names are sanitised (punctuation and large run-count numbers stripped)
    to keep MLflow metric keys short.
    """

    def replace_(meth):
        m_ = meth.replace(',', ' ').replace('(', ' ').replace(')', '').replace('=', '')

        m_ = m_.replace('num_models', '')

        # Replace number of runs (e.g. 3 to 5 digit numbers) with ''
        return re.sub(r'\d{3,5}', '', m_)

    least = "remove_least_influential_first_Metrics.ACCURACY"
    most = "remove_most_influential_first_Metrics.ACCURACY"

    methods = df["Method"].unique()

    for m in methods:
        if methodwise:
            mlflow.start_run(run_name=m)

        df_resp_1 = df[df["Method"] == m].copy()
        df_resp_1[least] = df_resp_1[least].astype(float)
        df_resp_1[most] = df_resp_1[most].astype(float)

        df_grouped = df_resp_1.groupby("axis", as_index=False).agg({least: ['mean', 'std'],
                                                                    most: ['mean', 'std']})

        df_grouped.columns = ["axis", "mean_least", "std_least", "mean_most", "std_most"]

        for i, row in df_grouped.iterrows():
            key_l = f"{replace_(m)} LEAST" if not methodwise else "LEAST"
            key_m = f"{replace_(m)} MOST" if not methodwise else "MOST"

            log_metric(key_l, row["mean_least"], step=int(row["axis"] * 100))
            log_metric(key_m, row["mean_most"], step=int(row["axis"] * 100))

            if params is not None:
                log_params(params)

        if methodwise:
            from utils_plot import aggregate_high_low, log_and_show_current_fig
            dataset_name = params["dataset_name"] if params is not None else ""
            aggregate_high_low(df_resp_1, col_low=least, col_high=most, title_postfix=dataset_name)
            log_and_show_current_fig(name="low_high_" + m)
            mlflow.end_run()


def expr_med_(exper_med, exper_func, fig, col=2, alt_metric=None, **exper_kwargs):
    """---Copied from OpenDataVal: enables saving the same experiment multiple times ---"""
    data_eval_perf = []

    metric = alt_metric if alt_metric is not None else exper_med.metric
    filtered_kwargs = filter_kwargs(
        exper_func,
        train_kwargs=exper_med.train_kwargs,
        metric=metric,
        model=exper_med.pred_model,
        plot="placeholder" if fig is not None else None,  # noqa
        **exper_kwargs,
    )

    row = math.ceil(exper_med.num_data_eval / col)

    for i, data_val in enumerate(exper_med.data_evaluators, start=1):
        if fig is not None:
            filtered_kwargs["plot"] = fig.add_subplot(row, col, i)

        eval_resp = exper_func(data_val, exper_med.fetcher, **filtered_kwargs)
        eval_resp["Method"] = str(data_val)
        data_eval_perf.append(eval_resp)

    df_resp = pd.DataFrame(data_eval_perf)
    df_resp = df_resp.explode(list(df_resp.columns[:-1]))

    return df_resp


def run_and_log_single_method(exper_med, base_class, num_exp, dve_kwargs, exp_name=None,
                              least="remove_least_influential_first_Metrics.ACCURACY",
                              most="remove_most_influential_first_Metrics.ACCURACY",
                              least_wort_acc="remove_least_influential_first_Metrics.WORST_CLASS_ACCURACY",
                              most_wort_acc="remove_most_influential_first_Metrics.WORST_CLASS_ACCURACY",
                              check_available=True, log_joint_bool=True):
    """Run a single method and log results to new run."""

    num_models = dve_kwargs["num_models_or_path"] if "num_models_or_path" in dve_kwargs else None
    num_models = dve_kwargs["num_models"] if "num_models" in dve_kwargs else num_models

    exp_name = exp_name if exp_name is not None else f"IDV-{exper_med.fetcher.dataset.dataset_name}-{num_models}"
    mlflow.set_experiment(exp_name)

    # Check if experiment is already available. If so, skip
    if check_available:
        r_name = str(base_class(**dve_kwargs))
        mlflow_runs = mlflow.search_runs(experiment_names=[exp_name],
                                         filter_string=f"tags.'mlflow.runName' = '{r_name}'")
        if len(mlflow_runs) > 0:
            print(f"Experiment {r_name} already available. Skipping...")
            return get_single_result_table(mlflow_runs.iloc[0]["artifact_uri"])

    dves = make_data_evaluator(base_class, num_exp, **dve_kwargs)
    exper_med.data_evaluators = []
    exper_med = exper_med.compute_data_values(data_evaluators=dves)

    # log with accuracy
    df_resp_1 = expr_med_(exper_med, remove_high_low, None)

    # log with worst_class_accuracy
    df_resp_2 = expr_med_(exper_med, remove_high_low, None, alt_metric=Metrics.WORST_CLASS_ACCURACY)

    # Log single runs to mlflow
    mlflow.end_run()
    log_single_method_df(df_resp_1, least=least, most=most, start_run=True)
    log_single_method_df(df_resp_2, least=least_wort_acc, most=most_wort_acc, start_run=False)

    params = get_expr_med_log_params(exper_med)
    params["num_exper"] = num_exp

    if (isinstance(base_class(), CDVM) and
            isinstance(num_models, str)):

        n_models_prob_epochs = num_models.split("-")[-1]
        n_mdls, prob, epochs = n_models_prob_epochs.split("_")[:3]

        params["num_models"] = int(n_mdls)
        dve_kwargs["prob"] = float(prob) if prob != "random" else prob
        params["epochs"] = int(epochs[:-4])

    log_params(dve_kwargs)
    log_params(params)

    # Log plot and table to mlflow
    from utils_plot import aggregate_high_low, log_and_show_current_fig
    dataset_name = params["dataset_name"] if params is not None else ""
    aggregate_high_low(df_resp_1, col_low=least, col_high=most, title_postfix=dataset_name)
    log_and_show_current_fig(name="low_high_" + str(dves[0]))
    log_table(data=df_resp_1, artifact_file="results_table.json")
    log_table(data=df_resp_2, artifact_file="results_worst_class_acc_table.json")

    log_data_vals(exper_med)
    mlflow.end_run()

    return df_resp_1


def get_single_result_table(artifact_uri_wo_table):
    """Returns results table from single run."""
    art_path = artifact_uri_wo_table + "/results_table.json"
    art = mlflow.artifacts.download_artifacts(art_path)

    with open(art, "r") as f:
        json_data = json.load(f)

    return pd.DataFrame(data=json_data["data"], columns=json_data["columns"])


def get_results_table(experiment_name):
    """Returns a concatenated data frame of all results tables from a given experiment."""

    runs = mlflow.search_runs(experiment_names=[experiment_name])
    df_resp = []

    for idx, r in runs.iterrows():
        if not r["tags.mlflow.runName"].startswith("Joint plots"):
            df = get_single_result_table(r["artifact_uri"])
            df_resp.append(df)

    return pd.concat(df_resp)


def log_joint(df_resp, params, exp_name):
    """Creates joint plots and logs them to mlflow run."""

    mlflow.end_run()
    mlflow.set_experiment(exp_name)

    # If no data frame is provided, collect result.json from artifacts
    if df_resp is None:
        df_resp = get_results_table(exp_name)

    from utils_plot import aggregate_any_two_from_many
    mlflow.start_run(run_name="Joint plots and table " + datetime.now().strftime('%Y-%m-%d_%H-%M'))
    aggregate_any_two_from_many(df_resp, title=params["dataset_name"])
    log_params(params)
    mlflow.log_table(data=df_resp, artifact_file="full_results_table.json")
    mlflow.end_run()


def run_multiple_models(exper_med, base_class, num_models, num_exper, dve_kwargs,
                        alt_train_kwargs={"epochs": 1, "batch_size": 32}, check_available=True, random=True, log_joint_bool=True):
    """Run ``base_class`` for each entry in ``num_models`` and log each to ``exp_name``.

    Optionally prepends a RandomEvaluator baseline (``random=True``) and logs a
    joint comparison plot after all models have run (``log_joint_bool=True``).
    Already-logged runs are skipped when ``check_available=True``.
    """
    if isinstance(num_models[0], int):
        name = f"{exper_med.fetcher.dataset.dataset_name}-{base_class.__name__}-{num_models[0]}-{num_exper}"
    else:
        name = f"{exper_med.fetcher.dataset.dataset_name}-{base_class.__name__}-npy-{num_exper}"

    mlflow.end_run()
    mlflow.set_experiment(name)

    if random:
        df_resp = [run_and_log_single_method(exper_med, RandomEvaluator, 50, {}, exp_name=name, check_available=check_available)]
    else:
        df_resp = []

    for n in num_models:
        kwargs_new = dve_kwargs.copy()
        if isinstance(base_class(), CDVM):

            kwargs_new["num_models_or_path"] = n

        else:
            kwargs_new["num_models"] = n

        df_resp.append(run_and_log_single_method(exper_med, base_class, num_exper,
                                                 kwargs_new, exp_name=name, check_available=check_available,
                                                 log_joint_bool=log_joint_bool))

    df_resp = pd.concat(df_resp)
    if log_joint_bool:
        # Log joint plots to mlflow
        params = get_expr_med_log_params(exper_med)
        params["num_models"] = num_models
        params["num_exper"] = num_exper
        params["alt_train_kwargs_idv"] = alt_train_kwargs

        log_joint(df_resp, params, name)

    return df_resp
