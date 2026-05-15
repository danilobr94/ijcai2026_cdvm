"""Run the three baseline data evaluators (RandomEvaluator, DataOobALT, DataBanzhaf)
across six datasets and log all results to the shared ``final-3-{dataset}-25``
MLflow experiments used by the plot notebooks.

Workflow per dataset (``run``):
  1. ``run_and_log_single_method`` runs ``num_exp`` independent repetitions of one
     evaluator, logs accuracy and worst-class-accuracy curves as MLflow metrics,
     plots high/low retention curves, and stores ``results_table.json``.
  2. ``run_and_plot_final`` calls step 1 for each of the three baselines, then
     logs a joint comparison plot and the combined results table.
"""
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from datetime import datetime

import mlflow
from opendataval.dataval import DataBanzhaf, RandomEvaluator

from opendataval.dataloader import mix_labels
from opendataval.experiment import ExperimentMediator

from methods.dataoob import DataOobALT
from methods.cdvm import CDVM


from utils_run import (get_expr_med_log_params, log_params,
                       log_single_method_df, expr_med_, remove_high_low, Metrics,
                       log_table, log_data_vals)

from utils_plot import log_and_show_current_fig, aggregate_any_two_from_many, aggregate_high_low


COLORS = {
    "RandomEvaluator": "#000000",
    "DataOobALT": "#800000",
    "DataBanzhaf": "#BE501E",
}

def make_and_log_final_plot(df_all, title, col="remove_least_influential_first_Metrics.ACCURACY"):
    """Plot mean accuracy vs. fraction removed for all methods, with right-edge labels."""

    plt.figure(figsize=(7, 4))
    i = 0
    labels = []
    label_pos = []
    labels_c = []

    for lbl, df_resp in df_all.groupby("Method"):

        if "Method" in df_resp.columns:
            assert df_resp["Method"].nunique() == 1, "Only one method should be present in the dataframe."

        df_resp = df_resp.drop(columns=["Method"])

        df_grouped = df_resp.astype(float).groupby("axis", as_index=False).agg({col: ['mean', 'std']})
        df_grouped.columns = ["axis", "mean", "std"]

        c = list(mcolors.TABLEAU_COLORS.keys())[i] if lbl != "RandomEvaluator()" else "black"
        i += 1
        plt.plot(df_grouped["axis"] * 100, df_grouped["mean"], label=lbl, c=c)
        l = lbl.split("(")[0]
        l += "_greedy" if "greedy" in lbl else ""

        labels.append(l)
        label_pos.append(df_grouped["mean"].iloc[-1])
        labels_c.append(c)

    # Make sure each label is visible
    label_pos_argsorted = np.argsort(label_pos)
    ax = plt.gca()
    ylim = ax.get_ylim()
    lim = (ylim[1] - ylim[0]) / 15

    for i in range(1, len(label_pos_argsorted)):
        diff = label_pos[label_pos_argsorted[i]] - label_pos[label_pos_argsorted[i - 1]]

        if diff < lim:
            label_pos[label_pos_argsorted[i]] += lim-diff

    # Check if last label pos exceeds ylim
    if label_pos[label_pos_argsorted[-1]] > ylim[1]:
        diff_top = label_pos[label_pos_argsorted[-1]] - ylim[1]
        label_pos = [x - diff_top for x in label_pos]

    for i, lbl in enumerate(labels):
        plt.text(98, label_pos[i], lbl, c=labels_c[i])

    plt.xlim(0, 125)
    plt.xticks(np.arange(0, 120, 20))
    plt.title(title)
    log_and_show_current_fig()


def run_and_log_single_method(exper_med_kwargs, base_class, num_exp, dve_kwargs, exp_name,
                              least="remove_least_influential_first_Metrics.ACCURACY",
                              most="remove_most_influential_first_Metrics.ACCURACY",
                              least_wort_acc="remove_least_influential_first_Metrics.WORST_CLASS_ACCURACY",
                              most_wort_acc="remove_most_influential_first_Metrics.WORST_CLASS_ACCURACY",
                              check_available=True):
    """Run ``num_exp`` repetitions of ``base_class`` and log results to a new MLflow run."""
    num_models = dve_kwargs["num_models_or_path"] if "num_models_or_path" in dve_kwargs else None
    num_models = dve_kwargs["num_models"] if "num_models" in dve_kwargs else num_models

    mlflow.set_experiment(exp_name)

    df_resp_1 = []
    df_resp_2 = []
    for _ in range(num_exp):
        exper_med = ExperimentMediator.model_factory_setup(**exper_med_kwargs)
        exper_med.data_evaluators = []
        dves = [base_class(**dve_kwargs)]
        exper_med = exper_med.compute_data_values(data_evaluators=dves)

        df_resp_1.append(expr_med_(exper_med, remove_high_low, None))

        # log with worst_class_accuracy
        df_resp_2.append(expr_med_(exper_med, remove_high_low, None, alt_metric=Metrics.WORST_CLASS_ACCURACY))

    df_resp_1 = pd.concat(df_resp_1)
    df_resp_2 = pd.concat(df_resp_2)

    # Log single runs to mlflow
    mlflow.end_run()
    c = COLORS.get(base_class.__name__)
    log_single_method_df(df_resp_1, least=least, most=most, start_run=True, color=c)
    log_single_method_df(df_resp_2, least=least_wort_acc, most=most_wort_acc, start_run=False, color=c)

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
    dataset_name = params["dataset_name"] if params is not None else ""
    aggregate_high_low(df_resp_1, col_low=least, col_high=most, title_postfix=dataset_name)
    log_and_show_current_fig(name="low_high_" + str(dves[0]))

    aggregate_high_low(df_resp_2, col_low=least_wort_acc, col_high=most_wort_acc, title_postfix=dataset_name+"worst_class_acc")
    log_and_show_current_fig(name="low_high_worst_class_" + str(dves[0]))

    log_table(data=df_resp_1, artifact_file="results_table.json")
    log_table(data=df_resp_2, artifact_file="results_worst_class_acc_table.json")

    log_data_vals(exper_med)
    mlflow.end_run()

    return df_resp_1


def run_and_plot_final(exper_med_kwargs, num_exper=50,
                       alt_train_kwargs_idv={"epochs": 1, "batch_size": 32},  # noqa
                       alt_train_kwargs_oob={"epochs": 15, "batch_size": 250},  # noqa
                       rdv_step=100, rdv_models=1000):
    """Run RandomEvaluator, DataOobALT, and DataBanzhaf; log all results and joint plots."""

    name = f"final-3-{exper_med_kwargs['dataset_name']}-{num_exper}"

    mlflow.end_run()
    mlflow.set_experiment(name)

    df_resp = [run_and_log_single_method(exper_med_kwargs, RandomEvaluator, 100, {},
                                         exp_name=name, check_available=False)]

    df_resp.append(run_and_log_single_method(exper_med_kwargs, DataOobALT, num_exper,  # noqa
                                             {"num_models": 1000, "alt_train_kwargs": alt_train_kwargs_oob},
                                             exp_name=name, check_available=False))

    df_resp.append(run_and_log_single_method(exper_med_kwargs, DataBanzhaf, num_exper,
                                             {"num_models": 1000},
                                             exp_name=name, check_available=False))

    df_resp = pd.concat(df_resp)

    mlflow.end_run()
    mlflow.set_experiment(name)
    mlflow.start_run(run_name="Joint plots and table " + datetime.now().strftime('%Y-%m-%d_%H-%M'))

    aggregate_any_two_from_many(df_resp, title=exper_med_kwargs['dataset_name'])
    log_params(exper_med_kwargs)
    mlflow.log_table(data=df_resp, artifact_file="full_results_table.json")
    make_and_log_final_plot(df_resp, name)

    mlflow.end_run()

    return df_resp


def run(dataset_name="bbc-embeddings", num_experiments=2,  # noqa
        alt_train_kwargs_idv={"epochs": 1, "batch_size": 32, "lr": 0.01},  # noqa
        alt_train_kwargs_oob={"epochs": 15, "batch_size": 250, "lr": 0.01},  # noqa
        device="cuda"):  # noqa
    """Configure an ExperimentMediator for ``dataset_name`` and run all baselines."""
    train_kwargs = {"epochs": 15, "batch_size": 250, "lr": 0.01}

    exper_med_kwargs = {
        "dataset_name": dataset_name,
        "cache_dir": "./data_files/",
        "force_download": False,
        "train_count": 1000,
        "valid_count": 500,
        "test_count": 500,
        "add_noise": mix_labels,
        "noise_kwargs": {'noise_rate': 0.1},
        "train_kwargs": train_kwargs,
        "model_name": "LogisticRegression",
        "metric_name": "accuracy",
        "random_state": 42,
        "device": torch.device(device)
    }

    print("Cuda available:", torch.cuda.is_available())
    print(f"\n\n\nRunning {dataset_name} with {num_experiments} experiments\n\n\n")
    run_and_plot_final(exper_med_kwargs, num_experiments, alt_train_kwargs_idv, alt_train_kwargs_oob)


run(dataset_name="nomao", num_experiments=25)
run(dataset_name="pol", num_experiments=25)
run(dataset_name="adult", num_experiments=25)
run(dataset_name="cifar10-embeddings", num_experiments=25)
run(dataset_name="bbc-embeddings", num_experiments=25)
run(dataset_name="imdb-embeddings", num_experiments=25)
