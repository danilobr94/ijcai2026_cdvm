"""Run CDVM (Contextual Data Value Model) experiments across datasets and log to MLflow.

For each dataset and dropout probability, ``run_dataset`` runs
``num_experiments`` independent repetitions over a grid of retention budgets
(``budgets``) and logs the aggregated accuracy curves to the shared
``final-3-{dataset}-{num_experiments}`` experiment used by the plot notebooks.

Attribution matrices (``data/attr_matr/``) and initial model weights
(``data/weights/``) are loaded from disk if available; otherwise they are
computed and cached automatically by ``maybe_train_and_store_attr_matrix``.
"""
import torch
import pandas as pd
import numpy as np
from opendataval.dataloader import mix_labels
from opendataval.experiment import ExperimentMediator
from opendataval.experiment.exper_methods import remove_high_low
from opendataval.metrics import Metrics
import mlflow
from tqdm import tqdm

from utils_run import expr_med_, log_single_method_df, log_table, log_data_vals
from utils_plot import log_and_show_current_fig, aggregate_high_low

from methods.cdvm import CDVM
from run_attr_computation import maybe_train_and_store_attr_matrix


def get_random_color():
    """Returns a random color from a fixed palette in hex format."""
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd",
              "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    return np.random.choice(colors)


def run_cdvm(exper_med, dve_kwargs):
    """Fit one CDVM instance and return accuracy and worst-class-accuracy result dataframes."""
    dves = [CDVM(**dve_kwargs)]
    exper_med.data_evaluators = []
    exper_med = exper_med.compute_data_values(data_evaluators=dves)

    df_resp_acc = expr_med_(exper_med, remove_high_low, None)
    df_resp_wort_class_acc = expr_med_(exper_med, remove_high_low, None, alt_metric=Metrics.WORST_CLASS_ACCURACY)

    return df_resp_acc, df_resp_wort_class_acc


def extract_budget_row(result, budget):
    """Return the accuracy dataframe row corresponding to retention ``budget``."""
    for df in result:
        if df is not None and "remove_least_influential_first_Metrics.ACCURACY" in df.columns:
            return df[df["axis"].astype(float) == 1 - budget / 1000]

    raise ValueError(f"No results found for budget {budget}. Ensure the budget is present in the results.")


def run_dataset(dataset_name, budgets, probs, num_models, num_experiments, num_epochs=15,
                only_budget_eq_prob=False, max_val="max+2std", use_banzhaf=False, add_banzhaf=False, alpha=0.8, l1_lambda=0.0):
    """Run CDVM for all combinations of ``probs`` × ``budgets`` and log to ``final-3``.

    For each dropout probability in ``probs``, ``num_experiments`` repetitions
    are run. Each repetition loads (or computes) the attribution matrix, then
    evaluates CDVM for every retention budget in ``budgets``.

    Args:
        dataset_name: Dataset identifier passed to ExperimentMediator.
        budgets: Retention budgets in permille (e.g. range(50, 350, 50)).
        probs: Dropout probabilities used to build the attribution matrix.
        num_models: Number of ensemble models for the attribution matrix.
        num_experiments: Number of independent repetitions per probability.
        num_epochs: Training epochs for both the base model and CDVM fine-tuning.
        only_budget_eq_prob: If True, only evaluate budget == prob * 1000 pairs.
        max_val: Max-value threshold strategy passed to CDVM (e.g. ``"max+2std"``).
        use_banzhaf: Replace CDVM objective with a Banzhaf-style objective.
        add_banzhaf: Add a Banzhaf regularisation term to the CDVM objective.
        alpha: Regularisation weight.
        l1_lambda: L1 penalty weight.
    """
    train_count, valid_count, test_count = 1000, 500, 500
    noise_kwargs = {'noise_rate': 0.1}
    train_kwargs = {"epochs": num_epochs, "batch_size": 250, "lr": 0.01}
    device = torch.device("cuda")

    name = f"final-3-{dataset_name}-{num_experiments}"
    mlflow.end_run()
    mlflow.set_experiment(name)

    results_all = []

    for prob in probs:

        results_prob = []
        for i in tqdm(range(num_experiments)):

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
                metric_name="accuracy",
                random_state=42,
                device=device,
            )

            pm, pw = maybe_train_and_store_attr_matrix(exper_med, num_models, prob, pbar=None, postfix=f"_batch_{i}")

            results_exp_i = []
            for budget in budgets:
                dve_kwargs = {"alt_train_kwargs": {"epochs": num_epochs, "batch_size": 32, "lr": 0.01},
                              "retention_budget": budget, "max_val": max_val, "prob": "frac", "alpha": alpha,
                              "use_banzhaf": use_banzhaf, "num_models_or_path": pm,
                              "add_banzhaf": add_banzhaf, "dataset_name": dataset_name, "l1_lambda": l1_lambda}

                if only_budget_eq_prob and prob != budget / 1000:
                    continue

                results_exp_i.append(extract_budget_row(run_cdvm(exper_med, dve_kwargs), budget))

            results_exp_df = pd.concat(results_exp_i, ignore_index=True) if len(results_exp_i) > 1 else results_exp_i[0]
            results_prob.append(results_exp_df)

        results_prob_df = pd.concat(results_prob, ignore_index=True)

        add_bnzf = ", +b" if add_banzhaf else ""
        if not only_budget_eq_prob:
            n = (f"DataAttrOpt({num_models}, {prob}, {num_epochs}, {dve_kwargs['max_val']}{add_bnzf}, alpha={alpha})"
                 if not use_banzhaf else f"DataBanzhaf({num_models}, {prob}, {num_epochs})")

            results_prob_df["Method"] = n
            log_single_method_df(df_resp=results_prob_df, color=get_random_color())
            mlflow.log_params(dve_kwargs)
            mlflow.log_params({"budget_eq_prob": False})

            aggregate_high_low(results_prob_df, title_postfix=dataset_name)
            log_and_show_current_fig(name="low_high_" + n)
            log_table(data=results_prob_df, artifact_file="results_table.json")

            log_data_vals(exper_med)
            mlflow.end_run()

        else:
            results_all.append(results_prob_df)

    if only_budget_eq_prob:
        results_all_df = pd.concat(results_all, ignore_index=True)
        results_all_df["Method"] = f"DataAttrOpt({num_models}, prob=frac, {num_epochs}, {dve_kwargs['max_val']}{add_bnzf})"
        log_single_method_df(df_resp=results_all_df, color=get_random_color())
        mlflow.log_params(dve_kwargs)
        mlflow.log_params({"budget_eq_prob": True})

        aggregate_high_low(results_all_df, title_postfix=dataset_name)
        log_and_show_current_fig(name="low_high_" + f"DataAttrOpt({num_models}, prob=frac, {num_epochs}, {dve_kwargs['max_val']})")
        log_table(data=results_all_df, artifact_file="results_table.json")

        mlflow.end_run()


if __name__ == "__main__":
    BUDGETS = range(50, 350, 50)              # retention budgets in permille
    SAMPLE_PROB = [0.03]                      # dropout probability for attribution matrix
    NUM_MODELS = 5000                         # ensemble size for attribution matrix
    NUM_EXPERIMENTS = 25                      # independent repetitions per config
    ONLY_BUDGET_EQ_PROB = False               # if True, only evaluate budget == prob * 1000 pairs
    USE_BANZHAF = False                       # replace CDVM objective with Banzhaf
    ADD_BANZHAF = False                       # add Banzhaf regularisation term
    ALPHA = "best"                            # regularisation weight
    L1_LAMBDA = 0.0                           # L1 penalty weight
    MAXVALs = ["best"]                        # max-value threshold strategies to sweep

    for m in MAXVALs:
        run_dataset("nomao", BUDGETS, SAMPLE_PROB,
                    NUM_MODELS, NUM_EXPERIMENTS, only_budget_eq_prob=ONLY_BUDGET_EQ_PROB,
                    max_val=m, use_banzhaf=USE_BANZHAF, add_banzhaf=ADD_BANZHAF, alpha=ALPHA, l1_lambda=L1_LAMBDA)

        run_dataset("pol", BUDGETS, SAMPLE_PROB,
                    NUM_MODELS, NUM_EXPERIMENTS, only_budget_eq_prob=ONLY_BUDGET_EQ_PROB,
                    max_val=m, use_banzhaf=USE_BANZHAF, add_banzhaf=ADD_BANZHAF, alpha=ALPHA, l1_lambda=L1_LAMBDA)

        run_dataset("cifar10-embeddings", BUDGETS, SAMPLE_PROB,
                    NUM_MODELS, NUM_EXPERIMENTS, only_budget_eq_prob=ONLY_BUDGET_EQ_PROB,
                    max_val=m, use_banzhaf=USE_BANZHAF, add_banzhaf=ADD_BANZHAF, alpha=ALPHA, l1_lambda=L1_LAMBDA)

        run_dataset("bbc-embeddings", BUDGETS, SAMPLE_PROB,
                    NUM_MODELS, NUM_EXPERIMENTS, only_budget_eq_prob=ONLY_BUDGET_EQ_PROB,
                    max_val=m, use_banzhaf=USE_BANZHAF, add_banzhaf=ADD_BANZHAF, alpha=ALPHA, l1_lambda=L1_LAMBDA)

        run_dataset("imdb-embeddings", BUDGETS, SAMPLE_PROB,
                    NUM_MODELS, NUM_EXPERIMENTS, only_budget_eq_prob=ONLY_BUDGET_EQ_PROB,
                    max_val=m, use_banzhaf=USE_BANZHAF, add_banzhaf=ADD_BANZHAF, alpha=ALPHA, l1_lambda=L1_LAMBDA)

        run_dataset("adult", BUDGETS, SAMPLE_PROB,
                    NUM_MODELS, NUM_EXPERIMENTS, only_budget_eq_prob=ONLY_BUDGET_EQ_PROB,
                    max_val=m, use_banzhaf=USE_BANZHAF, add_banzhaf=ADD_BANZHAF, alpha=ALPHA, l1_lambda=L1_LAMBDA)
