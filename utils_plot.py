import itertools
import time

import numpy as np
import matplotlib.pyplot as plt
from mlflow import log_figure

from utils_run import find_run


def get_and_preprocess_run(dataset_name, run_wildcard, exp_name=None, keep_only=6,
                           col="remove_least_influential_first_Metrics.ACCURACY"):
    """Fetch a run from MLflow and return a grouped mean/std dataframe."""
    exp_name = "final-3-" + dataset_name + "-25" if exp_name is None else exp_name
    df = find_run(exp_name, run_wildcard)

    if "Method" in df.columns:
        assert df["Method"].nunique() == 1

    if not run_wildcard.startswith("PruningOptimization"):
        df = df.drop(columns=["Method"])
        df = df.astype(float).groupby("axis", as_index=False).agg({col: ["mean", "std"]})
        df.columns = ["axis", "mean", "std"]

    if len(df) > keep_only:
        df = df.iloc[-keep_only:]

    return df


def log_and_show_current_fig(name="fig", show=False):
    """Log matplotlib figure to mlflow."""
    plt.tight_layout()
    pth = f"{name}.pdf"
    log_figure(plt.gcf(), pth)
    if show:
        plt.show()
    plt.close()


def aggregate_high_low(df_resp, ax=None, title_postfix="",
                       col_low="remove_least_influential_first_Metrics.ACCURACY",
                       col_high="remove_most_influential_first_Metrics.ACCURACY",
                       **kwargs):
    """Plots results of multiple runs of the same experiments of 'remove_high_low' in one plot."""
    fig, ax = plt.subplots() if ax is None else ax
    aggregate_plot(df_resp, ax=ax, col=col_low, label="Low", **kwargs)

    aggregate_plot(df_resp, ax=ax, mean_color="darkred", shade_color="lightcoral",
                   col=col_high, label="High", **kwargs)

    title = df_resp["Method"].iloc[0] + " - " + title_postfix
    ax.set_title(title)

    return ax


def aggregate_any_two_from_many(df_resp, ax=None,
                                col="remove_least_influential_first_Metrics.ACCURACY",
                                title="",
                                **kwargs):
    """Aggregate all pairwise combinations from dataframe."""
    methods = df_resp["Method"].unique()

    for m1, m2 in itertools.combinations(methods, 2):
        df_resp_1 = df_resp[df_resp["Method"] == m1]
        df_resp_2 = df_resp[df_resp["Method"] == m2]

        aggregate_two(df_resp_1, df_resp_2, ax=ax, col=col, title=title, **kwargs)

        plt.tight_layout()
        log_and_show_current_fig(name=f"{title}_{m1}_vs_{m2}")
        time.sleep(5)


def aggregate_two(df_resp_1, df_resp_2, ax=None,
                  col="remove_least_influential_first_Metrics.ACCURACY",
                  title="",
                  **kwargs):
    """Plots results of two experiments for column 'col' in one plot."""
    ax = plt.subplots()[1] if ax is None else ax

    aggregate_plot(df_resp_1, ax=ax, mean_color="dodgerblue", shade_color="lightblue",
                   col=col, label=df_resp_1["Method"].iloc[0], **kwargs)

    aggregate_plot(df_resp_2, ax=ax, mean_color="darkred", shade_color="lightcoral",
                   col=col, label=df_resp_2["Method"].iloc[0], **kwargs)

    if title != "":
        ax.set_title(title)

    return ax


def aggregate_plot(df_resp, ax=None,
                   shade_color="lightblue", mean_color="dodgerblue",
                   xlabel="Number of removed data points", ylabel="Accuracy",
                   title="", label="",
                   col="remove_least_influential_first_Metrics.ACCURACY",
                   y_lim=None):
    """Plots mean and std of multiple runs of 'remove_high_low' for column 'col'.

    The dataframe must only contain results for one method.

    """

    ax_ = plt.subplots()[1] if ax is None else ax

    if "Method" in df_resp.columns:
        assert df_resp["Method"].nunique() == 1, "Only one method should be present in the dataframe."

    df_resp = df_resp.drop(columns=["Method"])

    df_grouped = df_resp.astype(float).groupby("axis", as_index=False).agg({col: ['mean', 'std']})
    df_grouped.columns = ["axis", "mean", "std"]

    ax_.fill_between(df_grouped["axis"],
                     df_grouped["mean"] - df_grouped["std"],
                     df_grouped["mean"] + df_grouped["std"],
                     alpha=0.3, color=shade_color)

    ax_.plot(df_grouped["axis"], df_grouped["mean"], color=mean_color, label=label)

    ax_.set_title(title)
    ax_.set_xlabel(xlabel)
    ax_.set_ylabel(ylabel)
    ax_.set_xticks(df_grouped["axis"][1::2])
    ax_.tick_params(labelsize=16)

    if y_lim is not None:
        ax_.set_ylim(y_lim)

    if label != "":
        ax_.legend(loc='lower left')

    if ax is None:
        log_and_show_current_fig(name=title)

    return ax_
