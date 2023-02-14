__all__ = [
    "plot_roc_pr_curves",
    "plot_distributions",
    "plot_ks_curve",
    "plot_regression_curves",
]

import os
import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def plot_roc_pr_curves(
    y_proba: np.ndarray,
    y_true: pd.Series,
    label: str,
    save_eval_artifacts: bool = False,
    eval_artifacts_path: str = ".",
) -> None:
    """Function to plot ROC and PR curves for classifier evaluation.
    Args:
        y_proba (np.ndarray): Array containing probabilities predicted by the model
        y_true (pd.Series): Array containing true label values
        label (str): Label to be displayed on titles (train, test)
        save_eval_artifacts (bool, optional): Whether or not to save visualizations. Defaults to False.
        eval_artifacts_path (bool, optional): Path to where curves should be dumped to. Defaults to current folder.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4), sharey=True)
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    ax1.plot(
        fpr, tpr, color="red", label=f"(AUC = {roc_auc_score(y_true, y_proba):.3f})"
    )
    ax1.plot([0, 1], [0, 1], color="navy")
    ax1.set_xlabel("FPR")
    ax1.set_ylabel("TPR")
    ax1.set_xlim((0, 1))
    ax1.set_ylim((0, 1.001))
    ax1.legend(loc=4)
    ax1.grid(alpha=0.15)
    ax1.set_title(f"{label.upper()} - ROC", fontsize=13)

    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ax2.plot(
        recall,
        precision,
        color="red",
        label=f"(AUC = {average_precision_score(y_true, y_proba):.3f}",
    )
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_xlim((0, 1))
    ax2.set_ylim((0, 1.001))
    ax2.legend(loc=4)
    ax2.grid(alpha=0.15)
    ax2.set_title(f"{label.upper()} - Precision-Recall", fontsize=13)
    plt.show()

    if save_eval_artifacts:
        fig.savefig(os.path.join(eval_artifacts_path, f"roc_pr_{label}.png"))


def plot_distributions(
    y_true: np.ndarray,
    scores: np.ndarray,
    label: str,
    save_eval_artifacts: bool = False,
    eval_artifacts_path: str = ".",
) -> None:
    """Function to plot score distributions for positive and negative classes
    Args:
        y_true (pd.Series): Array containing true label values
        scores (np.ndarray): Predicted credit scores
        label (str): Label to be displayed on titles (train, test)
        show_viz (bool, optional): Whether or not visualizations should be displayed on screen. Defaults to False.
        save_eval_artifacts (bool, optional): Whether or not to save visualizations. Defaults to False.
        eval_artifacts_path (bool, optional): Path to where curves should be dumped to. Defaults to current folder.
    """
    df = pd.DataFrame({"Label": y_true, "Predicted Score": scores})
    default = df[df.Label == 1.0]
    non_default = df[df.Label == 0.0]
    fig, ax = plt.subplots(1, 1, figsize=(15, 4))
    sns.distplot(
        default["Predicted Score"], bins=30, label="Default", color="red", ax=ax
    )
    sns.distplot(
        non_default["Predicted Score"],
        bins=30,
        label="Non-Default",
        color="blue",
        ax=ax,
    )
    ax.set_xlabel("Credit Score")
    ax.grid(alpha=0.15)
    ax.legend()
    ax.set_title(f"{label.upper()} - Score Distribution", fontsize=13)
    plt.show()

    if save_eval_artifacts:
        fig.savefig(os.path.join(eval_artifacts_path, f"distribution_{label}.png"))


def plot_ks_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    label: str,
    target_name: str = "default",
    save_eval_artifacts: bool = False,
    eval_artifacts_path: str = ".",
) -> None:
    """Function to plot score distributions for positive and negative classes
    Args:
        y_true (pd.Series): Array containing true label values
        y_proba (np.ndarray): Array containing probabilities predicted by the model
        label (str): Label to be displayed on titles (train, test)
        target_name (str, optional): Name of the class of interest. Defaults to "default".
        show_viz (bool, optional): Whether or not visualizations should be displayed on screen. Defaults to False.
        save_eval_artifacts (bool, optional): Whether or not to save visualizations. Defaults to False.
        eval_artifacts_path (bool, optional): Path to where curves should be dumped to. Defaults to current folder.
    """
    label_col = target_name.capitalize()
    probability_col = f"Probability of {label_col}"
    default_pct_col = f"% {label_col}"
    non_default_pct_col = f"% Non-{label_col}"

    probabilities_and_labels_train = pd.DataFrame(
        {
            label_col: y_true,
            probability_col: y_proba,
        }
    ).sort_values(probability_col)

    probabilities_and_labels_train[default_pct_col] = (
        probabilities_and_labels_train[label_col].cumsum()
        / probabilities_and_labels_train[label_col].sum()
    )
    probabilities_and_labels_train[non_default_pct_col] = (
        1 - probabilities_and_labels_train[label_col]
    ).cumsum() / (1 - probabilities_and_labels_train[label_col]).sum()
    ks = np.max(
        probabilities_and_labels_train[non_default_pct_col]
        - probabilities_and_labels_train[default_pct_col]
    )

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(
        probabilities_and_labels_train[probability_col],
        probabilities_and_labels_train[default_pct_col],
        label=default_pct_col,
        color="red",
    )
    ax.plot(
        probabilities_and_labels_train[probability_col],
        probabilities_and_labels_train[non_default_pct_col],
        label=non_default_pct_col,
        color="navy",
    )
    plt.title(f"{label} - Kolmogorov-Smirnov | KS-Coefficient = {ks:.3f}")
    plt.legend()
    plt.grid(alpha=0.30, linestyle="--")
    plt.show()

    if save_eval_artifacts:
        fig.savefig(os.path.join(eval_artifacts_path, f"ks_{label}.png"))


def plot_regression_curves(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    name: str,
    save_eval_artifacts: bool = False,
    eval_artifacts_path: str = ".",
) -> None:
    """Method that performs graphical evaluation of a regression model.
    It generates the following visualizations:
        * Distribution of predicted vs. true values
        * Residuals Distribution
        * QQ-Plot
        * Scatterplot of predicted vs. true values
    Args:
        y_true (np.ndarray): True target values
        y_pred (np.ndarray): Predicted target values
        name (str): Name to show in the title.
        convert_exp (bool, optional): Whether or not output needs to be converted from log transformation. Defaults to False.
        save_eval_artifacts (bool, optional): Whether or not to save visualizations. Defaults to False.
        eval_artifacts_path (bool, optional): Path to where curves should be dumped to. Defaults to current folder.
    """
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(14, 5))
    residuals = y_true - y_pred

    # Predicted vs True Values Distribution
    ax1.set_title("Histogram", fontsize=13)
    ax1.hist(y_true, color="lightblue", edgecolor="navy", alpha=1, label="True")
    ax1.hist(y_pred, color="red", edgecolor="red", alpha=0.6, label="Predicted")
    ax1.legend()

    # Residuals distribution
    ax2.set_title("Residuals Distribution", fontsize=13)
    sns.distplot(
        residuals,
        bins=30,
        color="blue",
        ax=ax2,
    )
    ax2.set_xlabel("Residuals")

    # QQ-Plot
    standardized_residuals = (np.sort(residuals) - np.mean(residuals)) / np.std(
        residuals, ddof=1
    )
    theoretical_quantiles = [
        scipy.stats.norm.ppf(p)
        for p in np.linspace(0.01, 0.99, len(standardized_residuals))
    ]
    ax3.set_title("QQ-Plot", fontsize=13)
    sns.regplot(
        x=theoretical_quantiles,
        y=standardized_residuals,
        fit_reg=False,
        scatter_kws={"color": "lightblue", "linewidth": 1, "edgecolors": "navy"},
        line_kws={"color": "red"},
        ax=ax3,
    )
    ax3.plot(
        [min(theoretical_quantiles), max(theoretical_quantiles)],
        [min(theoretical_quantiles), max(theoretical_quantiles)],
        color="red",
        linestyle="--",
    )
    ax3.set_xlim((min(theoretical_quantiles), max(theoretical_quantiles)))
    ax3.set_ylim((min(theoretical_quantiles), max(theoretical_quantiles)))
    ax3.set_ylabel("Standardized Residuals")
    ax3.set_xlabel("Theoretical Quantiles")

    # Scatterplot of predicted vs. true values
    ax4.set_title("Predicted x True", fontsize=13)
    min_val = min([y_pred.min(), y_true.min()])
    max_val = max([y_pred.max(), y_true.max()])
    sns.scatterplot(x=y_pred, y=y_true, color="lightblue", edgecolor="navy", ax=ax4)
    ax4.plot([min_val, max_val], [min_val, max_val], color="red", ls="dashed")
    ax4.set_xlim([min_val, max_val])
    ax4.set_ylim([min_val, max_val])
    ax4.set_xlabel("Predicted Values")
    ax4.set_ylabel("True Values")

    plt.tight_layout()
    plt.show()

    if save_eval_artifacts:
        fig.savefig(
            os.path.join(eval_artifacts_path, f"{name}_regression_evaluation.png")
        )
