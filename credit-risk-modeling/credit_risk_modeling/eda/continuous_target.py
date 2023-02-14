__all__ = ["boxplot_by_categories", "regression_plot", "correlation_heatmap"]

from typing import List, Optional
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def boxplot_by_categories(
    data: pd.DataFrame, features: List[str], target_variable: str
) -> None:
    """Function that generates boxplot for a continuous target variable, breaking it down by a categorical variable.

    Args:
        data (pd.DataFrame): Dataframe containing features and the target variable.
        features (List[str]): List of categorical features to be considered for the breakdown.
        target_variable (str): The name of the target variable in the dataframe.
    """

    _categories_threhsold = 10

    n_features = len(features)
    viz_by_feature = 2
    _ = plt.figure(figsize=(15, n_features * 4))

    for i, feature in tqdm(enumerate(features)):
        ax = plt.subplot(n_features, viz_by_feature, 2 * i + 1)
        categories = data[feature].unique()
        n_categories = len(categories)
        sns.boxplot(data=data, x=feature, y=target_variable, ax=ax)
        if n_categories > _categories_threhsold:
            ax.set_xticklabels(labels=categories.tolist(), rotation=90)
        ax.grid(alpha=0.3, linestyle="--")
        ax = plt.subplot(n_features, viz_by_feature, 2 * i + 2)
        for category in categories:
            sns.distplot(
                a=data.loc[data[feature] == category, target_variable],
                bins=30,
                label=category,
                ax=ax,
            )
        ax.grid(alpha=0.3, linestyle="--")
        ax.legend()

    plt.tight_layout()
    plt.show()


def regression_plot(
    data: pd.DataFrame,
    features: List[str],
    target_variable: str,
    sample_size: Optional[int] = None,
) -> None:
    """Function that generates scatterplots for features vs. target variables, along with correlation coefficient.

    Args:
        data (pd.DataFrame): Dataframe containing features and the target variable.
        features (List[str]): List of numeric features to be considered.
        target_variable (str): The name of the target variable in the dataframe.
        sample_size (Optional[int], optional): Sample size for computating efficiency, in case data size is large. Defaults to None.
    """
    data_ = data.copy()
    if sample_size:
        data_ = data.sample(n=sample_size, random_state=99)

    n_features = len(features)
    _ = plt.figure(figsize=(12, n_features * 4))

    for i, feature in tqdm(enumerate(features)):
        ax = plt.subplot(n_features, 1, i + 1)
        corr = np.corrcoef(x=data_[feature], y=data_[target_variable])
        sns.regplot(
            data=data_,
            x=feature,
            y=target_variable,
            scatter_kws={"alpha": 0.3},
            line_kws={"color": "red"},
            ax=ax,
        )
        ax.set_title(f"{feature.upper()} - Correlation = {corr:.2f}")
        ax.grid(alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.show()


def correlation_heatmap(
    data: pd.DataFrame, features: List[str], target_variable: str
) -> None:
    """Function that plots a heatmap for the correlation coefficients among features and target variable.

    Args:
        data (pd.DataFrame): Dataframe containing features and the target variable.
        features (List[str]): List of numeric features to be considered.
        target_variable (str): The name of the target variable in the dataframe.
    """
    corr_matrix = data.loc[:, [target_variable] + features].corr()
    mask = np.ones_like(corr_matrix)
    mask[np.tril_indices_from(corr_matrix, k=-1)] = 0.0
    _, ax = plt.subplots(1, 1, figsize=(10, 8))
    sns.heatmap(
        data=corr_matrix,
        mask=mask,
        linewidths=1,
        vmax=1.0,
        vmin=-1.0,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        ax=ax,
    )
    plt.show()
