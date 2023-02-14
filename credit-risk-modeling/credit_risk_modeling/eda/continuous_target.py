__all__ = ["boxplot_by_categories"]

from typing import List
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def boxplot_by_categories(
    data: pd.DataFrame, features: List[str], target_variable: str
) -> None:
    """Function that generates boxplot for a continuous target variable, breaking it down by a categorical variable.

    Args:
        data (pd.DataFrame): Dataframe containing features and the target variable.
        features (List[str]): List of categorical featurs to be considered for the breakdown.
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
