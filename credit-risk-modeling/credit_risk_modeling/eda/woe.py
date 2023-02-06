__all__ = ["compute_woe", "plot_woe_by_category"]

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def compute_woe(
    df: pd.DataFrame,
    id_column_name: str,
    feature_column_name: str,
    target_column_name: str,
    sort_by_woe: bool = True,
) -> pd.DataFrame:
    """Function that computes Weight of Evidence (WoE) for each category within a variable, based on target values.

    Args:
        df (pd.DataFrame): Input dataframe containing labels and target values.
        id_column_name (str): Name of column containing loan IDs.
        feature_column_name (str): Name of column containing feature.
        target_column_name (str): Name of column containing target variable values.
        sort_by_woe (bool, optional): Whether or not sorting by WoE - for numeric features we sort by the actual categories. Defaults to True.

    Returns:
        pd.DataFrame: DataFrame containing WoE for each one of feature's categories along with Information Valuye (IV)
    """

    # GENERATE CONTINGENCY MATRIX
    matrix = pd.pivot_table(
        data=df,
        index=feature_column_name,
        columns=target_column_name,
        values=id_column_name,
        aggfunc=pd.Series.count,
    )
    label_columns = ["Non-Default", "Default"]
    matrix.columns = label_columns

    # COMPUTE WOE
    matrix["Total Obs"] = matrix.sum(axis=1)
    for label in label_columns:
        matrix[f"% {label}"] = matrix[label] / matrix[label].sum()
    matrix["WoE"] = np.log(
        matrix[f"% {label_columns[1]}"] / matrix[f"% {label_columns[0]}"]
    )

    # COMPUTE IV
    matrix["IV"] = (
        matrix["WoE"]
        * (matrix[f"% {label_columns[1]}"] - matrix[f"% {label_columns[0]}"])
    ).sum()

    # SORT BY WOE OR FEATURE VALUE
    if sort_by_woe:
        matrix = matrix.sort_values(by="WoE")
    return matrix


def plot_woe_by_category(df: pd.DataFrame, rotate: bool = False) -> None:
    """Function that plots WoE for each one of feature's categories.
    The main purpose is to help us decide on categories to consider/aggregate.

    Args:
        df (pd.DataFrame): Dataframe containing WoE values for a feature.
        rotate (bool, optional): Whether or not categories should be displayed with 90º-rotation. Defaults to False.
    """
    woe_column = "WoE"
    _, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(df.index, df[woe_column], "o--", color="black")
    ax.set_xlabel(df.index.name.capitalize())
    if rotate:
        plt.xticks(rotation=90)
    ax.set_ylabel(woe_column)
    plt.grid(alpha=0.3, linestyle="--")
    plt.show()
