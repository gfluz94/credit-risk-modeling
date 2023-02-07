__all__ = ["get_fine_classes"]

from typing import Tuple
import numpy as np
import pandas as pd


def get_fine_classes(s: pd.Series, n_buckets: int) -> Tuple[pd.Series, Tuple[int, int]]:
    """Function that generates categorical classes on top of continuous variables

    Args:
        s (pd.Series): Series containing continuous values
        n_buckets (int): Number of equally-sizes buckets

    Returns:
        Tuple[pd.Series, Tuple[int, int]]: Tuple containing series with final categories and (lower, upper) boundaries
    """
    s_intervals = pd.cut(s, bins=n_buckets)
    s_ordered = pd.cut(s, bins=n_buckets, labels=np.arange(n_buckets))
    categories = s_intervals.cat.categories
    min_value = int(categories[0].left)
    max_value = int(categories[-1].right) + 1

    output = pd.concat([s_ordered, s_intervals.astype(str)], axis=1)
    output[f"{s.name}_categories"] = output.apply(
        lambda x: f"{x.iloc[0]:02d}. {x.iloc[-1]}", axis=1
    )
    return output[f"{s.name}_categories"], (min_value, max_value)
