__all__ = ["convert_probabilities_to_scores"]

import numpy as np


def convert_probabilities_to_scores(y_proba: np.ndarray) -> np.ndarray:
    """Function to convert probabilities into credit scores within (220, 750).
    Args:
        y_proba (np.ndarray): Array containing output probabilities.
    Returns:
        np.ndarray: Array containing corresponding credit scores
    """
    double_decrease_factor = 20 / np.log(2)
    constant = 600 - np.log(50) * double_decrease_factor
    y_proba = np.clip(y_proba, 1e-4, 0.9999)
    return constant - np.log(y_proba / (1 - y_proba)) * double_decrease_factor
