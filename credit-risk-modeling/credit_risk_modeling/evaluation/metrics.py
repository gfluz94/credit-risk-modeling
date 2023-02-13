__all__ = ["get_classification_metrics_across_thresholds", "get_regression_metrics"]

from enum import Enum, auto
from typing import Dict
import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    recall_score,
    precision_score,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


class RegressionMetric(Enum):
    R2 = auto()
    RMSE = auto()
    MAE = auto()


def get_classification_metrics_for_threshold(
    y_proba: np.ndarray, y_true: pd.Series, threshold: float = 0.5
) -> Dict[str, float]:
    """Function to return dictionary with classification metrics.
    (Recall, Precision, F1-Score, ROC-AUC, Average Precision)
    Args:
        y_proba (np.ndarray): Array containing probabilities predicted by the model
        y_true (pd.Series): Array containing true label values
        threshold (float, optional): Classification threshold for metric computation. Defaults to 0.5.
    Returns:
        Dict[str, float]: Metric name and corresponding metric value
    """
    y_pred = (y_proba > threshold) * 1.0
    n_predicted = y_pred.sum()
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    f1 = (
        None
        if (recall + precision) == 0
        else 2 * recall * precision / (recall + precision)
    )
    auc = roc_auc_score(y_true, y_proba)
    avg_p = average_precision_score(y_true, y_proba)
    return {
        "# DEFAULT": n_predicted,
        "RECALL": recall,
        "PRECISION": precision,
        "F1": f1,
        "ROC-AUC": auc,
        "AVERAGE PRECISION": avg_p,
    }


def get_classification_metrics_across_thresholds(
    y_proba: np.ndarray, y_true: pd.Series
) -> pd.DataFrame:
    """Function to return dataframe with classification metrics for thresholds within [0.05, 0.95].
    (Recall, Precision, F1-Score, ROC-AUC, Average Precision)
    Args:
        y_proba (np.ndarray): Array containing probabilities predicted by the model
        y_true (pd.Series): Array containing true label values
    Returns:
        pd.DataFrame: Metric name and corresponding metric value
    """
    thresholds = np.linspace(0.05, 0.95, 19)
    return pd.DataFrame(
        [
            get_classification_metrics_for_threshold(
                y_proba, y_true, threshold=threshold
            )
            for threshold in thresholds
        ],
        index=thresholds,
    )


def rmse_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Method that computes Root Mean Squared Error (RMSE).
    Args:
        y_true (np.ndarray): True target values
        y_pred (np.ndarray): Predicted Values
    Returns:
        float: Value of RMSE metric
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def get_regression_metrics(
    y_pred: np.ndarray,
    y_true: pd.Series,
) -> Dict[str, float]:
    """Function to return dictionary with regression metrics.
    (R2-Score, RMSE, MAE)
    Args:
        y_pred (np.ndarray): Array containing predicted values
        y_true (pd.Series): Array containing true values
    Returns:
        Dict[str, float]: Metric name and corresponding metric value
    """
    return {
        "# OBS": len(y_pred),
        RegressionMetric.RMSE.name: rmse_metric(y_true=y_true, y_pred=y_pred),
        RegressionMetric.R2.name: r2_score(y_true=y_true, y_pred=y_pred),
        RegressionMetric.MAE.name: mean_absolute_error(y_true=y_true, y_pred=y_pred),
    }
