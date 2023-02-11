__all__ = ["ZeroInflatedXGBoost"]

from typing import Union
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin
from xgboost import XGBClassifier, XGBRegressor

from credit_risk_modeling.exceptions.models import (
    NotFittedYet,
    ZerosNotPresentForZeroInflatedRegression,
)


class ZeroInflatedXGBoost(BaseEstimator, RegressorMixin):
    """Class that implements Zero Inflated Regression, on top of XGBoost classifier and regressor.
    Basically, it is a meta-model, where the first model is a classifier, which outputs how likely it is that the target is zero.
    If it predicts that it is different from zero, then we have our final regression model, on top of it.
    Parameters:
        xgboost_classifier (XGBClassifier): XGBoost classifier for model's first layer.
        xgboost_regressor (XGBRegressor): XGBoost regressor for final output.
    """

    def __init__(
        self,
        xgboost_classifier: XGBClassifier,
        xgboost_regressor: XGBRegressor,
    ) -> None:
        """_summary_
        Args:
            xgboost_classifier (XGBClassifier): XGBoost classifier for model's first layer.
            xgboost_regressor (XGBRegressor): XGBoost regressor for final output.
        """
        self._xgboost_classifier = xgboost_classifier
        self._xgboost_regressor = xgboost_regressor
        self._inflated_zero = 0
        self._classifier_fitted = False
        self._regressor_fitted = False

    @property
    def xgboost_classifier(self) -> XGBClassifier:
        """(XGBClassifier) XGBoost classifier for model's first layer."""
        return self._xgboost_classifier

    @property
    def xgboost_regressor(self) -> XGBRegressor:
        """(XGBRegressor) XGBoost regressor for final output."""
        return self._xgboost_regressor

    def fit(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray] = None):
        """Method that is invoked to train both models - and hence, the meta-model.
        Args:
            X (pd.DataFrame): Dataframe containing the features to be fed to the model
            y (Union[pd.Series, np.ndarray], optional): Target values
        Raises:
            ZerosNotPresentForZeroInflatedRegression: Exception raised when there are no zeros, hence regular regression should be used in this case.
            TypeError: Exception raised when y is neither a numpy array nor a pandas Series.
        """
        # 1 - Not zero; 0 - Zero
        y_classification = (y != self._inflated_zero) * 1.0

        if y_classification.mean() == 1:
            raise ZerosNotPresentForZeroInflatedRegression(
                "There are no zero values in target variable. In this scenario, carry out regular regression."
            )

        # TRAINING CLASSIFIER FIRST
        self._classifier_fitted = self._xgboost_classifier.fit(X, y_classification)
        self._classifier_fitted = True

        non_zero_indices = np.where(y_classification == 1)[0]

        if isinstance(y, pd.Series):
            y_regression = y.iloc[non_zero_indices]
        elif isinstance(y, np.ndarray):
            y_regression = y[non_zero_indices]
        else:
            raise TypeError("`y` needs to be either a pandas Series or a numpy array")

        # TRAINING REGRESSOR, ON TOP OF CLASSIFIER
        self._xgboost_regressor.fit(X.iloc[non_zero_indices, :], y_regression)
        self._regressor_fitted = True

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Method that accepts a dataset for inference purposes (after model training).
        Args:
            X (pd.DataFrame): Dataframe containing the features to be fed to the model.
        Raises:
            NotFittedYet: Exception raised when models haven't already been trained.
        Returns:
            np.ndarray: Array of predicted values
        """
        if not self._classifier_fitted or not self._regressor_fitted:
            raise NotFittedYet("Classifier and regressor need to be fitted first!")
        y_pred = np.zeros(len(X))
        predicted_zeros = self._xgboost_classifier.predict(X)
        non_zero_indices = np.where(predicted_zeros == 1)[0]
        y_pred[non_zero_indices] = self._xgboost_regressor.predict(
            X.iloc[non_zero_indices, :]
        )
        return y_pred
