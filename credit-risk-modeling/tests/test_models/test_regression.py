import numpy as np
import pandas as pd
import pytest

from xgboost import XGBClassifier, XGBRegressor
from credit_risk_modeling.models.regression import ZeroInflatedXGBoost
from credit_risk_modeling.exceptions.models import (
    NotFittedYet,
    ZerosNotPresentForZeroInflatedRegression,
)


class TestZeroInflatedXGBoost(object):
    def test_fitRaisesZerosNotPresentForZeroInflatedRegression(
        self, model_input_no_zeros: pd.DataFrame
    ):
        X, y = model_input_no_zeros.iloc[:, :-1], model_input_no_zeros.iloc[:, -1]
        model = ZeroInflatedXGBoost(
            xgboost_classifier=XGBClassifier(),
            xgboost_regressor=XGBRegressor(),
        )

        with pytest.raises(ZerosNotPresentForZeroInflatedRegression):
            model.fit(X, y)

    def test_fitRaisesNotFittedYet(self, model_input_no_zeros: pd.DataFrame):
        X = model_input_no_zeros.iloc[:, :-1]
        model = ZeroInflatedXGBoost(
            xgboost_classifier=XGBClassifier(),
            xgboost_regressor=XGBRegressor(),
        )

        with pytest.raises(NotFittedYet):
            model.predict(X)

    def test_predictReturnsExpectedOutput(
        self,
        model_input: pd.DataFrame,
    ):
        # OUTPUT
        model = ZeroInflatedXGBoost(
            xgboost_classifier=XGBClassifier(),
            xgboost_regressor=XGBRegressor(),
        )
        X = model_input.iloc[:, :-1]
        y = model_input.iloc[:, -1]
        model.fit(X, y)
        output = model.predict(X)

        # EXPECTED
        expected = np.zeros(3)

        # ASSERT
        np.testing.assert_array_almost_equal(expected, output)
