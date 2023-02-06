from typing import Any, Dict
from datetime import datetime
import pandas as pd
import numpy as np
import pytest

from credit_risk_modeling.feature_engineering import TimeSinceCalculator
from credit_risk_modeling.exceptions.base import DataTypeNotAllowedForTransformation
from credit_risk_modeling.exceptions.cleaning import FieldNotFound
from credit_risk_modeling.exceptions.feature_engineering import TimeUnitNotAvailable


class TestTimeSinceCalculator(object):
    _REFERENCE_DATE = datetime(2017, 12, 1)

    def test_transform_raisesDataTypeNotAllowedForTransformation(self):
        base_transformer = TimeSinceCalculator(
            field_name="earliest_cr_line",
            reference_date=self._REFERENCE_DATE,
            time_unit="month",
            winsorize_max=True,
        )
        with pytest.raises(DataTypeNotAllowedForTransformation):
            base_transformer.transform(np.ones(shape=10))

    def test_transform_raisesFieldNotFound(self, time_since_input_dict: Dict[str, Any]):
        base_transformer = TimeSinceCalculator(
            field_name="NON_EXISTING_FIELD",
            reference_date=self._REFERENCE_DATE,
            time_unit="month",
            winsorize_max=True,
        )
        with pytest.raises(FieldNotFound):
            base_transformer.transform(time_since_input_dict)

    def test_transform_raisesTimeUnitNotAvailable(
        self
    ):
        with pytest.raises(TimeUnitNotAvailable):
            _ = TimeSinceCalculator(
                field_name="earliest_cr_line",
                reference_date=self._REFERENCE_DATE,
                time_unit="second",
                winsorize_max=True,
            )

    def test_transform_outputsDictionary(self, time_since_input_dict: Dict[str, Any]):
        # OUTPUT
        base_transformer = TimeSinceCalculator(
            field_name="earliest_cr_line",
            reference_date=self._REFERENCE_DATE,
            time_unit="month",
            winsorize_max=True,
        )
        output = base_transformer.transform(time_since_input_dict)

        # EXPECTED
        expected = time_since_input_dict.copy()
        expected["months_since_earliest_cr_line"] = 274.9953797819257

        # ASSERT
        expected == output

    def test_transform_outputsDataFrame(self, time_since_input_dict: Dict[str, Any]):
        # OUTPUT
        base_transformer = TimeSinceCalculator(
            field_name="earliest_cr_line",
            reference_date=datetime(2017, 12, 1),
            time_unit="month",
            winsorize_max=True,
        )
        output = base_transformer.transform(pd.DataFrame([time_since_input_dict]))

        # EXPECTED
        expected = pd.DataFrame([time_since_input_dict])
        expected["months_since_earliest_cr_line"] = 274.9953797819257

        # ASSERT
        pd.testing.assert_frame_equal(expected, output)
