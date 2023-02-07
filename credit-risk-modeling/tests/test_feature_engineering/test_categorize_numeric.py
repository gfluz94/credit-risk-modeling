from typing import Any, Dict
import pandas as pd
import numpy as np
import pytest

from credit_risk_modeling.feature_engineering import NumericCategoriesCreator
from credit_risk_modeling.exceptions.base import DataTypeNotAllowedForTransformation
from credit_risk_modeling.exceptions.cleaning import FieldNotFound


class TestNumericCategoriesCreator(object):
    _BOUNDARIES = [0, 3, 10, 20, 30]

    def test_transform_raisesDataTypeNotAllowedForTransformation(self):
        base_transformer = NumericCategoriesCreator(
            field_name="dti", boundaries=self._BOUNDARIES
        )
        with pytest.raises(DataTypeNotAllowedForTransformation):
            base_transformer.transform(np.ones(shape=10))

    def test_transform_raisesFieldNotFound(self, dummy_input_dict: Dict[str, Any]):
        base_transformer = NumericCategoriesCreator(
            field_name="NON_EXISTING_FIELD", boundaries=self._BOUNDARIES
        )
        with pytest.raises(FieldNotFound):
            base_transformer.transform(dummy_input_dict)

    def test_transform_outputsDictionary(self, dummy_input_dict: Dict[str, Any]):
        # OUTPUT
        base_transformer = NumericCategoriesCreator(
            field_name="dti", boundaries=self._BOUNDARIES
        )
        output = base_transformer.transform(dummy_input_dict)

        # EXPECTED
        expected = dummy_input_dict.copy()
        del expected["dti"]
        expected["dti_0-3"] = 0.0
        expected["dti_4-10"] = 0.0
        expected["dti_11-20"] = 0.0
        expected["dti_21-30"] = 1.0

        # ASSERT
        expected == output

    def test_transform_outputsDataFrame(self, dummy_input_dict: Dict[str, Any]):
        # OUTPUT
        base_transformer = NumericCategoriesCreator(
            field_name="dti", boundaries=self._BOUNDARIES
        )
        output = base_transformer.transform(pd.DataFrame([dummy_input_dict]))

        # EXPECTED
        expected = pd.DataFrame([dummy_input_dict])
        expected = expected.drop(columns=["dti"])
        expected["dti_0-3"] = 0.0
        expected["dti_4-10"] = 0.0
        expected["dti_11-20"] = 0.0
        expected["dti_21-30"] = 1.0

        # ASSERT
        pd.testing.assert_frame_equal(expected, output)
