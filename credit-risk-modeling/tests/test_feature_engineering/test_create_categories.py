from typing import Any, Dict
import pandas as pd
import numpy as np
import pytest

from credit_risk_modeling.feature_engineering import OHECategoriesCreator
from credit_risk_modeling.exceptions.base import DataTypeNotAllowedForTransformation
from credit_risk_modeling.exceptions.cleaning import FieldNotFound


class TestOHECategoriesCreator(object):
    _FINAL_CATEGORIES = {
        "A": ["A"],
        "B": ["B"],
        "C": ["C"],
        "D": ["D"],
        "E": ["E"],
        "F": ["F"],
        "G": ["G"],
    }

    def test_transform_raisesDataTypeNotAllowedForTransformation(self):
        base_transformer = OHECategoriesCreator(
            field_name="grade", final_categories_dict=self._FINAL_CATEGORIES
        )
        with pytest.raises(DataTypeNotAllowedForTransformation):
            base_transformer.transform(np.ones(shape=10))

    def test_transform_raisesFieldNotFound(self, dummy_input_dict: Dict[str, Any]):
        base_transformer = OHECategoriesCreator(
            field_name="NON_EXISTING_FIELD",
            final_categories_dict=self._FINAL_CATEGORIES,
        )
        with pytest.raises(FieldNotFound):
            base_transformer.transform(dummy_input_dict)

    def test_transform_outputsDictionary(self, dummy_input_dict: Dict[str, Any]):
        # OUTPUT
        base_transformer = OHECategoriesCreator(
            field_name="grade", final_categories_dict=self._FINAL_CATEGORIES
        )
        output = base_transformer.transform(dummy_input_dict)

        # EXPECTED
        expected = dummy_input_dict.copy()
        del expected["grade"]
        expected["grade_A"] = 0.0
        expected["grade_B"] = 1.0
        expected["grade_C"] = 0.0
        expected["grade_D"] = 0.0
        expected["grade_E"] = 0.0
        expected["grade_F"] = 0.0
        expected["grade_G"] = 0.0

        # ASSERT
        expected == output

    def test_transform_outputsDataFrame(self, dummy_input_dict: Dict[str, Any]):
        # OUTPUT
        base_transformer = OHECategoriesCreator(
            field_name="grade", final_categories_dict=self._FINAL_CATEGORIES
        )
        output = base_transformer.transform(pd.DataFrame([dummy_input_dict]))

        # EXPECTED
        expected = pd.DataFrame([dummy_input_dict])
        expected = expected.drop(columns=["grade"])
        expected["grade_A"] = 0.0
        expected["grade_B"] = 1.0
        expected["grade_C"] = 0.0
        expected["grade_D"] = 0.0
        expected["grade_E"] = 0.0
        expected["grade_F"] = 0.0
        expected["grade_G"] = 0.0

        # ASSERT
        pd.testing.assert_frame_equal(expected, output)
