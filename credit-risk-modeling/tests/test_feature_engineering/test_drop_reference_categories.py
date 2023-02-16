from typing import Any, Dict
import pandas as pd
import numpy as np
import pytest

from credit_risk_modeling.feature_engineering import ReferenceCategoriesDropper
from credit_risk_modeling.exceptions.base import DataTypeNotAllowedForTransformation
from credit_risk_modeling.exceptions.cleaning import FieldNotFound


class TestReferenceCategoriesDropper(object):
    _REFERENCE_CATEGORIES = [
        "emp_length",
        "home_ownership",
    ]

    def test_transform_raisesDataTypeNotAllowedForTransformation(self):
        base_transformer = ReferenceCategoriesDropper(
            reference_categories=self._REFERENCE_CATEGORIES
        )
        with pytest.raises(DataTypeNotAllowedForTransformation):
            base_transformer.transform(np.ones(shape=10))

    def test_transform_raisesFieldNotFound(self, dummy_input_dict: Dict[str, Any]):
        base_transformer = ReferenceCategoriesDropper(
            reference_categories=self._REFERENCE_CATEGORIES + ["NON_EXISTING_FIELD"]
        )
        with pytest.raises(FieldNotFound):
            base_transformer.transform(dummy_input_dict)

    def test_transform_outputsDictionary(self, dummy_input_dict: Dict[str, Any]):
        # OUTPUT
        base_transformer = ReferenceCategoriesDropper(
            reference_categories=self._REFERENCE_CATEGORIES
        )
        output = base_transformer.transform(dummy_input_dict)

        # EXPECTED
        expected = dummy_input_dict.copy()
        del expected["emp_length"]
        del expected["home_ownership"]
        expected = {k: v for k, v in expected.items()}

        # ASSERT
        expected == output

    def test_transform_outputsDataFrame(self, dummy_input_dict: Dict[str, Any]):
        # OUTPUT
        base_transformer = ReferenceCategoriesDropper(
            reference_categories=self._REFERENCE_CATEGORIES
        )
        output = base_transformer.transform(pd.DataFrame([dummy_input_dict]))

        # EXPECTED
        expected = pd.DataFrame([dummy_input_dict])
        expected = expected.drop(columns=["emp_length", "home_ownership"])
        expected = expected.loc[:, sorted(expected.columns)]

        # ASSERT
        pd.testing.assert_frame_equal(expected, output)
