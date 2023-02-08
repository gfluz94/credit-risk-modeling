from typing import Any, Dict
import re
import pandas as pd
import numpy as np
import pytest

from credit_risk_modeling.cleaning import NumericWinsorizer
from credit_risk_modeling.exceptions.base import DataTypeNotAllowedForTransformation
from credit_risk_modeling.exceptions.cleaning import FieldNotFound


class TestNumericWinsorizer(object):
    def test_transform_raisesDataTypeNotAllowedForTransformation(self):
        base_transformer = NumericWinsorizer(
            field_name="int_rate", lower=None, upper=10.0
        )
        with pytest.raises(DataTypeNotAllowedForTransformation):
            base_transformer.transform(np.ones(shape=10))

    def test_transform_raisesFieldNotFound(self, dummy_input_dict: Dict[str, Any]):
        base_transformer = NumericWinsorizer(
            field_name="NON_EXISTING_FIELD", lower=None, upper=10.0
        )
        with pytest.raises(FieldNotFound):
            base_transformer.transform(dummy_input_dict)

    def test_transform_outputsDictionary(self, dummy_input_dict: Dict[str, Any]):
        # OUTPUT
        base_transformer = NumericWinsorizer(
            field_name="int_rate", lower=None, upper=10.0
        )
        output = base_transformer.transform(dummy_input_dict)

        # EXPECTED
        expected = dummy_input_dict.copy()
        expected["int_rate"] = 10.0

        # ASSERT
        expected == output

    def test_transform_outputsDataFrame(self, dummy_input_dict: Dict[str, Any]):
        # OUTPUT
        base_transformer = NumericWinsorizer(
            field_name="int_rate", lower=None, upper=10.0
        )
        output = base_transformer.transform(pd.DataFrame([dummy_input_dict]))

        # EXPECTED
        expected = pd.DataFrame([dummy_input_dict])
        expected["int_rate"] = 10.0

        # ASSERT
        pd.testing.assert_frame_equal(expected, output)
