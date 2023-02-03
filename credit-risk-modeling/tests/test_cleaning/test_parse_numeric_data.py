from typing import Any, Dict
import re
import pandas as pd
import numpy as np
import pytest

from credit_risk_modeling.cleaning import NumericExtractor
from credit_risk_modeling.exceptions.base import DataTypeNotAllowedForTransformation
from credit_risk_modeling.exceptions.cleaning import FieldNotFound


class TestNumericExtractor(object):
    def test_transform_raisesDataTypeNotAllowedForTransformation(self):
        base_transformer = NumericExtractor(
            field_name="emp_length",
            regex_extraction=r"(.+)\syears?",
            post_mapping={r"10\+\s?": str(10), r"< 1\s?": str(0)},
        )
        with pytest.raises(DataTypeNotAllowedForTransformation):
            base_transformer.transform(np.ones(shape=10))

    def test_transform_raisesFieldNotFound(self, dummy_input_dict: Dict[str, Any]):
        base_transformer = NumericExtractor(
            field_name="NON_EXISTING_FIELD",
            regex_extraction=r"(.+)\syears?",
            post_mapping={r"10\+\s?": str(10), r"< 1\s?": str(0)},
        )
        with pytest.raises(FieldNotFound):
            base_transformer.transform(dummy_input_dict)

    def test_transform_outputsDictionary(self, dummy_input_dict: Dict[str, Any]):
        # OUTPUT
        base_transformer = NumericExtractor(
            field_name="emp_length",
            regex_extraction=r"(.+)\syears?",
            post_mapping={r"10\+\s?": str(10), r"< 1\s?": str(0)},
        )
        output = base_transformer.transform(dummy_input_dict)

        # EXPECTED
        expected = dummy_input_dict.copy()
        expected["emp_length"] = int(re.findall(r"(\d+)", expected["emp_length"])[0])

        # ASSERT
        expected == output

    def test_transform_outputsDataFrame(self, dummy_input_dict: Dict[str, Any]):
        # OUTPUT
        base_transformer = NumericExtractor(
            field_name="emp_length",
            regex_extraction=r"(.+)\syears?",
            post_mapping={r"10\+\s?": str(10), r"< 1\s?": str(0)},
        )
        output = base_transformer.transform(pd.DataFrame([dummy_input_dict]))

        # EXPECTED
        expected = pd.DataFrame([dummy_input_dict])
        expected["emp_length"] = (
            expected["emp_length"].str.extract(r"(\d+)").iloc[:, 0].astype(int)
        )

        # ASSERT
        pd.testing.assert_frame_equal(expected, output)
