from typing import Any, Dict
from datetime import datetime
import pandas as pd
import numpy as np
import pytest

from credit_risk_modeling.cleaning import DatetimeConverter
from credit_risk_modeling.exceptions.base import DataTypeNotAllowedForTransformation
from credit_risk_modeling.exceptions.cleaning import FieldNotFound


class TestDatetimeConverter(object):
    def test_transform_raisesDataTypeNotAllowedForTransformation(self):
        base_transformer = DatetimeConverter(
            field_names=["issue_d", "earliest_cr_line"], datetime_format="%b-%y"
        )
        with pytest.raises(DataTypeNotAllowedForTransformation):
            base_transformer.transform(np.ones(shape=10))

    def test_transform_raisesFieldNotFound(self, dummy_input_dict: Dict[str, Any]):
        base_transformer = DatetimeConverter(
            field_names=["NON_EXISTING_FIELD"], datetime_format="%b-%y"
        )
        with pytest.raises(FieldNotFound):
            base_transformer.transform(dummy_input_dict)

    def test_transform_outputsDictionary(self, dummy_input_dict: Dict[str, Any]):
        # OUTPUT
        base_transformer = DatetimeConverter(
            field_names=["issue_d", "earliest_cr_line"], datetime_format="%b-%y"
        )
        output = base_transformer.transform(dummy_input_dict)

        # EXPECTED
        expected = dummy_input_dict.copy()
        expected["issue_d"] = datetime.strptime(expected["issue_d"], "%b-%y")
        expected["earliest_cr_line"] = datetime.strptime(
            expected["earliest_cr_line"], "%b-%y"
        )

        # ASSERT
        expected == output

    def test_transform_outputsDataFrame(self, dummy_input_dict: Dict[str, Any]):
        # OUTPUT
        base_transformer = DatetimeConverter(
            field_names=["issue_d", "earliest_cr_line"], datetime_format="%b-%y"
        )
        output = base_transformer.transform(pd.DataFrame([dummy_input_dict]))

        # EXPECTED
        expected = pd.DataFrame([dummy_input_dict])
        expected["issue_d"] = pd.to_datetime(expected["issue_d"], format="%b-%y")
        expected["earliest_cr_line"] = pd.to_datetime(
            expected["earliest_cr_line"], format="%b-%y"
        )

        # ASSERT
        pd.testing.assert_frame_equal(expected, output)
