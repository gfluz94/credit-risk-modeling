__all__ = ["DatetimeConverter"]

from typing import Any, Dict, List, Optional
from datetime import datetime
import pandas as pd

from credit_risk_modeling.base import BaseTransformer
from credit_risk_modeling.exceptions.cleaning import FieldNotFound


class DatetimeConverter(BaseTransformer):
    """Class that transforms data records from loan data into datetime format

    Parameters:
        field_names (List[str]): Name of fields that require conversion to datetime format
        datetime_format (str): Datetime format from input
    """

    def __init__(self, field_names: List[str], datetime_format: str) -> None:
        """Constructor method for DatetimeConverter class

        Args:
            field_names (List[str]): Name of fields that require conversion to datetime format
            datetime_format (str): Datetime format from input
        """
        super(DatetimeConverter, self).__init__()
        self._field_names = field_names
        self._datetime_format = datetime_format

    @property
    def field_names(self) -> List[str]:
        """(List[str]) Name of fields that require conversion to datetime format"""
        return self._field_names

    @property
    def datetime_format(self) -> str:
        """(str) Datetime format from input"""
        return self._datetime_format

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit for the transformation

        Args:
            X (pd.DataFrame): Input dataframe containing features
            y (Optional[pd.Series], optional): Target variables. Defaults to None.
        """
        return self

    def _transform_df(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform method for pandas.DataFrame

        Args:
            X (pd.DataFrame): Input dataframe containing features

        Raises:
            FieldNotFound: Name of fields must be found in dataframe

        Returns:
            pd.DataFrame: Output daframe containing transformed features
        """
        for field_name in self._field_names:
            if field_name not in X.columns:
                raise FieldNotFound(f"{field_name} is not in the dataframe!")
            X[field_name] = pd.to_datetime(X[field_name], format=self._datetime_format)
        return X

    def _transform_dict(self, X: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Transform method for python dictionary

        Args:
            X (Dict[str, Any]): Input dictionary containing features

        Raises:
            FieldNotFound: Name of fields must be found in dictionary

        Returns:
            Dict[str, Any]: Output dictionary containing transformed features
        """
        for field_name in self._field_names:
            if field_name not in X.keys():
                raise FieldNotFound(f"{field_name} is not in the dictionary keys!")
            X[field_name] = datetime.strptime(X[field_name], self._datetime_format)
        return X
