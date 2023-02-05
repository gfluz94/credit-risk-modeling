__all__ = ["TimeSinceCalculator"]

from typing import Any, Dict, Optional
from enum import Enum
from datetime import datetime
import pandas as pd
import numpy as np

from credit_risk_modeling.base.estimator import BaseTransformer
from credit_risk_modeling.exceptions.cleaning import FieldNotFound
from credit_risk_modeling.exceptions.feature_engineering import TimeUnitNotAvailable


class TimeUnit(Enum):
    day = "D"
    month = "M"
    year = "Y"


class TimeSinceCalculator(BaseTransformer):
    """Class that computes total time elapsed, according to a reference date

    Parameters:
        field_name (str): Name of field that requires parsing
        reference_date (datetime): Reference date against which time elapsed is computed
        time_unit (str, optional): Output's unit of time. Defaults to "month".
        winsorize_max (bool, optional): Whether or not to winsorize extreme/odd values. Defaults to True.
    """

    def __init__(
        self,
        field_name: str,
        reference_date: datetime,
        time_unit: str = "month",
        winsorize_max: bool = True,
    ) -> None:
        """Constructor method for TimeSinceCalculator class

        Args:
            field_name (str): Name of field that requires parsing
            reference_date (datetime): Reference date against which time elapsed is computed
            time_unit (str, optional): Output's unit of time. Defaults to "month".
            winsorize_max (bool, optional): Whether or not to winsorize extreme/odd values. Defaults to True.
        """
        super(TimeSinceCalculator, self).__init__()
        self._field_name = field_name
        self._reference_date = reference_date
        if time_unit not in TimeUnit.__members__.keys():
            raise TimeUnitNotAvailable(
                f"{time_unit} not available. Units available are: {'/'.join(TimeUnit.__members__.keys())}"
            )
        self._time_unit = time_unit
        self._winsorize_max = winsorize_max
        self._output_name = f"{self._time_unit}s_since_{self._field_name}"

    @property
    def field_name(self) -> str:
        """(str) Name of field that requires parsing"""
        return self._field_name

    @property
    def reference_date(self) -> str:
        """(datetime) Reference date against which time elapsed is compute"""
        return self._reference_date

    @property
    def time_unit(self) -> str:
        """(str) Output's unit of time"""
        return self._time_unit

    @property
    def winsorize_max(self) -> str:
        """(bool) Whether or not to winsorize extreme/odd values"""
        return self._winsorize_max

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
        if self._field_name not in X.columns:
            raise FieldNotFound(f"{self._field_name} is not in the dataframe!")
        X[self._output_name] = (
            self._reference_date - X[self._field_name]
        ) / np.timedelta64(1, TimeUnit.__members__[self._time_unit].value)
        return X

    def _transform_dict(self, X: Dict[str, Any]) -> Dict[str, Any]:
        """Transform method for python dictionary

        Args:
            X (Dict[str, Any]): Input dictionary containing features

        Raises:
            FieldNotFound: Name of fields must be found in dictionary

        Returns:
            Dict[str, Any]: Output dictionary containing transformed features
        """
        if self._field_name not in X.keys():
            raise FieldNotFound(f"{self._field_name} is not in the dataframe!")
        X[self._output_name] = np.timedelta64(
            self._reference_date - X[self._field_name]
        ) / np.timedelta64(1, TimeUnit.__members__[self._time_unit].value).astype(
            "timedelta64[us]"
        )
        return X
