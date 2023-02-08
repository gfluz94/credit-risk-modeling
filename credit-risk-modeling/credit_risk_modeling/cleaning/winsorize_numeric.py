__all__ = ["NumericWinsorizer"]

from typing import Any, Dict, Optional
import pandas as pd
import numpy as np

from credit_risk_modeling.base.estimator import BaseTransformer
from credit_risk_modeling.exceptions.cleaning import FieldNotFound


class NumericWinsorizer(BaseTransformer):
    """Class that winsorizes a numeric value, so that we can clip extreme values.

    Parameters:
        field_name (str): Name of field containing current categories.
        lower
    """

    def __init__(
        self,
        field_name: str,
        lower: Optional[float] = None,
        upper: Optional[float] = None,
    ) -> None:
        """Constructor method for NumericWinsorizer class

        Args:
            field_name (str): Name of field containing current categories.
            lower (float, optional): Lower value to be considered. Defaults to None.
            upper (float, optional): Upper value to be considered. Defaults to None.
        """
        super(NumericWinsorizer, self).__init__()
        self._field_name = field_name
        self._lower = lower
        self._upper = upper

    @property
    def lower(self) -> Optional[float]:
        """(float) Lower value to be considered"""
        return self._lower

    @property
    def upper(self) -> Optional[float]:
        """(float) Lower value to be considered"""
        return self._upper

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
        X[self._field_name] = np.clip(
            X[self._field_name], a_min=self._lower, a_max=self._upper
        )
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
        if self._field_name not in X.keys():
            raise FieldNotFound(f"{self._field_name} is not in the dictionary!")
        if self._upper:
            X[self._field_name] = min(self._upper, X[self._field_name])
        if self._lower:
            X[self._field_name] = max(self._lower, X[self._field_name])
        return X
