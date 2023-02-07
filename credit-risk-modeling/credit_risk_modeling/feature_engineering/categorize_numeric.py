__all__ = ["NumericCategoriesCreator"]

from typing import Any, Dict, List, Optional, Union
import pandas as pd

from credit_risk_modeling.base.estimator import BaseTransformer
from credit_risk_modeling.exceptions.cleaning import FieldNotFound


class NumericCategoriesCreator(BaseTransformer):
    """Class that creates One-Hot-Encoded features, on top of an existing continuous variable.

    Parameters:
        field_name (str): Name of field containing current categories.
        boundaries (List[Union[int, float]]): Boundaries to consider for each different bucket
    """

    def __init__(
        self,
        field_name: str,
        boundaries: List[Union[int, float]],
    ) -> None:
        """Constructor method for NumericCategoriesCreator class

        Args:
            field_name (str): Name of field containing current categories.
            boundaries (List[Union[int, float]]): Boundaries to consider for each different bucket
        """
        super(NumericCategoriesCreator, self).__init__()
        self._field_name = field_name
        self._boundaries = boundaries

    @property
    def field_name(self) -> str:
        """(str) Name of field containing current categories"""
        return self._field_name

    @property
    def boundaries(self) -> List[Union[int, float]]:
        """(datetime) Reference date against which time elapsed is compute"""
        return self._boundaries

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
        for lower, upper in zip(self._boundaries, self._boundaries[1:]):
            X[f"{self._field_name}_{lower-1}-{upper}"] = (
                (X[self._field_name] > lower) & (X[self._field_name] <= upper)
            ).astype(float)
        X = X.drop(columns=[self._field_name])
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
            raise FieldNotFound(f"{self._field_name} is not in the dataframe!")
        for lower, upper in zip(self._boundaries, self._boundaries[1:]):
            X[f"{self._field_name}_{lower-1}-{upper}"] = float(
                (X[self._field_name] > lower) and (X[self._field_name] <= upper)
            )
        del X[self._field_name]
        return X
