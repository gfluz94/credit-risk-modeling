__all__ = ["OHECategoriesCreator"]

from typing import Any, Dict, Optional
import pandas as pd

from credit_risk_modeling.base.estimator import BaseTransformer
from credit_risk_modeling.exceptions.cleaning import FieldNotFound


class OHECategoriesCreator(BaseTransformer):
    """Class that creates One-Hot-Encoded features, on top of an existing categorical variable.

    Parameters:
        field_name (str): Name of field containing current categories.
        final_categories_dict (Dict[str, str]): Mapping from final category to groupped categories.
    """

    def __init__(
        self,
        field_name: str,
        final_categories_dict: Dict[str, str],
    ) -> None:
        """Constructor method for OHECategoriesCreator class

        Args:
            field_name (str): Name of field containing current categories.
            final_categories_dict (Dict[str, str]): Mapping from final category to groupped categories.
        """
        super(OHECategoriesCreator, self).__init__()
        self._field_name = field_name
        self._final_categories_dict = final_categories_dict

    @property
    def field_name(self) -> str:
        """(str) Name of field containing current categories"""
        return self._field_name

    @property
    def final_categories_dict(self) -> Dict[str, str]:
        """(Dict[str, str]): Mapping from final category to groupped categories"""
        return self._final_categories_dict

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
        for final_category, categories in self._final_categories_dict.items():
            X[f"{self._field_name}_{final_category}"] = (
                X[self._field_name].isin(categories).astype(float)
            )
        X = X.drop(columns=[self._field_name])
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
        for final_category, categories in self._final_categories_dict.items():
            X[f"{self._field_name}_{final_category}"] = 1.0 * (
                X[self._field_name] in categories
            )
        del X[self._field_name]
        return X
