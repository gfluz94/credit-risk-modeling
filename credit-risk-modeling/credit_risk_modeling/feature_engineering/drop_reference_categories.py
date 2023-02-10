__all__ = ["ReferenceCategoriesDropper"]

from typing import Any, Dict, List, Optional
import pandas as pd

from credit_risk_modeling.base.estimator import BaseTransformer
from credit_risk_modeling.exceptions.cleaning import FieldNotFound


class ReferenceCategoriesDropper(BaseTransformer):
    """Class that removes reference categories from the final dataframe, before it is fed to the actual model.

    Parameters:
        reference_categories (List[str]): List of reference categories.
    """

    def __init__(
        self,
        reference_categories: List[str],
    ) -> None:
        """Constructor method for ReferenceCategoriesDropper class

        Args:
            reference_categories (List[str]): List of reference categories.
        """
        super(ReferenceCategoriesDropper, self).__init__()
        self._reference_categories = reference_categories

    @property
    def reference_categories(self) -> str:
        """(List[str]) List of reference categories"""
        return self._reference_categories

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
        if len(set(self._reference_categories) - set(X.columns)) > 0:
            raise FieldNotFound(
                "Some categories are not present in the final dataframe!"
            )
        return X.drop(columns=self._reference_categories)

    def _transform_dict(self, X: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Transform method for python dictionary

        Args:
            X (Dict[str, Any]): Input dictionary containing features

        Raises:
            FieldNotFound: Name of fields must be found in dictionary

        Returns:
            Dict[str, Any]: Output dictionary containing transformed features
        """
        if len(set(self._reference_categories) - set(X.keys())) > 0:
            raise FieldNotFound(
                "Some categories are not present in the final dictionary!"
            )
        for reference_category in self.reference_categories:
            del X[reference_category]
        return X
