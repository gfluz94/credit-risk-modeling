__all__ = ["BaseTransformer"]

from typing import Any, Dict, Optional, Union
from abc import ABC, abstractmethod
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

from credit_risk_modeling.exceptions.base import DataTypeNotAllowedForTransformation


class BaseTransformer(ABC, BaseEstimator, TransformerMixin):
    """Abstract Base Class for transformers that will be created for specific operations to the data.

    Raises:
        DataTypeNotAllowedForTransformation: Once used for transformation, this class only accepts either pandas.DataFrame or dictionary.
    """

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Method to fit transformer.

        Args:
            X (pd.DataFrame): Dataframe containing features.
            y (Optional[pd.Series], optional): In case, target variable needs to be used in the logic. Defaults to None.
        """
        pass

    @abstractmethod
    def _transform_dict(self, X: Dict[str, Any]) -> Dict[str, Any]:
        """Method to apply transformations to a dictionary.

        Args:
            X (Dict[str, Any]): Dictionary containing features.

        Returns:
            Dict[str, Any]: Dictionary with trasformartions applied to it.
        """
        pass

    @abstractmethod
    def _transform_df(self, X: pd.DataFrame) -> pd.DataFrame:
        """Method to apply transformations to a pandas DataFrame.

        Args:
            X (pd.DataFrame): Dataframe containing features.

        Returns:
            pd.DataFrame: Dataframe with trasformartions applied to it.
        """
        pass

    def transform(
        self, X: Union[pd.DataFrame, Dict[str, Any]]
    ) -> Union[pd.DataFrame, Dict[str, Any]]:
        """Method to wrap input type specific transformation methods.

        Args:
            X (Union[pd.DataFrame, Dict[str, Any]]): Dataframe or dictionary containing features.

        Raises:
            DataTypeNotAllowedForTransformation: Argument needs to be either a pandas.DataFrame or a python dictionary.

        Returns:
            Union[pd.DataFrame, Dict[str, Any]]: Dataframe or dictionary with applied transformations to them.
        """
        if isinstance(X, pd.DataFrame):
            return self._transform_df(X.copy())
        elif isinstance(X, dict):
            return self._transform_dict(X.copy())
        else:
            raise DataTypeNotAllowedForTransformation(
                "Accepted types are either a pandas.DataFrame or a python dictionary"
            )
