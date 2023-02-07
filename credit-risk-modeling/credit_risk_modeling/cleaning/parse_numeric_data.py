__all__ = ["NumericExtractor"]

from typing import Any, Dict, List, Optional
import re
import pandas as pd

from credit_risk_modeling.base import BaseTransformer
from credit_risk_modeling.exceptions.cleaning import FieldNotFound


class NumericExtractor(BaseTransformer):
    """Class that parses text into numeric data format

    Parameters:
        field_name (str): Name of field that requires parsing
        regex_extraction (str): Regex pattern for information extraction
    """

    def __init__(
        self,
        field_name: str,
        regex_extraction: str,
        post_mapping: Optional[Dict[str, str]] = None,
    ) -> None:
        """Constructor method for DatetimeConverter class

        Args:
            field_name (str): Name of field that requires parsing
            regex_extraction (str): Regex pattern for information extraction
            post_mapping (Optional[Dict[str, str]]): Mapping after extraction. Defaults to None.
        """
        super(NumericExtractor, self).__init__()
        self._field_name = field_name
        self._regex_extraction = regex_extraction
        self._post_mapping = post_mapping

    @property
    def field_name(self) -> List[str]:
        """(str) Name of field that requires parsing"""
        return self._field_name

    @property
    def regex_extraction(self) -> str:
        """(str) Regex pattern for information extraction"""
        return self._regex_extraction

    @property
    def post_mapping(self) -> Optional[Dict[str, str]]:
        """(Optional[Dict[str, str]]) Mapping after extraction"""
        return self._post_mapping

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
        X.loc[~X[self._field_name].isnull(), self._field_name] = (
            X.loc[~X[self._field_name].isnull(), self._field_name]
            .astype(str)
            .str.extract(self._regex_extraction)
            .iloc[:, 0]
        )
        if self._post_mapping:
            for pattern, replacement in self._post_mapping.items():
                X.loc[~X[self._field_name].isnull(), self._field_name] = X.loc[
                    ~X[self._field_name].isnull(), self._field_name
                ].str.replace(pattern, replacement, regex=True)
        X.loc[~X[self._field_name].isnull(), self._field_name] = X.loc[
            ~X[self._field_name].isnull(), self._field_name
        ].astype(int)
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
        if X[self._field_name]:
            X[self._field_name] = re.findall(
                pattern=self._regex_extraction, string=X[self._field_name]
            )[0]
            if self._post_mapping:
                for pattern, replacement in self._post_mapping.items():
                    X[self._field_name] = re.sub(
                        pattern=pattern, repl=replacement, string=X[self._field_name]
                    )
        return X
