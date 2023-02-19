from typing import Any, Dict, Tuple
from enum import Enum, auto
from functools import reduce
import os
import pandas as pd
import dill

from sklearn.base import BaseEstimator


class TargetVariable(Enum):
    pd = auto()
    lgd = auto()
    ead = auto()


class PredictorService(object):
    """Trained models wrapper to predict PD, LGD and EAD during inference serving time.
    Parameters:
        models_path (str, optional): Path within Docker container where models are located. Defaults to "/app/models".
    """

    _LOAN_AMOUNT_FIELD = "funded_amnt"

    def __init__(self, models_path: str = "models"):
        """Constructor method for PredictorService
        Args:
            models_path (str, optional): Path within Docker container where models are located. Defaults to "/app/models".
        """
        self._models_path = models_path

    @property
    def models_path(self) -> str:
        """(str) Path within Docker container where models are located."""
        return self._models_path

    def _get_models(self) -> Dict[str, Tuple[BaseEstimator]]:
        """Method that allows for loading and instantiation of trained models, like a singleton.
        Returns:
            Dict[str, Tuple[BaseEstimator]]: Preprocessor and model (value) for each target name (key)
        """
        if not hasattr(self, "_models"):
            self._models = {}
            for target in list(TargetVariable.__members__.keys()):
                with open(
                    os.path.join(self._models_path, f"{target}_preprocessing.pkl"), "rb"
                ) as f:
                    preprocessor = dill.load(f)
                with open(
                    os.path.join(self._models_path, f"{target}_model.pkl"), "rb"
                ) as f:
                    model = dill.load(f)
                self._models[target] = (preprocessor, model)

        return self._models

    def _get_cleaner(self) -> BaseEstimator:
        """Method that allows for loading and instantiation of common cleaning pipeline, like a singleton.
        Returns:
            BaseEstimator: Cleaning pipeline object
        """
        if not hasattr(self, "_cleaner"):
            with open(os.path.join(self._models_path, "cleaner.pkl"), "rb") as f:
                self._cleaner = dill.load(f)
        return self._cleaner

    def _clean_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Method to clean raw request
        Args:
            request (Dict[str, Any]): Raw request coming from server
        Returns:
            Dict[str, Any]: Request with cleaned and newly created fields
        """
        earliest_cr_line = "earliest_cr_line"
        issue_d = "issue_d"
        if request[earliest_cr_line] is None:
            request[earliest_cr_line] = request[issue_d]
        return self._get_cleaner().transform(request)

    def _convert_to_pandas(self, request: Dict[str, Any]) -> pd.DataFrame:
        """Method to convert raw request to a dataframe
        Args:
            request (Dict[str, Any]): Request containing features to be fed to the model
        Returns:
            pandas.DataFrame: Dataframe containing features
        """
        df = pd.DataFrame([request])
        return df.astype(float)

    def _post_processor(
        self, prediction: float, target_variable: TargetVariable
    ) -> float:
        """Method to post-process predicted values from available models.
        It clips the value within the range [0, 1], and for LGD we need (1 - prediction).
        Args:
            prediction (float): Predicted value from the model
            target_variable (TargetVariable): Target value we are currently predicting (PD/LGD/EAD)
        Returns:
            float: Final predicted value
        """
        prediction = min(prediction, 1.0)
        prediction = max(prediction, 0.0)
        if TargetVariable.__members__[target_variable] == TargetVariable.lgd:
            return 1 - prediction
        return prediction

    def predict(self, request_inputs: Dict[str, Any]) -> Dict[str, float]:
        """Method for prediction whenever a request invokes the model in production.
        Args:
            request_inputs (Dict[str, Any]): Raw input from the server
        Returns:
            Dict[str, float]: Final predictions for PD, LGD, EAD and EL.
        """
        predictions = {}
        cleaned_inputs = self._clean_request(request_inputs)
        for target in list(TargetVariable.__members__.keys()):
            preprocessor, model = self._get_models()[target]
            preprocessed_inputs = preprocessor.transform(cleaned_inputs)
            inputs = self._convert_to_pandas(preprocessed_inputs)
            if hasattr(model, "predict_proba"):
                prediction = model.predict_proba(inputs)[:, 1].tolist()[0]
            else:
                prediction = model.predict(inputs).tolist()[0]
            predictions[target.upper()] = self._post_processor(
                prediction, target_variable=target
            )
        predictions["EL"] = (
            reduce(lambda a, b: a * b, predictions.values())
            * request_inputs[self._LOAN_AMOUNT_FIELD]
        )
        return predictions
