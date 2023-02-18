from typing import Any, Dict
from app.predictor import PredictorService


class TestPredictorService(object):

    service = PredictorService(
        model_path="output",
    )

    def test_predict(
        self,
    ):
        # OUTPUT
        output = self.service.predict(prediction_input_json)

        # EXPECTED
        expected = expected_prediction_output

        # ASSERT
        assert output == expected
