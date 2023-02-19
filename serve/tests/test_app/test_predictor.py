from serve.app.predictor import PredictorService


class TestPredictorService(object):

    service = PredictorService(
        models_path="models/artifacts",
    )

    def test_predict(
        self,
    ):
        # INPUT
        prediction_input_json = {
            "id": "sep",
            "member_id": "mv",
            "funded_amnt": 20_000,
            "term": "36 months",
            "int_rate": 10,
            "grade": "B",
            "emp_length": "3 years",
            "home_ownership": "MORTGAGE",
            "annual_inc": 80_000,
            "issue_d": "May-15",
            "purpose": "medical",
            "addr_state": "PA",
            "dti": 7.5,
            "earliest_cr_line": "Apr-10",
            "verification_status": "Not Verified",
            "initial_list_status": "w",
            "inq_last_6mths": 0,
            "total_rev_hi_lim": 25_000,
            "delinq_2yrs": 0,
            "pub_rec": 0,
            "open_acc": 5,
            "total_acc": 20,
        }

        # OUTPUT
        output = self.service.predict(prediction_input_json)
        print(output)

        # EXPECTED
        expected = {
            "PD": 0.37194486655999126,
            "LGD": 1.0,
            "EAD": 0.8873217701911926,
            "EL": 6600.695548190767,
        }

        # ASSERT
        assert output == expected
