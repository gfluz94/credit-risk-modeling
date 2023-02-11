from typing import Any, Dict
from datetime import datetime
import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def dummy_input_dict() -> Dict[str, Any]:
    return {
        "id": 1077501,
        "member_id": 1296599,
        "loan_amnt": 5000,
        "funded_amnt": 5000,
        "term": "36 months",
        "int_rate": 10.65,
        "installment": 162.87,
        "grade": "B",
        "emp_length": "10+ years",
        "home_ownership": "RENT",
        "annual_inc": 24000.0,
        "issue_d": "Dec-11",
        "loan_status": "Fully Paid",
        "purpose": "credit_card",
        "addr_state": "AZ",
        "dti": 27.65,
        "delinq_2yrs": 0.0,
        "earliest_cr_line": "Jan-95",
        "verification_status": "Verified",
        "initial_list_status": "f",
        "inq_last_6mths": 1.0,
        "open_acc": 3.0,
        "pub_rec": 0.0,
        "total_acc": 9.0,
        "acc_now_delinq": 0.0,
        "total_rev_hi_lim": None,
        "months_since_issue_date": 72.01790591182571,
        "months_since_earliest_cr_line": 394.98141645619006,
    }


@pytest.fixture(scope="module")
def time_since_input_dict() -> Dict[str, Any]:
    return {
        "id": 1077501,
        "member_id": 1296599,
        "earliest_cr_line": datetime.strptime("Jan-95", "%b-%y"),
    }


@pytest.fixture(scope="module")
def scores_and_true_labels() -> pd.DataFrame:
    np.random.seed(99)
    return pd.DataFrame(
        {
            "true_labels": np.random.choice([0, 1], size=100),
            "pd": np.random.rand(100),
        }
    )


@pytest.fixture(scope="module")
def probabilities_of_default() -> np.ndarray:
    np.random.seed(99)
    return np.round(np.linspace(0, 1, 21), 2)


@pytest.fixture(scope="module")
def model_input_no_zeros() -> pd.DataFrame:
    """Example dataframe for testing purposes
    Returns:
        pandas.DataFrame: Mocked dataframe for raising exception when training regressor
    """
    return pd.DataFrame(
        {"X1": [10.0, 15.0, 20.0], "X2": [0.0, 1.0, 0.0], "y": [200.0, 300.0, 400.0]}
    )


@pytest.fixture(scope="module")
def model_input() -> pd.DataFrame:
    """Example dataframe for testing purposes
    Returns:
        pandas.DataFrame: Mocked dataframe for fitting a Zero-Inflated Regressor
    """
    return pd.DataFrame(
        {"X1": [10.0, 15.0, 20.0], "X2": [0.0, 1.0, 0.0], "y": [200.0, 0, 400.0]}
    )
