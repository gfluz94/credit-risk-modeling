from typing import Any, Dict
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
        "earliest_cr_line": "Jan-65",
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
