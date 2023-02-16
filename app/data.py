from typing import Optional
from pydantic import BaseModel


class FeatureStoreDataRequest(BaseModel):
    """Data model for incoming request from a user from a Feature Store, in the ideal scenario.
    Parameters:
        id (str): Loan's id.
        member_id (str): Member that requested loan.
        funded_amnt (float): Total ammount commited to that loan at a point in time.
        term (str, optional): Number of payments on the loan. Defaults to '60 months'.
        int_rate (float, optional): Interest rate on the loan. Defaults to None.
        grade (str, optional): Loan grade by external provider. Defaults to 'G'.
        emp_length (str, optional): Employment length, in years. Defaults to '< 1 year'.
        home_ownership (str, optional): Home ownership status provided by the borrower during registration. Defaults to 'RENT'.
        annual_inc (float, optional): Self-reported annual income provided by the borrower during the registration. Defaults to None.
        issue_d (str, optional): The month which the loan was funded. Defaults to None.
        purpose (str, optional): A category provided by the borrower for the loan request. Defaults to 'educational'.
        addr_state (str, optional): State provided by the borrower in the loan application. Defaults to 'NV'.
        dti (float, optional): Ratio calculated using borrower's total monthly debt payments on the total debt obligations (excluding mortgages and requested loan) divided by borrower's self-reported monthly income. Defaults to None.
        earliest_cr_line (str, optional): The month the borrower's earliest reported credit line was opened. Defaults to None.
        verification_status (str, optional): Verification provided by LC. Defaults to 'Verified'.
        initial_list_status (str, optional): Initial listing status of the loan. Defaults to 'f'.
        inq_last_6mths (float, optional): Number of inquiries paid in last 6 months. Defaults to None.
        total_rev_hi_lim (float, optional): The revolving high credit limit. Defaults to None.
        delinq_2yrs (float, optional): Number of 30+ days past-due incidences of delinquency in the borrower's credit file for the past 2 years. Defaults to None.
        pub_rec (float, optional): Number of derogatory public records. Defaults to None.
        total_acc (float, optional): Total number of credit lines in borrower's credit file. Defaults to None.
    """

    id: str
    member_id: str
    funded_amnt: float
    term: Optional[str] = "60 months"
    int_rate: Optional[float] = None
    grade: Optional[str] = "G"
    emp_length: Optional[str] = "< 1 year"
    home_ownership: Optional[str] = "RENT"
    annual_inc: Optional[float] = None
    issue_d: Optional[str] = None
    purpose: Optional[str] = "educational"
    addr_state: Optional[str] = "NV"
    dti: Optional[float] = None
    earliest_cr_line: Optional[str] = None
    verification_status: Optional[str] = "Verified"
    initial_list_status: Optional[str] = "f"
    inq_last_6mths: Optional[float] = None
    total_rev_hi_lim: Optional[float] = None
    delinq_2yrs: Optional[float] = None
    pub_rec: Optional[float] = None
    total_acc: Optional[float] = None
