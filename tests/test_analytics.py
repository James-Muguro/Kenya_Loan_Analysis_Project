import pandas as pd
from datetime import datetime
from src.analytics import AnalyticsEngine


def test_analyze_temporal_patterns_coercion():
    # Create sample data with mixed Loan_Status values
    data = {
        'Loan_ID': ['L1','L2','L3','L4','L5','L6'],
        'Gender': ['Male']*6,
        'Married': ['Yes']*6,
        'Dependents': ['0']*6,
        'Education': ['Graduate']*6,
        'Self_Employed': ['No']*6,
        'ApplicantIncome': [1000,2000,1500,1200,1300,1100],
        'CoapplicantIncome': [0,0,0,0,0,0],
        'LoanAmount': ['200','300','unknown','250','400','150'],
        'Loan_Amount_Term': [360,360,360,360,360,360],
        'Credit_History': [1,1,0,1,1,0],
        'Property_Area': ['Urban']*6,
        # mixed Loan_Status: strings, numeric, case variants
        'Loan_Status': ['Y', 'N', 1, 0, 'yes', 'No'],
        'county': ['Nairobi']*6,
        'application_date': [
            pd.to_datetime('2023-01-15'),
            pd.to_datetime('2023-02-20'),
            pd.to_datetime('2023-04-05'),
            pd.to_datetime('2023-05-11'),
            pd.to_datetime('2023-07-07'),
            pd.to_datetime('2023-10-30')
        ]
    }
    df = pd.DataFrame(data)

    engine = AnalyticsEngine(config={})
    diagnostics = engine.analyze_temporal_patterns(df)

    # Diagnostics should be a dict with expected keys
    assert isinstance(diagnostics, dict)
    assert 'loan_status_strings' in diagnostics
    assert 'loan_status_nan_after_coercion' in diagnostics
    assert 'loanamount_nan_after_coercion' in diagnostics

    # We expect there to be some string Loan_Status entries
    assert diagnostics['loan_status_strings'] >= 2
    # 'unknown' in LoanAmount should coerce to NaN
    assert diagnostics['loanamount_nan_after_coercion'] >= 1
