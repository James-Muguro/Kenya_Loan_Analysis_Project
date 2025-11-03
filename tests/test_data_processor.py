"""
Test suite for data processing module.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.data_processor import DataProcessor, DataSchema

@pytest.fixture
def sample_config():
    return {
        'annual_interest_rate': 12.0
    }

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'Loan_ID': ['LP001', 'LP002'],
        'Gender': ['Male', 'Female'],
        'Married': ['Yes', 'No'],
        'Dependents': ['0', '1'],
        'Education': ['Graduate', 'Not Graduate'],
        'Self_Employed': ['No', 'Yes'],
        'ApplicantIncome': [5000, 3000],
        'CoapplicantIncome': [1000, 0],
        'LoanAmount': [200, 150],
        'Loan_Amount_Term': [360, 300],
        'Credit_History': [1.0, 0.0],
        'Property_Area': ['Urban', 'Rural'],
        'Loan_Status': ['Y', 'N'],
        'county': ['Nairobi', 'Kiambu'],
        'application_date': ['2023-01-01', '2023-02-01']
    })

def test_data_schema_validation(sample_data):
    """Test data schema validation."""
    processor = DataProcessor({'annual_interest_rate': 12.0})
    
    # Test valid schema
    processor._validate_schema(sample_data)
    
    # Test invalid schema
    invalid_data = sample_data.drop('Gender', axis=1)
    with pytest.raises(ValueError):
        processor._validate_schema(invalid_data)

def test_clean_data(sample_data, sample_config):
    """Test data cleaning functionality."""
    processor = DataProcessor(sample_config)
    
    # Add some missing values
    sample_data.loc[0, 'Gender'] = None
    sample_data.loc[1, 'LoanAmount'] = None
    
    cleaned_data = processor.clean_data(sample_data)
    
    # Check if missing values are handled
    assert cleaned_data['Gender'].isnull().sum() == 0
    assert cleaned_data['LoanAmount'].isnull().sum() == 0
    
    # Check if derived features are created
    assert 'TotalIncome' in cleaned_data.columns
    assert 'Income_to_Loan' in cleaned_data.columns
    assert 'EMI' in cleaned_data.columns

def test_encode_features(sample_data, sample_config):
    """Test feature encoding functionality."""
    processor = DataProcessor(sample_config)
    
    encoded_data = processor.encode_features(sample_data)
    
    # Check categorical encoding
    assert encoded_data['Gender'].dtype in ['int64', 'float64']
    assert encoded_data['Married'].dtype in ['int64', 'float64']
    
    # Check numerical scaling
    assert encoded_data['ApplicantIncome'].mean() == pytest.approx(0, abs=1e-10)
    assert encoded_data['LoanAmount'].std() == pytest.approx(1, abs=1e-10)

def test_prepare_features(sample_data, sample_config):
    """Test feature preparation functionality."""
    processor = DataProcessor(sample_config)
    
    X, y = processor.prepare_features(sample_data)
    
    # Check if unnecessary columns are dropped
    assert 'Loan_ID' not in X.columns
    assert 'application_date' not in X.columns
    
    # Check if target is properly separated
    assert isinstance(y, pd.Series)
    assert y.name == 'Loan_Status'
    
    # Check shapes
    assert len(X) == len(y)
    assert all(col in X.columns for col in ['ApplicantIncome', 'LoanAmount'])