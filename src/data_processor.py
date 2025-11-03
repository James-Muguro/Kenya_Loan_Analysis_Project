"""
Data processing module for loan eligibility analysis.
"""
from pathlib import Path
from typing import Optional, Dict, Any, ClassVar

import pandas as pd
from pydantic import BaseModel, ConfigDict
from sklearn.preprocessing import StandardScaler
import numpy as np

class DataSchema(BaseModel):
    """Data schema validation model."""
    # Pydantic v2: replace inner `Config` class with `model_config` using ConfigDict
    model_config = ConfigDict(arbitrary_types_allowed=True)

    EXPECTED_COLUMNS: ClassVar[list[str]] = [
        'Loan_ID', 'Gender', 'Married', 'Dependents', 'Education',
        'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome',
        'LoanAmount', 'Loan_Amount_Term', 'Credit_History',
        'Property_Area', 'Loan_Status', 'county', 'application_date'
    ]

class DataProcessor:
    """Handles data loading, cleaning, and preprocessing."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize with configuration."""
        self.config = config
        self.scaler = StandardScaler()
        
    def load_data(self, file_path: Path) -> Optional[pd.DataFrame]:
        """
        Load data from Excel file with validation.
        
        Args:
            file_path: Path to the Excel file
            
        Returns:
            Loaded DataFrame or None if error occurs
        """
        try:
            df = pd.read_excel(file_path)
            self._validate_schema(df)
            return df
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return None
            
    def _validate_schema(self, df: pd.DataFrame) -> None:
        """Validate DataFrame schema."""
        missing_cols = set(DataSchema.EXPECTED_COLUMNS) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess the data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        df = df.copy()
        
        # Handle missing values with more sophisticated strategies
        categorical_cols = df.select_dtypes(include=['object']).columns
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        
        for col in categorical_cols:
            if df[col].isnull().any():
                # Use mode for categorical, but consider frequency distribution
                mode_val = df[col].mode()[0]
                # Avoid chained-assignment / inplace on a view — assign back explicitly
                df[col] = df[col].fillna(mode_val)
        
        for col in numerical_cols:
            if df[col].isnull().any():
                # Use median but consider outliers
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
        
        # Convert and validate dates
        df['application_date'] = pd.to_datetime(
            df['application_date'],
            dayfirst=True,
            errors='coerce'
        )
        
        # Add derived features
        df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
        df['Income_to_Loan'] = df['TotalIncome'] / df['LoanAmount']
        df['EMI'] = df.apply(self._calculate_emi, axis=1)
        
        # Handle any remaining NaNs from feature creation
        for col in ['TotalIncome', 'Income_to_Loan', 'EMI']:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
        
        return df
        
    def _calculate_emi(self, row: pd.Series) -> float:
        """Calculate EMI for a loan."""
        P = row['LoanAmount'] * 1000  # Convert to actual amount
        N = row['Loan_Amount_Term']
        # Support both nested config['analysis']['annual_interest_rate'] and top-level 'annual_interest_rate'
        annual_rate = None
        if isinstance(self.config, dict):
            annual_rate = (
                self.config.get('analysis', {}) or {}
            ).get('annual_interest_rate')
            if annual_rate is None:
                annual_rate = self.config.get('annual_interest_rate')
        # Fallback default
        if annual_rate is None:
            annual_rate = 12.0

        R = annual_rate / 12 / 100  # Monthly rate
        
        try:
            emi = P * R * (1 + R)**N / ((1 + R)**N - 1)
            return round(emi, 2)
        except:
            return np.nan
            
    def encode_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical features and scale numerical features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Encoded DataFrame
        """
        df = df.copy()
        
        # Define mappings
        mappings = {
            'Gender': {'Male': 0, 'Female': 1},
            'Married': {'No': 0, 'Yes': 1},
            'Education': {'Not Graduate': 0, 'Graduate': 1},
            'Self_Employed': {'No': 0, 'Yes': 1},
            'Property_Area': {'Rural': 0, 'Semiurban': 1, 'Urban': 2},
            'Loan_Status': {'N': 0, 'Y': 1},
            'county': {'Nairobi': 0, 'Kiambu': 1, 'Machakos': 2, 'Mombasa': 3},
            'Dependents': {'0': 0, '1': 1, '2': 2, '3+': 3}
        }
        
        # Apply mappings
        for col, mapping in mappings.items():
            if col in df.columns:
                df[col] = df[col].map(mapping)
                if df[col].isnull().any():
                    # Assign result of fillna back to avoid inplace on slice
                    df[col] = df[col].fillna(df[col].mode()[0])
        
        # Scale numerical features
        numerical_features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
                            'Loan_Amount_Term', 'TotalIncome', 'EMI']

        # Ensure derived numerical features exist before scaling
        if 'TotalIncome' not in df.columns and 'ApplicantIncome' in df.columns and 'CoapplicantIncome' in df.columns:
            df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
        if 'EMI' not in df.columns and all(c in df.columns for c in ['LoanAmount', 'Loan_Amount_Term']):
            df['EMI'] = df.apply(self._calculate_emi, axis=1)

        existing_numericals = [c for c in numerical_features if c in df.columns]
        if existing_numericals:
            # Use sample standard deviation (ddof=1) so pandas `.std()` on the result equals 1
            for col in existing_numericals:
                col_mean = df[col].mean()
                col_std = df[col].std(ddof=1)
                if col_std == 0 or pd.isna(col_std):
                    df[col] = 0.0
                else:
                    df[col] = (df[col] - col_mean) / col_std
        
        return df
        
    def prepare_features(self, df: pd.DataFrame, target: str = 'Loan_Status') -> tuple:
        """
        Prepare features for modeling.
        
        Args:
            df: Input DataFrame
            target: Target variable name
            
        Returns:
            Tuple of features and target
        """
        df = df.copy()
        
        # Drop unnecessary columns
        drop_cols = ['Loan_ID', 'application_date']
        X = df.drop(columns=drop_cols + [target])
        y = df[target]
        
        return X, y