"""
Advanced analytics and insights module for loan eligibility analysis.
"""
from typing import Dict, List, Any
from datetime import datetime

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)

class AnalyticsEngine:
    """Handles advanced analytics and insights generation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize with configuration."""
        self.config = config
        
    def perform_clustering(
        self,
        df: pd.DataFrame,
        features: List[str],
        n_clusters: int = 3
    ) -> pd.DataFrame:
        """
        Perform customer segmentation using K-means clustering.
        
        Args:
            df: Input DataFrame
            features: List of features for clustering
            n_clusters: Number of clusters
            
        Returns:
            DataFrame with cluster labels
        """
        df = df.copy()

        # Ensure requested features exist
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            raise KeyError(f"Missing clustering features in dataframe: {missing_features}")

        # Coerce relevant columns to numeric to avoid aggregation errors later
        agg_cols = ['ApplicantIncome', 'LoanAmount', 'Credit_History', 'Loan_Status']
        for col in agg_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Prepare feature matrix and drop rows with non-numeric / missing feature values
        feature_matrix = df[features].apply(pd.to_numeric, errors='coerce')
        valid_idx = feature_matrix.dropna().index
        if len(valid_idx) < len(df):
            num_dropped = len(df) - len(valid_idx)
            logger.warning("%d rows excluded from clustering due to missing/non-numeric feature values", num_dropped)

        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(feature_matrix.loc[valid_idx])

        # Perform clustering on valid rows only
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10
        )
        labels = kmeans.fit_predict(scaled_features)

        # Assign cluster labels; keep NaN for excluded rows
        df['Cluster'] = np.nan
        df.loc[valid_idx, 'Cluster'] = labels

        # Calculate cluster characteristics safely on numeric columns
        agg_mapping = {}
        if 'ApplicantIncome' in df.columns:
            agg_mapping['ApplicantIncome'] = ['mean', 'std']
        if 'LoanAmount' in df.columns:
            agg_mapping['LoanAmount'] = ['mean', 'std']
        if 'Credit_History' in df.columns:
            agg_mapping['Credit_History'] = 'mean'
        if 'Loan_Status' in df.columns:
            agg_mapping['Loan_Status'] = 'mean'

        diagnostics: dict = {}
        try:
            cluster_stats = df.dropna(subset=['Cluster']).groupby('Cluster').agg(agg_mapping).round(2)
            diagnostics['cluster_stats'] = cluster_stats
        except Exception as e:
            # Fallback: compute per-cluster numeric summaries manually
            logger.warning("failed to compute cluster_stats with .agg: %s. Falling back to manual aggregation.", e)
            cluster_stats = df.dropna(subset=['Cluster']).groupby('Cluster')[list(agg_mapping.keys())].apply(
                lambda g: g.apply(pd.to_numeric, errors='coerce').agg(['mean', 'std'] if g.name in ['ApplicantIncome', 'LoanAmount'] else 'mean')
            )
            diagnostics['cluster_stats'] = cluster_stats

        diagnostics['rows_excluded'] = int(len(df) - len(valid_idx))
        logger.info("Cluster Characteristics:\n%s", diagnostics['cluster_stats'])

        return df, diagnostics
        
    def analyze_temporal_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze and visualize temporal patterns in loan applications.
        
        Args:
            df: Input DataFrame
        """
        df = df.copy()

        # Extract temporal features
        df['Year'] = df['application_date'].dt.year
        df['Month'] = df['application_date'].dt.month
        df['Quarter'] = df['application_date'].dt.quarter
        
        # Create temporal visualizations
        # Use 'domain' type for pie chart in bottom-right cell
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{"type": "xy"}, {"type": "xy"}], [{"type": "xy"}, {"type": "domain"}]],
            subplot_titles=(
                'Monthly Application Volume',
                'Approval Rate by Quarter',
                'Average Loan Amount by Month',
                'Application Distribution by County'
            )
        )
        
        # Monthly application volume
        monthly_volume = df.groupby('Month').size()
        fig.add_trace(
            go.Scatter(
                x=monthly_volume.index,
                y=monthly_volume.values,
                mode='lines+markers',
                name='Applications'
            ),
            row=1, col=1
        )
        
        # Quarterly approval rate
        # Ensure Loan_Status is numeric (e.g., Y/N -> 1/0) before averaging
        diagnostics: dict = {}
        # Count strings in original Loan_Status
        loan_status_strings = df['Loan_Status'].apply(lambda v: isinstance(v, str)).sum()
        diagnostics['loan_status_strings'] = int(loan_status_strings)

        if df['Loan_Status'].dtype == object:
            df['Loan_Status_numeric'] = df['Loan_Status'].map({
                'Y': 1, 'y': 1, 'Yes': 1, 'YES': 1, 'yes': 1,
                'N': 0, 'n': 0, 'No': 0, 'NO': 0, 'no': 0
            })
            # fallback to coercion if mapping produced NaNs
            df['Loan_Status_numeric'] = pd.to_numeric(df['Loan_Status_numeric'], errors='coerce')
        else:
            df['Loan_Status_numeric'] = pd.to_numeric(df['Loan_Status'], errors='coerce')

        diagnostics['loan_status_nan_after_coercion'] = int(df['Loan_Status_numeric'].isna().sum())
        quarterly_approval = df.groupby('Quarter')['Loan_Status_numeric'].mean()
        fig.add_trace(
            go.Bar(
                x=quarterly_approval.index,
                y=quarterly_approval.values,
                name='Approval Rate'
            ),
            row=1, col=2
        )
        
        # Monthly average loan amount
        # Ensure LoanAmount is numeric
        df['LoanAmount'] = pd.to_numeric(df['LoanAmount'], errors='coerce')
        diagnostics['loanamount_nan_after_coercion'] = int(df['LoanAmount'].isna().sum())
        monthly_amount = df.groupby('Month')['LoanAmount'].mean()
        fig.add_trace(
            go.Bar(
                x=monthly_amount.index,
                y=monthly_amount.values,
                name='Avg Loan Amount'
            ),
            row=2, col=1
        )
        
        # County distribution
        county_dist = df['county'].value_counts()
        fig.add_trace(
            go.Pie(
                labels=county_dist.index,
                values=county_dist.values,
                name='County Distribution'
            ),
            row=2, col=2
        )

        fig.update_layout(height=800, showlegend=False)

        # Attach figure to diagnostics instead of calling fig.show() so tests and
        # non-interactive environments won't print large Plotly HTML/JS blobs.
        diagnostics['figure'] = fig

        # Return diagnostics so callers (and tests / UI) can surface coercion info
        return diagnostics
        
    def analyze_risk_patterns(self, df: pd.DataFrame) -> None:
        """
        Analyze and visualize risk patterns in loan applications.
        
        Args:
            df: Input DataFrame
        """
        df = df.copy()
        
        # Calculate risk metrics
        df['Risk_Score'] = df.apply(self._calculate_risk_score, axis=1)
        df['DTI_Ratio'] = self._calculate_dti_ratio(df)
        
        # Create risk visualizations
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Risk Score Distribution',
                'DTI Ratio vs Loan Status',
                'Risk by Education & Employment',
                'Risk by Property Area'
            )
        )
        
        # Risk score distribution
        fig.add_trace(
            go.Histogram(
                x=df['Risk_Score'],
                name='Risk Score'
            ),
            row=1, col=1
        )
        
        # DTI ratio vs loan status
        for status in df['Loan_Status'].unique():
            status_data = df[df['Loan_Status'] == status]
            fig.add_trace(
                go.Box(
                    y=status_data['DTI_Ratio'],
                    name=f'Status {status}',
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # Risk by education and employment
        edu_emp_risk = df.groupby(['Education', 'Self_Employed'])['Risk_Score'].mean().unstack()
        fig.add_trace(
            go.Heatmap(
                z=edu_emp_risk.values,
                x=edu_emp_risk.columns,
                y=edu_emp_risk.index,
                colorscale='RdYlGn_r'
            ),
            row=2, col=1
        )
        
        # Risk by property area
        area_risk = df.groupby('Property_Area')['Risk_Score'].mean()
        fig.add_trace(
            go.Bar(
                x=area_risk.index,
                y=area_risk.values,
                name='Risk by Area'
            ),
            row=2, col=2
        )

        fig.update_layout(height=800, showlegend=False)

        # Do not call fig.show() here. Return the figure so the caller (e.g., the
        # Streamlit UI) can render it explicitly. This avoids noisy test output.
        return {"figure": fig}
        
    def _calculate_risk_score(self, row: pd.Series) -> float:
        """Calculate risk score based on multiple factors."""
        score = 0
        
        # Credit history weight
        score += row['Credit_History'] * 40
        
        # DTI ratio weight
        dti_ratio = (row['EMI'] * 12) / (row['TotalIncome'] * 1000)
        if dti_ratio <= 0.3:
            score += 30
        elif dti_ratio <= 0.5:
            score += 15
        
        # Employment and education weight
        if row['Self_Employed'] == 1:
            score += 10
        if row['Education'] == 1:
            score += 10
        
        # Loan amount to income ratio weight
        if row['LoanAmount'] / row['TotalIncome'] <= 2:
            score += 10
            
        return score
        
    @staticmethod
    def _calculate_dti_ratio(df: pd.DataFrame) -> pd.Series:
        """Calculate Debt-to-Income ratio."""
        monthly_income = df['TotalIncome'] * 1000  # Convert to actual amount
        return (df['EMI'] * 12) / monthly_income
        
    def generate_insights_report(self, df: pd.DataFrame) -> str:
        """
        Generate a comprehensive insights report.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Markdown formatted report
        """
        # Calculate key metrics
        total_applications = len(df)

        # Create a safe numeric series for Loan_Status. Prefer any previously
        # computed numeric column, otherwise try mapping common string values
        # (Y/N, Yes/No). If a cell contains a short sequence of Y/N characters
        # (unexpected but seen in some ingested datasets), we interpret it as
        # the proportion of Ys as a heuristic. Fall back to NaN for unknowns.
        def _loan_status_numeric(series: pd.Series) -> pd.Series:
            if 'Loan_Status_numeric' in series.index:
                return pd.to_numeric(series['Loan_Status_numeric'], errors='coerce')

            s = series if isinstance(series, pd.Series) else pd.Series(series)
            mapping = {
                'Y': 1, 'y': 1, 'Yes': 1, 'YES': 1, 'yes': 1,
                'N': 0, 'n': 0, 'No': 0, 'NO': 0, 'no': 0
            }

            # First try direct mapping
            mapped = s.map(mapping)

            # Heuristic: strings that are sequences of Y/N characters -> proportion
            def _heuristic(v):
                if isinstance(v, str):
                    ss = v.strip()
                    if len(ss) > 1 and all(ch in 'YyNn' for ch in ss):
                        ys = sum(1 for ch in ss if ch in 'Yy')
                        return ys / len(ss)
                return np.nan

            mask_na = mapped.isna()
            if mask_na.any():
                mapped.loc[mask_na] = s[mask_na].apply(_heuristic)

            return pd.to_numeric(mapped, errors='coerce')

        loan_status_numeric = _loan_status_numeric(df['Loan_Status']) if 'Loan_Status' in df.columns else pd.Series(dtype=float)
        approval_rate = float(loan_status_numeric.mean(skipna=True) * 100) if not loan_status_numeric.empty else 0.0

        # Ensure LoanAmount and TotalIncome are numeric for averages
        if 'LoanAmount' in df.columns:
            loanamount_numeric = pd.to_numeric(df['LoanAmount'], errors='coerce')
        else:
            loanamount_numeric = pd.Series(dtype=float)

        if 'TotalIncome' in df.columns:
            totalincome_numeric = pd.to_numeric(df['TotalIncome'], errors='coerce')
        else:
            totalincome_numeric = pd.Series(dtype=float)

        avg_loan_amount = float(loanamount_numeric.mean(skipna=True)) if not loanamount_numeric.empty else 0.0
        avg_income = float(totalincome_numeric.mean(skipna=True)) if not totalincome_numeric.empty else 0.0
        
        # Generate report
        # Ensure Risk_Score exists; compute it if possible, otherwise fill with NaN
        if 'Risk_Score' not in df.columns:
            try:
                # Coerce numeric inputs used by the risk function where present
                for c in ['Credit_History', 'EMI', 'TotalIncome', 'Self_Employed', 'Education', 'LoanAmount']:
                    if c in df.columns:
                        df[c] = pd.to_numeric(df[c], errors='coerce')

                # Compute DTI_Ratio if EMI and TotalIncome are available
                if 'EMI' in df.columns and 'TotalIncome' in df.columns:
                    df['DTI_Ratio'] = self._calculate_dti_ratio(df)
                else:
                    df['DTI_Ratio'] = np.nan

                # Apply risk score row-wise defensively
                def _safe_risk(row):
                    try:
                        return self._calculate_risk_score(row)
                    except Exception:
                        return np.nan

                df['Risk_Score'] = df.apply(_safe_risk, axis=1)
            except Exception as e:
                logger.warning("Could not compute Risk_Score automatically: %s", e)
                df['Risk_Score'] = np.nan

        report = f"""
        # Loan Analysis Insights Report
        Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        ## Key Metrics
        - Total Applications: {total_applications:,}
        - Approval Rate: {approval_rate:.1f}%
        - Average Loan Amount: ₹{avg_loan_amount:,.2f}k
        - Average Total Income: ₹{avg_income:,.2f}k
        
        ## Risk Analysis
        - High Risk Applications: {len(df[df['Risk_Score'] < 50])} ({len(df[df['Risk_Score'] < 50])/total_applications*100:.1f}%)
        - Medium Risk Applications: {len(df[(df['Risk_Score'] >= 50) & (df['Risk_Score'] < 75)])} ({len(df[(df['Risk_Score'] >= 50) & (df['Risk_Score'] < 75)])/total_applications*100:.1f}%)
        - Low Risk Applications: {len(df[df['Risk_Score'] >= 75])} ({len(df[df['Risk_Score'] >= 75])/total_applications*100:.1f}%)
        
        ## Recommendations
        1. Focus on applicants with:
           - Credit History = 1
           - DTI Ratio <= 0.5
           - Education = Graduate
           
        2. Consider special programs for:
           - Self-employed applicants with strong business financials
           - First-time homebuyers in high-growth counties
           
        3. Risk Mitigation:
           - Implement stricter verification for high-risk applications
           - Consider collateral requirements for large loan amounts
           - Monitor temporal patterns for seasonal adjustments
        """
        
        return report