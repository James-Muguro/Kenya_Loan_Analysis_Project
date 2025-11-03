"""
Main application module for loan eligibility analysis.
"""
from pathlib import Path
import os
import sys

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import streamlit as st
from src.config import load_config
from src.data_processor import DataProcessor
from src.model_trainer import ModelTrainer
from src.analytics import AnalyticsEngine

def main():
    """Main application entry point."""
    # Load configuration
    config = load_config()
    
    # Initialize components
    data_processor = DataProcessor(config)
    model_trainer = ModelTrainer(config['model'])
    analytics_engine = AnalyticsEngine(config['analysis'])
    
    # Set up Streamlit interface
    st.title("Kenya Loan Eligibility Analysis")
    
    # Data loading and processing
    st.header("1. Data Processing")
    data_file = st.file_uploader("Upload loan data (Excel file)", type=['xlsx'])
    
    if data_file:
        df = data_processor.load_data(data_file)
        
        if df is not None:
            st.success("Data loaded successfully!")
            
            # Display data overview
            st.subheader("Data Overview")
            st.write(df.head())
            st.write(f"Total records: {len(df)}")
            
            # Clean and preprocess data
            df_cleaned = data_processor.clean_data(df)
            st.success("Data cleaning completed!")
            
            # Feature engineering and encoding
            df_processed = data_processor.encode_features(df_cleaned)
            X, y = data_processor.prepare_features(df_processed)
            
            # Model training and evaluation
            st.header("2. Model Training")
            if st.button("Train Models"):
                with st.spinner("Training models..."):
                    results = model_trainer.train_and_evaluate(X, y)
                    
                st.success("Model training completed!")
                
                # Display model performance
                st.subheader("Model Performance")
                model_trainer.plot_model_comparison(results)
                
                # Display feature importance
                st.subheader("Feature Importance")
                model_trainer.plot_feature_importance()
                
                # Save best model
                model_dir = Path("models")
                model_dir.mkdir(exist_ok=True)
                model_trainer.save_model(model_dir)
                
            # Advanced Analytics
            st.header("3. Advanced Analytics")
            
            # Clustering Analysis
            st.subheader("Customer Segmentation")
            if st.button("Perform Clustering"):
                with st.spinner("Performing clustering analysis..."):
                    df_clustered, clustering_diag = analytics_engine.perform_clustering(
                        df_cleaned,
                        config['clustering']['features']
                    )
                st.success("Clustering completed!")
                # Surface diagnostics
                if clustering_diag.get('rows_excluded', 0) > 0:
                    st.warning(f"{clustering_diag['rows_excluded']} rows were excluded from clustering due to missing/non-numeric features.")
                if 'cluster_stats' in clustering_diag:
                    st.subheader('Cluster Summary')
                    st.dataframe(clustering_diag['cluster_stats'])
                
            # Temporal Analysis
            st.subheader("Temporal Patterns")
            if st.button("Analyze Temporal Patterns"):
                with st.spinner("Analyzing temporal patterns..."):
                    temporal_diag = analytics_engine.analyze_temporal_patterns(df_cleaned)
                st.success("Temporal analysis completed!")
                # Surface diagnostics
                if temporal_diag.get('loan_status_strings', 0) > 0:
                    st.info(f"Detected {temporal_diag['loan_status_strings']} Loan_Status entries that were strings and coerced.")
                if temporal_diag.get('loan_status_nan_after_coercion', 0) > 0:
                    st.warning(f"{temporal_diag['loan_status_nan_after_coercion']} Loan_Status values could not be coerced to numeric and are treated as missing.")
                if temporal_diag.get('loanamount_nan_after_coercion', 0) > 0:
                    st.warning(f"{temporal_diag['loanamount_nan_after_coercion']} LoanAmount values could not be coerced to numeric and are treated as missing.")
                # If a figure was returned, render it explicitly in Streamlit
                if temporal_diag and isinstance(temporal_diag, dict) and 'figure' in temporal_diag:
                    st.subheader('Temporal Visualizations')
                    st.plotly_chart(temporal_diag['figure'], use_container_width=True)
                st.success("Temporal analysis completed!")
                
            # Risk Analysis
            st.subheader("Risk Patterns")
            if st.button("Analyze Risk Patterns"):
                with st.spinner("Analyzing risk patterns..."):
                    risk_result = analytics_engine.analyze_risk_patterns(df_cleaned)
                # Render returned figure if present
                if risk_result and isinstance(risk_result, dict) and 'figure' in risk_result:
                    st.subheader('Risk Visualizations')
                    st.plotly_chart(risk_result['figure'], use_container_width=True)
                st.success("Risk analysis completed!")
                
            # Generate Insights Report
            st.header("4. Insights Report")
            if st.button("Generate Report"):
                with st.spinner("Generating insights report..."):
                    report = analytics_engine.generate_insights_report(df_cleaned)
                st.markdown(report)
                
if __name__ == "__main__":
    main()