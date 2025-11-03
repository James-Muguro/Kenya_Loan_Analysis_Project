"""
Model training and evaluation module for loan eligibility analysis.
"""
from pathlib import Path
from typing import Dict, Any, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class ModelTrainer:
    """Handles model training, evaluation, and persistence."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize with configuration."""
        self.config = config
        self.models = {
            'logistic': LogisticRegression(
                class_weight='balanced',
                max_iter=1000,
                random_state=42
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                class_weight='balanced',
                random_state=42
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                random_state=42
            )
        }
        self.best_model = None
        self.feature_importance = None
        
    def train_and_evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2
    ) -> Dict[str, Dict[str, float]]:
        """
        Train multiple models and evaluate their performance.
        
        Args:
            X: Feature DataFrame
            y: Target series
            test_size: Proportion of data for testing
            
        Returns:
            Dictionary of model performances
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        results = {}
        best_score = 0
        
        for name, model in self.models.items():
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_pred_proba),
                'cv_score': np.mean(cross_val_score(model, X, y, cv=5))
            }
            
            results[name] = metrics
            
            # Track best model
            if metrics['cv_score'] > best_score:
                best_score = metrics['cv_score']
                self.best_model = model
                
                # Calculate feature importance
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance = pd.Series(
                        model.feature_importances_,
                        index=X.columns
                    ).sort_values(ascending=False)
                elif hasattr(model, 'coef_'):
                    self.feature_importance = pd.Series(
                        abs(model.coef_[0]),
                        index=X.columns
                    ).sort_values(ascending=False)
                # Normalize importance to sum to 1 for comparability
                if self.feature_importance is not None:
                    total = float(self.feature_importance.sum())
                    if total > 0:
                        self.feature_importance = (self.feature_importance / total)
        
        return results
        
    def save_model(self, path: Path) -> None:
        """
        Save the best model and feature importance.
        
        Args:
            path: Path to save the model
        """
        if self.best_model is None:
            raise ValueError("No model has been trained yet")
            
        model_path = path / 'best_model.joblib'
        joblib.dump(self.best_model, model_path)
        
        if self.feature_importance is not None:
            importance_path = path / 'feature_importance.joblib'
            joblib.dump(self.feature_importance, importance_path)
            
    @staticmethod
    def load_model(path: Path) -> Tuple[Any, pd.Series]:
        """
        Load a saved model and its feature importance.
        
        Args:
            path: Path to load the model from
            
        Returns:
            Tuple of (model, feature_importance)
        """
        model_path = path / 'best_model.joblib'
        importance_path = path / 'feature_importance.joblib'
        
        model = joblib.load(model_path)
        feature_importance = joblib.load(importance_path)
        
        return model, feature_importance
        
    def plot_model_comparison(self, results: Dict[str, Dict[str, float]]) -> None:
        """
        Create interactive visualization of model comparisons.
        
        Args:
            results: Dictionary of model performances
        """
        metrics = list(next(iter(results.values())).keys())
        models = list(results.keys())
        
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=metrics,
            vertical_spacing=0.2
        )
        
        for i, metric in enumerate(metrics):
            row = i // 3 + 1
            col = i % 3 + 1
            
            values = [results[model][metric] for model in models]
            
            fig.add_trace(
                go.Bar(x=models, y=values, name=metric),
                row=row, col=col
            )
            
        fig.update_layout(
            height=800,
            title_text="Model Performance Comparison",
            showlegend=False
        )
        
        fig.show()
        
    def plot_feature_importance(self) -> None:
        """Create interactive visualization of feature importance."""
        if self.feature_importance is None:
            raise ValueError("No feature importance available")
            
        fig = go.Figure(
            go.Bar(
                x=self.feature_importance.values,
                y=self.feature_importance.index,
                orientation='h'
            )
        )
        
        fig.update_layout(
            title="Feature Importance",
            xaxis_title="Importance Score",
            yaxis_title="Feature",
            height=600
        )
        
        fig.show()