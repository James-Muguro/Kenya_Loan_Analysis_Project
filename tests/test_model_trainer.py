"""
Test suite for model training module.
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from src.model_trainer import ModelTrainer

@pytest.fixture
def sample_config():
    return {
        'model': {
            'random_state': 42,
            'class_weights': {0: 0.7, 1: 0.3}
        }
    }

@pytest.fixture
def sample_data():
    np.random.seed(42)
    n_samples = 100
    
    X = pd.DataFrame({
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples),
        'feature3': np.random.randn(n_samples)
    })
    
    # Generate binary target with some relationship to features
    y = pd.Series((X['feature1'] + X['feature2'] > 0).astype(int))
    
    return X, y

def test_model_training(sample_data, sample_config):
    """Test model training and evaluation functionality."""
    X, y = sample_data
    trainer = ModelTrainer(sample_config)
    
    results = trainer.train_and_evaluate(X, y)
    
    # Check if all models are trained
    assert all(model in results for model in ['logistic', 'random_forest', 'gradient_boosting'])
    
    # Check if all metrics are computed
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'cv_score']
    assert all(metric in results['logistic'] for metric in metrics)
    
    # Check if metrics are in valid range
    for model_results in results.values():
        for metric, value in model_results.items():
            assert 0 <= value <= 1

def test_model_persistence(sample_data, sample_config, tmp_path):
    """Test model saving and loading functionality."""
    X, y = sample_data
    trainer = ModelTrainer(sample_config)
    
    # Train model
    trainer.train_and_evaluate(X, y)
    
    # Save model
    trainer.save_model(tmp_path)
    
    # Check if files are created
    assert (tmp_path / 'best_model.joblib').exists()
    assert (tmp_path / 'feature_importance.joblib').exists()
    
    # Load model
    loaded_model, loaded_importance = ModelTrainer.load_model(tmp_path)
    
    # Make predictions with loaded model
    original_predictions = trainer.best_model.predict(X)
    loaded_predictions = loaded_model.predict(X)
    
    # Check if predictions match
    assert np.array_equal(original_predictions, loaded_predictions)
    
    # Check if feature importance matches
    assert np.array_equal(
        trainer.feature_importance.values,
        loaded_importance.values
    )

def test_feature_importance(sample_data, sample_config):
    """Test feature importance calculation."""
    X, y = sample_data
    trainer = ModelTrainer(sample_config)
    
    trainer.train_and_evaluate(X, y)
    
    # Check if feature importance is calculated
    assert trainer.feature_importance is not None
    assert len(trainer.feature_importance) == X.shape[1]
    
    # Check if importance values are normalized
    assert trainer.feature_importance.sum() == pytest.approx(1.0, rel=1e-9)
    
    # Check if all features are included
    assert all(feature in trainer.feature_importance.index for feature in X.columns)