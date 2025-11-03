"""
Configuration management for loan eligibility analysis.
"""
from pathlib import Path
from typing import Dict, Any
import json

def load_config() -> Dict[str, Any]:
    """
    Load configuration from config file.
    
    Returns:
        Dictionary containing configuration
    """
    config_path = Path(__file__).parent.parent / 'config' / 'config.json'
    
    if not config_path.exists():
        # Create default config if not exists
        default_config = {
            'data': {
                'input_path': 'data/loan_data.xlsx',
                'model_path': 'models',
                'test_size': 0.2
            },
            'model': {
                'random_state': 42,
                'class_weights': {0: 0.7, 1: 0.3}
            },
            'clustering': {
                'n_clusters': 3,
                'features': ['ApplicantIncome', 'LoanAmount']
            },
            'analysis': {
                'annual_interest_rate': 12.0,
                'risk_thresholds': {
                    'low': 75,
                    'medium': 50,
                    'high': 25
                }
            }
        }
        
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=4)
            
    with open(config_path, 'r') as f:
        return json.load(f)