from setuptools import setup, find_packages

setup(
    name="kenya_loan_analysis",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'pandas>=2.1.0',
        'numpy>=1.24.0',
        'matplotlib>=3.7.0',
        'seaborn>=0.12.0',
        'scikit-learn>=1.3.0',
        'ipywidgets>=8.0.0',
        'openpyxl>=3.1.0',
        'python-dotenv>=1.0.0',
        'joblib>=1.3.0',
        'pytest>=7.4.0',
        'black>=23.9.0',
        'isort>=5.12.0',
        'pylint>=2.17.0',
        'pydantic>=2.4.0',
        'streamlit>=1.27.0',
        'plotly>=5.17.0'
    ],
    python_requires='>=3.8',
)