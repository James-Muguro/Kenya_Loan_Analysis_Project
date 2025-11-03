# Kenya Loan Analysis

> A reproducible, testable microloan analysis pipeline for Kenyan loan applications with interactive visualization and robust data handling.

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-pytest-green)](https://docs.pytest.org/)
[![Code Style](https://img.shields.io/badge/code%20style-defensive-orange)]()

## Overview

This project provides a complete pipeline for analyzing microloan application data with emphasis on:

- **Defensive data handling** — Robust numeric coercion with comprehensive diagnostics
- **Modular architecture** — Clear separation between ETL, modeling, analytics, and UI
- **Reproducibility** — Test coverage and deterministic results
- **Interactive exploration** — Streamlit UI for dynamic analysis and visualization

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Data Schema](#data-schema)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Testing](#testing)
- [Diagnostics](#diagnostics)
- [Contributing](#contributing)
- [License](#license)

## Features

- ✅ **Data Processing** — ETL pipeline with feature engineering (EMI, DTI ratios)
- ✅ **Machine Learning** — Model training with cross-validation and persistence
- ✅ **Advanced Analytics** — Clustering, temporal patterns, and risk analysis
- ✅ **Interactive UI** — Streamlit dashboard with Plotly visualizations
- ✅ **Comprehensive Testing** — Unit tests for core functionality
- ✅ **Quality Diagnostics** — Detailed reporting on data quality issues

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd kenya-loan-analysis
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv .venv
   
   # On macOS/Linux
   source .venv/bin/activate
   
   # On Windows
   .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

Launch the Streamlit application:

```bash
streamlit run src/app.py
```

Then:
1. Upload an Excel or CSV file matching the expected schema
2. Explore data quality diagnostics
3. Train models and view performance metrics
4. Run advanced analytics (clustering, temporal analysis, risk scoring)
5. Generate and download insights reports

## Data Schema

The pipeline expects the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `Loan_ID` | string | Unique loan identifier |
| `Gender` | categorical | Applicant gender |
| `Married` | categorical | Marital status |
| `Dependents` | numeric/categorical | Number of dependents |
| `Education` | categorical | Education level |
| `Self_Employed` | categorical | Employment type |
| `ApplicantIncome` | numeric | Primary applicant income |
| `CoapplicantIncome` | numeric | Co-applicant income |
| `LoanAmount` | numeric | Requested loan amount |
| `Loan_Amount_Term` | numeric | Loan term in months |
| `Credit_History` | numeric | Credit history indicator |
| `Property_Area` | categorical | Property location type |
| `Loan_Status` | categorical | Approval status (Y/N variants) |
| `county` | categorical | Kenyan county |
| `application_date` | date | Application submission date |

### Data Notes

- **Date parsing**: `application_date` supports day-first format
- **Numeric coercion**: Non-numeric values are converted to NaN with diagnostic reporting
- **Loan status**: Handles heterogeneous values (Y, N, Yes, No, sequences) with robust conversion logic

## Project Structure

```
.
├── src/
│   ├── data_processor.py    # ETL, cleaning, feature engineering
│   ├── model_trainer.py     # ML pipelines, evaluation, persistence
│   ├── analytics.py         # Clustering, temporal, risk analysis
│   └── app.py              # Streamlit UI application
├── tests/                   # Unit tests (pytest)
├── models/                  # Saved model artifacts
├── requirements.txt         # Python dependencies
└── README.md
```

### Module Responsibilities

**`data_processor.py`**
- Load and validate input data
- Feature engineering (EMI, TotalIncome, DTI)
- Encoding and normalization
- Schema validation via `DataSchema` helper

**`model_trainer.py`**
- Training pipeline configuration
- Cross-validation and evaluation
- Feature importance analysis
- Model serialization

**`analytics.py`**
- KMeans clustering analysis
- Temporal pattern detection (monthly/quarterly)
- Risk scoring and segmentation
- Returns diagnostic dicts and Plotly figures

**`app.py`**
- Streamlit UI components
- File upload handling
- Visualization rendering
- Report generation

## Usage

### Programmatic Access

```python
from src.data_processor import load_and_process_data
from src.model_trainer import train_model
from src.analytics import run_clustering_analysis

# Load and process data
df, diagnostics = load_and_process_data('data.xlsx')

# Train model
model, metrics = train_model(df)

# Run analytics
cluster_results, fig = run_clustering_analysis(df)
```

### Command Line

```bash
# Run tests
pytest -v

# Run with coverage
pytest --cov=src --cov-report=html

# Run Streamlit app
streamlit run src/app.py --server.port 8501
```

## Testing

Run the test suite:

```bash
# All tests
pytest

# Verbose output
pytest -v

# Specific test file
pytest tests/test_analytics.py

# With coverage report
pytest --cov=src --cov-report=term-missing
```

Test coverage includes:
- Data coercion and validation
- Numeric conversion edge cases
- Diagnostic generation
- Feature engineering logic

## Diagnostics

The pipeline provides comprehensive diagnostics at each stage:

### Data Processing
- Counts of values coerced to NaN
- Invalid date formats
- Missing required columns
- Data type mismatches

### Analytics
- Original `Loan_Status` value distribution
- Conversion success/failure rates
- Rows excluded from clustering
- Feature missingness reports

### Design Contract

**Inputs**: DataFrame or Excel/CSV file matching schema  
**Outputs**: Processed DataFrame, diagnostics dict, Plotly figures (where applicable)  
**Error Handling**: Returns diagnostics for data quality issues; raises only for programming errors

## Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork** the repository and create a feature branch
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Write tests** for new functionality
   ```bash
   pytest tests/
   ```

3. **Follow conventions**
   - Use type hints where applicable
   - Return diagnostics for data operations
   - Keep functions small and focused
   - Use module-level logging instead of `print()`

4. **Submit a pull request** with:
   - Clear description of changes
   - Link to related issues
   - Test coverage for new code

### Development Best Practices

- Return Plotly figure objects from analytics functions
- Let `app.py` handle rendering (avoid `fig.show()`)
- Use defensive DataFrame operations (avoid chained assignment)
- Provide structured diagnostics for debugging

## Roadmap

- [ ] Add anonymized sample dataset for demos
- [ ] Implement CI/CD with GitHub Actions
- [ ] Add pre-commit hooks for code quality
- [ ] Expand test coverage to >80%
- [ ] Add API documentation with Sphinx
- [ ] Support additional data formats (Parquet, JSON)

## License

This project is licensed under the terms specified in the [LICENSE](LICENSE) file.

Dependencies are listed in `requirements.txt` and subject to their respective licenses.

## Support

- **Issues**: [Open an issue](../../issues) for bugs or feature requests
- **Discussions**: [Start a discussion](../../discussions) for questions or ideas
- **Documentation**: Check inline docstrings and module comments

---

**Made with** 🇰🇪 **for transparent, defensible loan analysis**
