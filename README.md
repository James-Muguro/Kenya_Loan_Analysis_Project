# Kenya Loan Analysis Project

A reproducible, testable loan-analysis pipeline focused on Kenyan microloan applications.
This repository contains modular code for data processing, analytics, model training, and a Streamlit UI for interactive exploration.

The project emphasizes defensible analysis, robust data coercion, and clear diagnostics so users can trust results when data are messy.

---

## Status

- ✅ Modularized into `src/` (data processing, modeling, analytics, Streamlit UI)
- ✅ Unit tests under `tests/` (run with `pytest`)
- ✅ Streamlit UI that renders Plotly figures returned by analytics functions

---

## Highlights / Why this repo

- Clear separation of concerns: ETL, modeling, analytics, and UI live in separate modules.
- Defensive data handling: numeric coercion with diagnostics for problematic rows/columns.
- Analytics functions return diagnostics and Plotly `figure` objects instead of calling `fig.show()` (prevents noisy test output and keeps UI in charge of rendering).
- Small test-suite that validates coercion and diagnostics behavior.

---

## Quick start (local)

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run the Streamlit app locally:

```bash
streamlit run src/app.py
```

3. In the Streamlit UI:

- Upload an Excel file (or CSV if supported) containing the expected schema (see Data Schema below).
- Run Model Training or Advanced Analytics (clustering, temporal, risk) and inspect diagnostics and figures.
- Generate a PDF/HTML insights report (if the app runtime supports it) or download model artifacts from `models/`.

---

## Data schema & expectations

The pipeline expects a DataFrame / Excel sheet with the following columns (see `src/data_processor.py` for the authoritative list):

- Loan_ID
- Gender
- Married
- Dependents
- Education
- Self_Employed
- ApplicantIncome
- CoapplicantIncome
- LoanAmount
- Loan_Amount_Term
- Credit_History
- Property_Area
- Loan_Status
- county
- application_date

Notes:
- `application_date` should be parseable as a date (day-first supported).
- Numeric fields may contain non-numeric noise; the pipeline will coerce to numeric using `pd.to_numeric(..., errors='coerce')` and report coercion counts in diagnostics.
- `Loan_Status` can contain heterogeneous values (e.g., `Y`, `N`, `Yes`, `No`, or long Y/N sequences). The analytics functions include heuristics to robustly convert such values to a numeric approval score and will include diagnostics describing conversions and any unconvertable entries.

---

## What each module does

- `src/data_processor.py` — data loading, cleaning, feature engineering (EMI, TotalIncome, DTI), and encoding. Uses defensive assignment (no pandas chained-assignment) and exposes a `DataSchema` helper (see file).
- `src/model_trainer.py` — training pipelines, cross-validation, model evaluation, feature importance extraction, and model persistence (to `models/`).
- `src/analytics.py` — clustering (KMeans), temporal analysis (monthly/quarterly patterns), risk analysis, and reporting helpers. Each function returns a diagnostics dict and, when relevant, a Plotly `figure` object.
- `src/app.py` — Streamlit application that wires components together and uses `st.plotly_chart` to render figures returned by analytics functions.

---

## Diagnostics & failure modes

The project surfaces diagnostics from the data processing and analytics steps so you can quickly see where data quality issues exist. Common diagnostics include:

- Counts of values coerced to `NaN` when numeric coercion was applied.
- The set of original `Loan_Status` strings encountered and how many were converted to numeric values versus left as NaN.
- Rows excluded from clustering because required features were non-numeric or missing.

Design contract (short):
- Inputs: DataFrame or Excel file matching the schema above.
- Outputs: Processed DataFrame + diagnostics dict (and Plotly `figure` objects where applicable).
- Error modes: The functions prefer to return diagnostics describing failures rather than raising on common data problems; they only raise for unexpected programming errors (e.g., missing required library).

---

## Tests

Run unit tests with:

```bash
pytest -q
```

The test-suite includes checks for coercion and diagnostics in `src/analytics.py` and core processing logic in `src/data_processor.py`.

---

## Development

- Use the `src/` package for changes. Keep functions small and return diagnostics for any coercion/cleaning step.
- Replace `print()` with structured logging (module-level `logger = logging.getLogger(__name__)`).
- When adding visualizations in `src/analytics.py`, return Plotly figure objects and let `src/app.py` render them.

Suggested next tasks (small improvements):
- Add a tiny sample (anonymized) CSV for CI/demo.
- Add GitHub Actions workflow that runs `pytest` and any Markdown linting on push.

---

## Contributing

Contributions welcome. Please open issues for bugs or feature requests. For code contributions, follow these steps:

1. Fork the repo and create a branch for your change.
2. Add/modify tests where appropriate.
3. Run `pytest` locally and ensure tests pass.
4. Open a pull request describing the change and link any relevant issue.

If you want me to open a PR that adds CI (tests + lint) or a small anonymized sample dataset, tell me which and I can implement it.

---

## License

This project uses open-source dependencies listed in `requirements.txt`. The project itself is published under the repository license in `LICENSE`.

---

## Contact

If you need help running the app or have questions about the pipeline, leave an issue or contact the repository owner.
