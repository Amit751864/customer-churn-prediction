# Customer Churn Prediction

A machine learning project that predicts whether a telecom customer is
likely to leave (churn) based on their account, service, and billing
details. The project includes a trained Logistic Regression model and an
interactive Streamlit dashboard.

## Features

- Data cleaning and exploratory data analysis (EDA)
- Preprocessing for numerical and categorical features
- Logistic Regression classification
- Model evaluation using accuracy, precision, recall, F1-score, and a
  confusion matrix
- Tuned prediction threshold (`0.3`) to improve churn recall
- Streamlit form for entering one customer’s details
- Prediction probability, churn prediction, and a simple risk indicator
- Saved model and preprocessing artifacts loaded by the dashboard

## Model results

The notebook’s reported results at the default threshold (`0.5`) were:

| Metric                  |  Score |
|-------------------------|-------:|
| Accuracy                |  0.807 |
| Precision (Churn = Yes) | 0.6604 |
| Recall (Churn = Yes)    | 0.5615 |
| F1-score (Churn = Yes)  | 0.6069 |

The selected threshold is `0.3`, which favors identifying more customers
who may churn. In the notebook’s test results at threshold `0.3`, recall
was `0.7567`, precision was `0.5202`, accuracy was `0.7502`, and
F1-score was `0.6166`.

**Note:** Lowering the threshold generally catches more potential
churners but can also increase false positives. The displayed risk level
is a dashboard rule based on the predicted probability; it is not a
separately trained model output.

## Project structure

``` text
ml/
├── app.py
├── first.ipynb
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── requirements.txt
└── saved_models/
    ├── churn_model.joblib
    ├── churn_preprocessor.joblib
    └── churn_threshold.joblib
```

## Requirements

- Windows 10/11 (the setup instructions below are for Windows)
- Python 3.12 recommended for the dashboard environment
- The saved model artifacts in `saved_models/`

## Setup and run on Windows

Open PowerShell or Command Prompt in the project folder.

### 1. Create and activate a virtual environment

``` powershell
cd C:\Users\kumar\Coding\ml
py -3.12 -m venv .venv
```

For Command Prompt, activate it with:

``` bat
.venv\Scripts\activate.bat
```

If you are using PowerShell and script activation is blocked, you can
start a Command Prompt session from PowerShell:

``` powershell
cmd /k ".venv\Scripts\activate.bat"
```

### 2. Install dependencies

``` powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Start the dashboard

``` powershell
python -m streamlit run app.py
```

Streamlit will show a local URL in the terminal, usually
`http://localhost:8501`. Open that URL in your browser.

## How to use the dashboard

1.  Start the app using the command above.
2.  Enter the customer’s service, contract, tenure, and billing
    information.
3.  Click **Submit Customer Details**.
4.  Review the estimated churn probability, predicted outcome, and
    dashboard risk indicator.

The app expects the same feature names and categories used during model
training. Keep the three files in `saved_models/` alongside `app.py` in
the structure shown above.

## Dataset

The project uses the Telco Customer Churn dataset file:

`WA_Fn-UseC_-Telco-Customer-Churn.csv`

Make sure the dataset file is present before rerunning the notebook. The
dashboard uses the saved artifacts and does not need to retrain the
model at startup.

## Troubleshooting

- **Missing model file:** Check that all three `.joblib` files exist in
  `saved_models/`.
- **Package or model-loading errors:** Use the Python environment with
  compatible versions of `scikit-learn`, `numpy`, and the saved
  artifacts. Scikit-learn model files may not be compatible across
  different library versions.
- **`streamlit` command not found:** Run the app with
  `python -m streamlit run app.py` from the activated virtual
  environment.
- **PowerShell activation blocked:** Use the Command Prompt activation
  method above.

## Future improvements

- Add charts and a more polished dashboard layout
- Save prediction history and allow CSV export
- Add model explainability to show factors associated with churn
- Deploy the dashboard so it can be accessed through a public URL

## Disclaimer

This is an educational machine learning project. Predictions are
estimates and should not be treated as guarantees about an individual
customer’s behavior.
