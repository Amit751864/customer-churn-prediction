
from pathlib import Path

readme_content = """# Customer Churn Prediction

## Project Overview
This machine learning project predicts whether a telecom customer
is likely to leave the service (churn) using customer account and
service information.

## Technologies Used
- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Joblib
- Jupyter Notebook

## Dataset
Telco Customer Churn dataset with 7,043 customer records.

## Machine Learning Workflow
1. Data cleaning and preprocessing
2. Exploratory Data Analysis (EDA)
3. Train-test split
4. One-hot encoding and numerical preprocessing
5. Logistic Regression model training
6. Model evaluation
7. Probability threshold comparison
8. Saving and loading the trained model
9. Customer churn prediction

## Model Results
- Accuracy at threshold 0.5: 80.70%
- Recall at threshold 0.5: 56.15%
- Recall at threshold 0.4: 66.58%
- Recall at threshold 0.3: 75.67%
- F1-score at threshold 0.3: 61.66%

Note: Results are based on the current test split. The threshold
of 0.3 increases recall but also increases false positives.

## Saved Model Files
- saved_models/churn_model.joblib
- saved_models/churn_preprocessor.joblib
- saved_models/churn_threshold.joblib

## How to Run
1. Install Python and Jupyter Notebook.
2. Install the required packages.
3. Open first.ipynb.
4. Run the notebook cells in order.

## Future Improvements
- Compare Logistic Regression with tree-based models.
- Tune the classification threshold using validation data.
- Build a Streamlit web application.
"""

Path("README.md").write_text(readme_content, encoding="utf-8")

print("README.md created successfully!")
print("Location:", Path("README.md").resolve())
