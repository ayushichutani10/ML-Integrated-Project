# Customer Churn Prediction (Batch 6)

Predict whether a telecom customer will leave the service.
**Main model:** Logistic Regression (discriminative model estimating the probability of class membership).
**Comparison model:** Random Forest.

## Project structure
```
churn-prediction/
├── data/                  # put Telco-Customer-Churn.csv here
├── models/                # trained model saved here
├── reports/               # metrics.txt + figures/ (auto-generated)
├── app.py                 # local web dashboard (run this)
├── web/index.html         # dashboard page
├── src/
│   ├── config.py          # paths and settings
│   ├── data_prep.py       # loading, cleaning, preprocessing pipeline
│   ├── generate_synthetic_data.py  # fallback data if the real CSV is missing
│   ├── train.py           # EDA, training, tuning, evaluation, plots
│   └── predict.py         # predict churn for a single customer
└── requirements.txt
```

## Setup and run
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python app.py                    # opens the dashboard at http://localhost:8000

# optional, without the dashboard:
python src/train.py
python src/predict.py
```

## Use the REAL dataset (important for submission)
1. Download "Telco Customer Churn" from Kaggle (blastchar/telco-customer-churn).
2. Save it as `data/Telco-Customer-Churn.csv`.
3. Run `python src/train.py`.

If the file is missing, the code generates a synthetic dataset with the same
columns so the pipeline still runs. Results from synthetic data are NOT real.

## Dashboard tabs
Overview, Data (raw vs cleaned), EDA charts, Training (pipeline and code), Results (metrics, confusion matrix, ROC, coefficients), Try it (live prediction with explanation).
The first launch trains the model automatically (about a minute), then it opens instantly.

## What the pipeline does
1. Cleans data (`TotalCharges` blanks -> numeric, drops `customerID`).
2. EDA plots: class balance, churn by contract/internet/payment, tenure and charges.
3. Stratified 80/20 train/test split.
4. Preprocessing: StandardScaler (numeric) + OneHotEncoder (categorical) inside a sklearn Pipeline, so there is no data leakage.
5. Logistic Regression with `class_weight="balanced"`, tuned via 5-fold CV on ROC-AUC.
6. Random Forest for comparison.
7. Metrics: accuracy, precision, recall, F1, ROC-AUC, confusion matrix, ROC curves.
8. Coefficient plot showing which features drive churn.

## Opening in Antigravity
File > Open Folder > select `churn-prediction`. Open the terminal in the IDE and run `python app.py`.
You can also ask the Antigravity agent to run `src/train.py` and explain the results.
