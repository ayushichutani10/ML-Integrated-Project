import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

# ---------------------------------------
# 1. LOAD DATASET
# ---------------------------------------
df = pd.read_csv("retail_customer_data.csv")

print("RETAIL CUSTOMER SPENDING ANALYSIS")
print("=" * 40)
print("\nDataset:")
print(df.to_string(index=False))

# ---------------------------------------
# 2. STATISTICAL MEASURES
# ---------------------------------------
columns = ["Age", "Income", "Visits", "Purchase_Amount"]

print("\n\nSTATISTICAL MEASURES")
print("=" * 40)

for col in columns:
    print(f"\n{col}")
    print("Mean:", round(df[col].mean(), 2))
    print("Median:", round(df[col].median(), 2))
    print("Variance:", round(df[col].var(), 2))
    print("Standard Deviation:", round(df[col].std(), 2))
    print("Minimum:", df[col].min())
    print("Maximum:", df[col].max())

# ---------------------------------------
# 3. QUANTILES
# ---------------------------------------
print("\n\nQUANTILES")
print("=" * 40)

for col in columns:
    print(f"\n{col}")
    print("Q1 (25%):", df[col].quantile(0.25))
    print("Q2 (50%):", df[col].quantile(0.50))
    print("Q3 (75%):", df[col].quantile(0.75))

# ---------------------------------------
# 4. CUSTOMER SEGMENTATION USING INCOME
# ---------------------------------------
q1 = df["Income"].quantile(0.25)
q2 = df["Income"].quantile(0.50)
q3 = df["Income"].quantile(0.75)

def income_group(income):
    if income <= q1:
        return "Low Income"
    elif income <= q2:
        return "Lower-Middle Income"
    elif income <= q3:
        return "Upper-Middle Income"
    else:
        return "High Income"

df["Income_Group"] = df["Income"].apply(income_group)

print("\n\nCUSTOMER GROUPS")
print("=" * 40)
print(df[["Customer_ID", "Income", "Income_Group"]].to_string(index=False))

# ---------------------------------------
# 5. COVARIANCE
# ---------------------------------------
covariance = df["Income"].cov(df["Purchase_Amount"])

print("\n\nCOVARIANCE")
print("=" * 40)
print("Income vs Purchase Amount:", round(covariance, 2))

# ---------------------------------------
# 6. CORRELATION
# ---------------------------------------
correlation = df["Income"].corr(df["Purchase_Amount"])

print("\nCORRELATION")
print("=" * 40)
print("Income vs Purchase Amount:", round(correlation, 4))

# ---------------------------------------
# 7. POLYNOMIAL REGRESSION
# ---------------------------------------
X = df[["Income"]]
y = df["Purchase_Amount"]

model = make_pipeline(
    PolynomialFeatures(degree=2),
    LinearRegression()
)

model.fit(X, y)

y_pred = model.predict(X)
r2 = model.score(X, y)

print("\n\nPOLYNOMIAL REGRESSION")
print("=" * 40)
print("Polynomial degree: 2")
print("R-squared:", round(r2, 4))

# ---------------------------------------
# 8. GET POLYNOMIAL EQUATION
# ---------------------------------------
poly = model.named_steps["polynomialfeatures"]
reg = model.named_steps["linearregression"]

coefficients = reg.coef_
intercept = reg.intercept_

print("\nPolynomial equation:")
print(
    f"Purchase Amount = "
    f"{coefficients[2]:.10f} * Income² + "
    f"{coefficients[1]:.6f} * Income + "
    f"{intercept:.2f}"
)

# ---------------------------------------
# 9. PLOT
# ---------------------------------------
sort_index = np.argsort(df["Income"].values)

X_sorted = X.iloc[sort_index]
y_pred_sorted = model.predict(X_sorted)

plt.figure(figsize=(8, 5))
plt.scatter(
    df["Income"],
    df["Purchase_Amount"],
    label="Customer Data"
)
plt.plot(
    X_sorted,
    y_pred_sorted,
    label="Polynomial Regression"
)

plt.xlabel("Income")
plt.ylabel("Purchase Amount")
plt.title("Income vs Purchase Amount")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ---------------------------------------
# 10. SAVE ANALYSED DATA
# ---------------------------------------
df.to_csv("retail_customer_analysis_output.csv", index=False)

print("\nAnalysis completed successfully.")
