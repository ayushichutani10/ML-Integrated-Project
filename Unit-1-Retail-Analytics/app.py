import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Retail Customer Analytics",
    page_icon="📊",
    layout="wide"
)

# ============================================================
# LOAD DATA
# ============================================================
@st.cache_data
def load_data():
    return pd.read_csv("retail_customer_data.csv")


df = load_data()

# ============================================================
# BASIC DATA PREPARATION
# ============================================================
required_columns = [
    "Customer_ID",
    "Age",
    "Income",
    "Visits",
    "Purchase_Amount"
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    st.error(
        "The CSV is missing these required columns: "
        + ", ".join(missing_columns)
    )
    st.stop()

df["Average_Spend_Per_Visit"] = (
    df["Purchase_Amount"] / df["Visits"]
)

# ============================================================
# QUANTILES
# ============================================================
income_q1 = df["Income"].quantile(0.25)
income_q2 = df["Income"].quantile(0.50)
income_q3 = df["Income"].quantile(0.75)

def income_group(income):
    if income <= income_q1:
        return "Low Income"
    elif income <= income_q2:
        return "Lower-Middle Income"
    elif income <= income_q3:
        return "Upper-Middle Income"
    else:
        return "High Income"

df["Income_Group"] = df["Income"].apply(income_group)

# ============================================================
# K-MEANS UNSUPERVISED LEARNING
# ============================================================
cluster_features = [
    "Age",
    "Income",
    "Visits",
    "Purchase_Amount"
]

scaler = StandardScaler()
X_cluster = scaler.fit_transform(df[cluster_features])

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(X_cluster)

cluster_purchase_order = (
    df.groupby("Cluster")["Purchase_Amount"]
    .mean()
    .sort_values()
    .index
)

cluster_names = {
    cluster_purchase_order[0]: "Low Value",
    cluster_purchase_order[1]: "Medium Value",
    cluster_purchase_order[2]: "High Value"
}

df["Customer_Segment"] = df["Cluster"].map(cluster_names)

# ============================================================
# HEADER
# ============================================================
st.title("📊 Retail Customer Analytics Dashboard")

st.markdown(
    """
    **Customer Spending Analysis using Statistics, Visualization,
    Supervised Learning and Unsupervised Learning**
    """
)

st.divider()

# ============================================================
# SIDEBAR FILTER
# ============================================================
st.sidebar.title("Dashboard Controls")

selected_segments = st.sidebar.multiselect(
    "Customer Segment",
    options=sorted(df["Customer_Segment"].unique()),
    default=sorted(df["Customer_Segment"].unique())
)

filtered_df = df[
    df["Customer_Segment"].isin(selected_segments)
].copy()

if len(filtered_df) == 0:
    st.warning("Select at least one customer segment.")
    st.stop()

# ============================================================
# KPI SECTION
# ============================================================
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Customers",
        len(filtered_df)
    )

with col2:
    st.metric(
        "Average Income",
        f"₹{filtered_df['Income'].mean():,.0f}"
    )

with col3:
    st.metric(
        "Average Purchase",
        f"₹{filtered_df['Purchase_Amount'].mean():,.0f}"
    )

with col4:
    st.metric(
        "Avg Spend / Visit",
        f"₹{filtered_df['Average_Spend_Per_Visit'].mean():,.0f}"
    )

st.divider()

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Statistics",
    "👥 Quantiles & Segments",
    "📊 Covariance & Relationship",
    "🤖 Supervised Learning",
    "🔵 Unsupervised Learning"
])

# ============================================================
# TAB 1: STATISTICS
# ============================================================
with tab1:

    st.header("1. Statistical Analysis")

    st.write(
        "Descriptive statistics for Age, Income, Visits and "
        "Purchase Amount."
    )

    statistics = pd.DataFrame({
        "Mean": filtered_df[
            ["Age", "Income", "Visits", "Purchase_Amount"]
        ].mean(),
        "Median": filtered_df[
            ["Age", "Income", "Visits", "Purchase_Amount"]
        ].median(),
        "Variance": filtered_df[
            ["Age", "Income", "Visits", "Purchase_Amount"]
        ].var(),
        "Standard Deviation": filtered_df[
            ["Age", "Income", "Visits", "Purchase_Amount"]
        ].std(),
        "Minimum": filtered_df[
            ["Age", "Income", "Visits", "Purchase_Amount"]
        ].min(),
        "Maximum": filtered_df[
            ["Age", "Income", "Visits", "Purchase_Amount"]
        ].max()
    })

    st.dataframe(
        statistics.round(2),
        width="stretch"
    )

    st.subheader("Distribution of Customer Variables")

    c1, c2 = st.columns(2)

    with c1:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(filtered_df["Age"], bins=8)
        ax.set_xlabel("Age")
        ax.set_ylabel("Customers")
        ax.set_title("Age Distribution")
        st.pyplot(fig)
        plt.close(fig)

    with c2:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(filtered_df["Income"], bins=8)
        ax.set_xlabel("Income")
        ax.set_ylabel("Customers")
        ax.set_title("Income Distribution")
        st.pyplot(fig)
        plt.close(fig)

# ============================================================
# TAB 2: QUANTILES & SEGMENTS
# ============================================================
with tab2:

    st.header("2. Quantile Analysis & Customer Grouping")

    st.subheader("Income Quantiles")

    quantile_table = pd.DataFrame({
        "Quantile": ["Q1 (25%)", "Q2 (50%)", "Q3 (75%)"],
        "Income": [
            income_q1,
            income_q2,
            income_q3
        ]
    })

    st.dataframe(
        quantile_table,
        width="stretch"
    )

    st.info(
        f"Q1 = ₹{income_q1:,.2f}, "
        f"Median = ₹{income_q2:,.2f}, "
        f"Q3 = ₹{income_q3:,.2f}"
    )

    st.subheader("Income-Based Customer Groups")

    group_counts = (
        filtered_df["Income_Group"]
        .value_counts()
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(9, 4))
    group_counts.plot(kind="bar", ax=ax)
    ax.set_xlabel("Income Group")
    ax.set_ylabel("Number of Customers")
    ax.set_title("Customers by Income Quantile Group")
    plt.xticks(rotation=20)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Customer Segment Statistics")

    segment_stats = (
        filtered_df
        .groupby("Customer_Segment")
        .agg(
            Customers=("Customer_ID", "count"),
            Average_Age=("Age", "mean"),
            Average_Income=("Income", "mean"),
            Average_Visits=("Visits", "mean"),
            Average_Purchase=("Purchase_Amount", "mean")
        )
        .round(2)
    )

    st.dataframe(
        segment_stats,
        width="stretch"
    )

# ============================================================
# TAB 3: COVARIANCE & RELATIONSHIP
# ============================================================
with tab3:

    st.header("3. Covariance & Relationship Analysis")

    covariance_matrix = filtered_df[
        ["Age", "Income", "Visits", "Purchase_Amount"]
    ].cov()

    correlation_matrix = filtered_df[
        ["Age", "Income", "Visits", "Purchase_Amount"]
    ].corr()

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Covariance Matrix")
        st.dataframe(
            covariance_matrix.round(2),
            width="stretch"
        )

    with c2:
        st.subheader("Correlation Matrix")
        st.dataframe(
            correlation_matrix.round(3),
            width="stretch"
        )

    income_covariance = filtered_df["Income"].cov(
        filtered_df["Purchase_Amount"]
    )

    income_correlation = filtered_df["Income"].corr(
        filtered_df["Purchase_Amount"]
    )

    st.info(
        f"Income vs Purchase Amount → "
        f"Covariance: {income_covariance:,.2f} | "
        f"Correlation: {income_correlation:.3f}"
    )

    st.subheader("Income vs Purchase Amount")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(
        filtered_df["Income"],
        filtered_df["Purchase_Amount"]
    )
    ax.set_xlabel("Income")
    ax.set_ylabel("Purchase Amount")
    ax.set_title("Income vs Purchase Amount")
    st.pyplot(fig)
    plt.close(fig)

# ============================================================
# TAB 4: SUPERVISED LEARNING
# ============================================================
with tab4:

    st.header("4. Supervised Learning — Purchase Prediction")

    st.markdown(
        """
        **Goal:** Predict `Purchase Amount` using:
        - Age
        - Income
        - Visits

        Since a target variable (`Purchase Amount`) is available,
        this is a **supervised learning** problem.
        """
    )

    X = filtered_df[
        ["Age", "Income", "Visits"]
    ]

    y = filtered_df["Purchase_Amount"]

    # Linear regression
    linear_model = LinearRegression()
    linear_model.fit(X, y)

    linear_pred = linear_model.predict(X)

    linear_r2 = r2_score(y, linear_pred)
    linear_mae = mean_absolute_error(y, linear_pred)
    linear_rmse = np.sqrt(
        mean_squared_error(y, linear_pred)
    )

    # Polynomial regression
    polynomial_model = make_pipeline(
        PolynomialFeatures(degree=2),
        LinearRegression()
    )

    polynomial_model.fit(X, y)

    polynomial_pred = polynomial_model.predict(X)

    polynomial_r2 = r2_score(y, polynomial_pred)
    polynomial_mae = mean_absolute_error(
        y,
        polynomial_pred
    )
    polynomial_rmse = np.sqrt(
        mean_squared_error(y, polynomial_pred)
    )

    st.subheader("Model Comparison")

    comparison = pd.DataFrame({
        "Model": [
            "Linear Regression",
            "Polynomial Regression (Degree 2)"
        ],
        "R²": [
            linear_r2,
            polynomial_r2
        ],
        "MAE": [
            linear_mae,
            polynomial_mae
        ],
        "RMSE": [
            linear_rmse,
            polynomial_rmse
        ]
    })

    st.dataframe(
        comparison.round(4),
        width="stretch"
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Linear R²",
            f"{linear_r2:.4f}"
        )

    with c2:
        st.metric(
            "Polynomial R²",
            f"{polynomial_r2:.4f}"
        )

    with c3:
        st.metric(
            "Polynomial RMSE",
            f"₹{polynomial_rmse:,.2f}"
        )

    st.subheader("Actual vs Predicted Purchase Amount")

    prediction_df = pd.DataFrame({
        "Actual": y.values,
        "Predicted": polynomial_pred
    })

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.scatter(
        prediction_df["Actual"],
        prediction_df["Predicted"]
    )

    minimum = min(
        prediction_df["Actual"].min(),
        prediction_df["Predicted"].min()
    )

    maximum = max(
        prediction_df["Actual"].max(),
        prediction_df["Predicted"].max()
    )

    ax.plot(
        [minimum, maximum],
        [minimum, maximum]
    )

    ax.set_xlabel("Actual Purchase Amount")
    ax.set_ylabel("Predicted Purchase Amount")
    ax.set_title("Actual vs Predicted Purchase Amount")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Prediction Table")

    prediction_output = filtered_df[
        ["Customer_ID", "Age", "Income", "Visits"]
    ].copy()

    prediction_output["Actual_Purchase"] = y.values
    prediction_output["Predicted_Purchase"] = (
        polynomial_pred.round(2)
    )

    st.dataframe(
        prediction_output,
        width="stretch"
    )

# ============================================================
# TAB 5: UNSUPERVISED LEARNING
# ============================================================
with tab5:

    st.header("5. Unsupervised Learning — K-Means")

    st.markdown(
        """
        K-Means clustering groups customers without using a target
        variable. The clustering uses:

        **Age + Income + Visits + Purchase Amount**
        """
    )

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Customer Segments")

        segment_counts = (
            filtered_df["Customer_Segment"]
            .value_counts()
        )

        fig, ax = plt.subplots(figsize=(7, 4))
        segment_counts.plot(kind="bar", ax=ax)
        ax.set_xlabel("Segment")
        ax.set_ylabel("Customers")
        ax.set_title("K-Means Customer Segments")
        plt.xticks(rotation=0)
        st.pyplot(fig)
        plt.close(fig)

    with c2:
        st.subheader("Income vs Purchase Amount")

        fig, ax = plt.subplots(figsize=(7, 4))

        for segment in sorted(
            filtered_df["Customer_Segment"].unique()
        ):
            data = filtered_df[
                filtered_df["Customer_Segment"] == segment
            ]

            ax.scatter(
                data["Income"],
                data["Purchase_Amount"],
                label=segment
            )

        ax.set_xlabel("Income")
        ax.set_ylabel("Purchase Amount")
        ax.set_title("Customer Clusters")
        ax.legend()

        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Cluster Summary")

    cluster_summary = (
        filtered_df
        .groupby("Customer_Segment")
        .agg(
            Customers=("Customer_ID", "count"),
            Average_Age=("Age", "mean"),
            Average_Income=("Income", "mean"),
            Average_Visits=("Visits", "mean"),
            Average_Purchase=("Purchase_Amount", "mean"),
            Average_Spend_Per_Visit=(
                "Average_Spend_Per_Visit",
                "mean"
            )
        )
        .round(2)
    )

    st.dataframe(
        cluster_summary,
        width="stretch"
    )

# ============================================================
# COMPLETE CUSTOMER DATA
# ============================================================
st.divider()

st.header("Complete Customer Dataset")

st.dataframe(
    filtered_df,
    width="stretch",
    hide_index=True
)

csv = filtered_df.to_csv(index=False).encode("utf-8")

st.download_button(
    "⬇️ Download Analysed Customer Data",
    data=csv,
    file_name="customer_analysis_output.csv",
    mime="text/csv"
)

st.caption(
    "Retail Customer Analytics | Statistics | Quantiles | "
    "Covariance | Supervised Learning | Unsupervised Learning | "
    "Polynomial Regression"
)
