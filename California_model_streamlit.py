import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

# -----------------------------
# Load California dataset
# -----------------------------
housing = fetch_california_housing(as_frame=True)
df = housing.frame

X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="California Housing Price Predictor", page_icon="🏡", layout="wide")

st.title("🏡 California Housing Price Prediction")
st.markdown("Enter housing features below to predict **Median House Price**")

# Sidebar for input
st.sidebar.header("Input Features")

# Create input fields for each feature
input_data = {}
for col in X.columns:
    input_data[col] = st.sidebar.number_input(f"{col}", float(X[col].min()), float(X[col].max()), float(X[col].mean()))

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# Predict button
if st.sidebar.button("🔮 Predict"):
    prediction = model.predict(input_df)[0]
    st.success(f"🏠 Predicted Median House Price: **${prediction*100000:.2f}**")

# Show dataset preview (optional)
with st.expander("See Dataset Preview"):
    st.dataframe(df.head())

# Show model info
st.sidebar.markdown("### About Model")
st.sidebar.info("This Linear Regression model is trained on California Housing dataset.")


joblib.dump(model, "model.pkl")