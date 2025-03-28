import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# === Load Data and Models ===
data = pd.read_excel("CCPP_dataset.xlsx", sheet_name=None)
df = pd.concat(data.values(), ignore_index=True)

X = df.drop(columns='PE')
y = df['PE']

models = {
    'Linear Regression': joblib.load("saved_models/linear_regression.joblib"),
    'Random Forest': joblib.load("saved_models/random_forest.joblib"),
    'Gradient Boosting': joblib.load("saved_models/gradient_boosting.joblib"),
    'Neural Network': joblib.load("saved_models/neural_network.joblib")
}

# === Streamlit UI ===
st.title("CCPP Model Explorer 🔍")
selected_model = st.selectbox("Select a model to evaluate:", list(models.keys()))

model = models[selected_model]
preds = model.predict(X)

# === Metrics ===
mae = mean_absolute_error(y, preds)
mse = mean_squared_error(y, preds)
r2 = r2_score(y, preds)

st.subheader("Model Performance 📊")
st.write(f"**R² Score:** {r2:.4f}")
st.write(f"**MAE:** {mae:.2f}")
st.write(f"**MSE:** {mse:.2f}")

# === Plot: Actual vs Predicted ===
st.subheader("Actual vs Predicted Power Output")
fig, ax = plt.subplots()
sns.scatterplot(x=y, y=preds, alpha=0.5, ax=ax)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
ax.set_xlabel("Actual PE")
ax.set_ylabel("Predicted PE")
ax.set_title(f"{selected_model} - Actual vs Predicted")
st.pyplot(fig)
