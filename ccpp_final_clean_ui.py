
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os
import streamlit as st
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# === Load Data ===
file_path = "CCPP_dataset.xlsx"
xls = pd.ExcelFile(file_path)
df = pd.concat([xls.parse(sheet) for sheet in xls.sheet_names], ignore_index=True)
X = df.drop(columns='PE')
y = df['PE']

# === Train and Save Models ===
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'Neural Network': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
}
os.makedirs("saved_models", exist_ok=True)
for name, model in models.items():
    model.fit(X, y)
    joblib.dump(model, f"saved_models/{name.replace(' ', '_').lower()}.joblib")

# === Sidebar Inputs ===
st.sidebar.header("🔧 Input Conditions")
AT = st.sidebar.slider("Ambient Temperature (°C)", float(X['AT'].min()), float(X['AT'].max()), float(X['AT'].mean()))
V = st.sidebar.slider("Exhaust Vacuum (cm Hg)", float(X['V'].min()), float(X['V'].max()), float(X['V'].mean()))
AP = st.sidebar.slider("Ambient Pressure (mbar)", float(X['AP'].min()), float(X['AP'].max()), float(X['AP'].mean()))
RH = st.sidebar.slider("Relative Humidity (%)", float(X['RH'].min()), float(X['RH'].max()), float(X['RH'].mean()))

user_input = pd.DataFrame([[AT, V, AP, RH]], columns=['AT', 'V', 'AP', 'RH'])

# === Navigation ===
st.sidebar.title("📂 Navigation")
section = st.sidebar.radio("Go to:", ["🏠 Main View", "📊 EDA", "🤖 Model Insights", "🏆 Best Model", "🛠️ Tuning"])

# === MAIN VIEW ===
if section == "🏠 Main View":
    st.title("⚡ CCPP - Model Explorer")

    selected_model_name = st.selectbox("Select Model:", list(models.keys()))
    model = joblib.load(f"saved_models/{selected_model_name.replace(' ', '_').lower()}.joblib")
    predicted_pe = model.predict(user_input)[0]

    st.subheader("🔮 Predicted Power Output")
    st.metric(label="Net Electrical Power Output (MW)", value=f"{predicted_pe:.2f}")

    # Save log
    log_file = "prediction_log.csv"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = {
        "Timestamp": timestamp,
        "Model": selected_model_name,
        "AT": AT, "V": V, "AP": AP, "RH": RH,
        "Predicted_PE": predicted_pe
    }
    log_df = pd.DataFrame([log_entry])
    if os.path.exists(log_file):
        log_df.to_csv(log_file, mode='a', header=False, index=False)
    else:
        log_df.to_csv(log_file, index=False)

    # Download button
    st.download_button(
        label="📥 Download Prediction Report",
        data=log_df.to_csv(index=False).encode('utf-8'),
        file_name=f"CCPP_Prediction_{timestamp.replace(':', '-')}.csv",
        mime='text/csv'
    )

    # Metrics
    full_preds = model.predict(X)
    st.subheader("📊 Model Performance")
    st.write(f"**R² Score:** {r2_score(y, full_preds):.4f}")
    st.write(f"**MAE:** {mean_absolute_error(y, full_preds):.2f}")
    st.write(f"**MSE:** {mean_squared_error(y, full_preds):.2f}")

    # Actual vs Predicted
    st.subheader("📈 Actual vs Predicted")
    fig, ax = plt.subplots()
    sns.scatterplot(x=y, y=full_preds, alpha=0.5, ax=ax)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    ax.set_xlabel("Actual PE")
    ax.set_ylabel("Predicted PE")
    st.pyplot(fig)

    # History
    st.subheader("📂 Prediction History")
    if os.path.exists(log_file):
        hist_df = pd.read_csv(log_file)
        st.dataframe(hist_df.tail(10))

        st.markdown("### Summary Stats")
        st.write(hist_df[['Predicted_PE']].describe())

        fig2, ax2 = plt.subplots()
        hist_df_sorted = hist_df.sort_values("Timestamp")
        ax2.plot(hist_df_sorted["Timestamp"], hist_df_sorted["Predicted_PE"], marker='o')
        ax2.set_xticklabels(hist_df_sorted["Timestamp"], rotation=45, ha='right')
        ax2.set_ylabel("Predicted PE")
        ax2.set_title("Power Output Trend")
        st.pyplot(fig2)
    else:
        st.info("No prediction history yet.")

# === EDA TAB ===
elif section == "📊 EDA":
    st.title("📊 Exploratory Data Analysis")

    st.title("Matplotlib Chart")
    features = ['AT', 'V', 'AP', 'RH', 'PE']
    fig1, axs1 = plt.subplots(3, 2, figsize=(12, 10))
    for i, feature in enumerate(features):
        row, col = divmod(i, 2)
        sns.histplot(df[feature], kde=True, ax=axs1[row][col])
        axs1[row][col].set_title(f'Distribution of {feature}')
    axs1[2][1].axis('off')
    st.pyplot(fig1)

    st.write("### Correlation Matrix")
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax2)
    st.pyplot(fig2)

    st.write("### Correlation Coefficients")
    fig3, axs3 = plt.subplots(2, 2, figsize=(12, 8))
    for i, feature in enumerate(['AT', 'V', 'AP', 'RH']):
        row, col = divmod(i, 2)
        sns.scatterplot(x=df[feature], y=df['PE'], alpha=0.5, ax=axs3[row][col])
        axs3[row][col].set_title(f"{feature} vs PE")
    st.pyplot(fig3)

# === MODEL INSIGHTS TAB ===
elif section == "🤖 Model Insights":
    st.title("🤖 Model Insights")
    selected_model_name = "Random Forest"
    model = joblib.load("saved_models/random_forest.joblib")
    if hasattr(model, "feature_importances_"):
        fig_imp, ax_imp = plt.subplots()
        sns.barplot(x=model.feature_importances_, y=X.columns, ax=ax_imp)
        ax_imp.set_title("Feature Importance")
        st.pyplot(fig_imp)
    else:
        st.info("Model does not support feature importances.")

# === BEST MODEL ===
elif section == "🏆 Best Model":
    st.title("🏆 Best Model Performance (Random Forest)")
    best_model = joblib.load("saved_models/random_forest.joblib")
    preds = best_model.predict(X)
    st.write(f"**R²:** {r2_score(y, preds):.4f}")
    st.write(f"**MAE:** {mean_absolute_error(y, preds):.2f}")
    st.write(f"**MSE:** {mean_squared_error(y, preds):.2f}")
    fig_best, ax_best = plt.subplots()
    sns.scatterplot(x=y, y=preds, alpha=0.5, ax=ax_best)
    ax_best.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    ax_best.set_title("Best Model - Actual vs Predicted")
    st.pyplot(fig_best)

# === TUNING TAB ===
elif section == "🛠️ Tuning":
    st.title("🛠️ Tuned Model Simulation")
    if st.button("Show Tuned Performance", key='tune_button'):
        tuned_model = RandomForestRegressor(n_estimators=300, max_depth=20, random_state=42)
        tuned_model.fit(X, y)
        preds = tuned_model.predict(X)
        st.write(f"**Tuned R²:** {r2_score(y, preds):.4f}")
        st.write(f"**Tuned MAE:** {mean_absolute_error(y, preds):.2f}")
        st.write(f"**Tuned MSE:** {mean_squared_error(y, preds):.2f}")
        fig_tuned, ax_tuned = plt.subplots()
        sns.scatterplot(x=y, y=preds, alpha=0.5, ax=ax_tuned)
        ax_tuned.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
        ax_tuned.set_title("Tuned Model - Actual vs Predicted")
        st.pyplot(fig_tuned)
