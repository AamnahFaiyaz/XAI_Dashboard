import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(page_title="Conference XAI Dashboard", layout="wide")
st.title("📊 Explainable AI Dashboard")
st.markdown("Conference-Level Interactive ML + XAI System")

# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    return joblib.load("trained_model.pkl")

model = load_model()
expected_features = model.feature_names_in_

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV or Excel File",
    type=["csv", "xlsx"]
)

# =========================================================
# MAIN LOGIC
# =========================================================

if uploaded_file is not None:

    try:
        # ----------------------------
        # READ FILE
        # ----------------------------
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.subheader("📂 Dataset Preview")
        st.dataframe(df.head())

        # ----------------------------
        # FEATURE ALIGNMENT
        # ----------------------------
        df_model = df.copy()

        df_model = df_model[[col for col in expected_features if col in df_model.columns]]

        for col in expected_features:
            if col not in df_model.columns:
                df_model[col] = 0

        df_model = df_model[expected_features]

        for col in df_model.columns:
            if df_model[col].dtype == "object":
                df_model[col] = df_model[col].astype("category").cat.codes

        df_model = df_model.apply(pd.to_numeric, errors="coerce").fillna(0)

        # ----------------------------
        # RUN PREDICTION
        # ----------------------------
        if st.sidebar.button("Run Prediction"):

            predictions = model.predict(df_model)

            result_df = df.copy()
            result_df["Prediction"] = predictions

            st.subheader("🔮 Predictions")
            st.dataframe(result_df)

            # =====================================================
            # CONFIDENCE PROBABILITIES (Classification only)
            # =====================================================

            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(df_model)
                confidence = np.max(probs, axis=1)
                result_df["Confidence"] = confidence

                st.subheader("📊 Prediction Confidence")
                st.dataframe(result_df[["Prediction", "Confidence"]].head())

            # =====================================================
            # DOWNLOAD PREDICTIONS
            # =====================================================

            csv_predictions = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Predictions",
                csv_predictions,
                "predictions.csv",
                "text/csv"
            )

            # =====================================================
            # PREDICTION DISTRIBUTION
            # =====================================================

            st.subheader("📈 Prediction Distribution")

            fig_dist, ax_dist = plt.subplots()
            pd.Series(predictions).hist(ax=ax_dist, bins=20)
            st.pyplot(fig_dist)

            csv_dist = pd.Series(predictions).to_csv().encode("utf-8")
            st.download_button(
                "📥 Download Prediction Distribution Data",
                csv_dist,
                "prediction_distribution.csv",
                "text/csv"
            )

            # =====================================================
            # CORRELATION HEATMAP
            # =====================================================

            st.subheader("🔥 Feature Correlation Heatmap")

            corr = df_model.corr()

            fig_corr, ax_corr = plt.subplots(figsize=(8,6))
            sns.heatmap(corr, cmap="coolwarm", ax=ax_corr)
            st.pyplot(fig_corr)

            csv_corr = corr.to_csv().encode("utf-8")
            st.download_button(
                "📥 Download Correlation Matrix",
                csv_corr,
                "correlation_matrix.csv",
                "text/csv"
            )

            # =====================================================
            # GLOBAL SHAP (Sampled for Speed)
            # =====================================================

            st.subheader("🌍 Global SHAP Feature Importance")

            sample_df = df_model.sample(min(100, len(df_model)), random_state=42)

            explainer = shap.Explainer(model, sample_df)
            shap_values = explainer(sample_df)

            fig_shap, ax_shap = plt.subplots()
            shap.plots.bar(shap_values, show=False)
            st.pyplot(fig_shap)

            shap_df = pd.DataFrame(
                np.abs(shap_values.values).mean(axis=0),
                index=sample_df.columns,
                columns=["Mean |SHAP|"]
            ).sort_values("Mean |SHAP|", ascending=False)

            csv_shap = shap_df.to_csv().encode("utf-8")
            st.download_button(
                "📥 Download SHAP Importance",
                csv_shap,
                "shap_importance.csv",
                "text/csv"
            )

            # =====================================================
            # INDIVIDUAL SHAP
            # =====================================================

            st.subheader("👤 Individual Student Explanation")

            selected_index = st.selectbox(
                "Select Row Index",
                range(len(df_model))
            )

            single_instance = df_model.iloc[[selected_index]]
            single_shap = explainer(single_instance)

            fig_ind, ax_ind = plt.subplots()
            shap.plots.waterfall(single_shap[0], show=False)
            st.pyplot(fig_ind)

    except Exception as e:
        st.error("Prediction failed.")
        st.write(str(e))

else:
    st.info("Upload a CSV or Excel file to begin.")