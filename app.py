import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from pathlib import Path

st.set_page_config(
    page_title="Crop Type Detection System",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ====================== LOAD MODEL ======================
@st.cache_resource
def load_model():
    model = joblib.load("models/uzcosmos_crop_xgboost_model.pkl")
    encoder = joblib.load("models/label_encoder.pkl")
    return model, encoder


model, label_encoder = load_model()

# ====================== HEADER ======================
st.markdown(
    """
    <h1 style='text-align: center; color: #1E3A8A;'>Crop Type Detection System</h1>
    <p style='text-align: center; font-size: 18px; color: #475569;'>
        Uzbekistan Agricultural Fields • NDVI Time Series Classification
    </p>
""",
    unsafe_allow_html=True,
)

st.markdown("---")

# ====================== SIDEBAR ======================
with st.sidebar:
    st.header("Filters")

    selected_crops = st.multiselect(
        "Crop Types",
        options=label_encoder.classes_.tolist(),
        default=["cotton", "wheat", "wheat-other"],
    )

    min_confidence = st.slider("Minimum Confidence", 0.0, 1.0, 0.70, 0.01)

# ====================== MAIN AREA ======================
col_main, col_stats = st.columns([3, 1.2])

with col_main:
    st.subheader("Upload NDVI Time Series Data")

    uploaded_file = st.file_uploader(
        "Upload CSV file (must contain 23 NDVI columns)",
        type=["csv"],
    )

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    ndvi_columns = [col for col in df.columns if str(col).isdigit()]

    if len(ndvi_columns) != 23:
        st.error("The uploaded file must contain exactly 23 NDVI columns.")
        st.stop()

    # Predictions
    X_new = df[ndvi_columns]
    predictions_encoded = model.predict(X_new)
    predictions = label_encoder.inverse_transform(predictions_encoded)
    proba = model.predict_proba(X_new)
    confidence = proba.max(axis=1)

    df["predicted_crop"] = predictions
    df["confidence"] = confidence.round(4)

    # Apply filters
    result = df[df["confidence"] >= min_confidence]
    if selected_crops:
        result = result[result["predicted_crop"].isin(selected_crops)]

    # ====================== RESULTS TABLE ======================
    with col_main:
        st.subheader(f"Prediction Results — {len(result)} Fields")

        # Clean table - only important columns
        table_cols = ["predicted_crop", "confidence"] + ndvi_columns[:4]
        st.dataframe(
            result[table_cols].style.format({"confidence": "{:.4f}"}),
            use_container_width=True,
            height=450,
        )

    # ====================== STATISTICS (RIGHT COLUMN) ======================
    with col_stats:
        st.subheader("Statistics")

        crop_counts = result["predicted_crop"].value_counts()

        st.metric("Total Fields", len(result))

        if not crop_counts.empty:
            st.metric("Most Common Crop", crop_counts.index[0])
            fig = px.pie(
                names=crop_counts.index,
                values=crop_counts.values,
                title="Crop Distribution",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No filtered results available for charting.")

        # Download
        csv = result.to_csv(index=False).encode()
        st.download_button(
            label="Download Full Results",
            data=csv,
            file_name="uzcosmos_crop_predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )

else:
    st.info("System will generate ready to download predictions")

st.caption("Trained on CAWa Uzbekistan Dataset • Test Accuracy: 88.38%")
