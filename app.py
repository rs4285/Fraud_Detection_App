# app.py
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import gdown
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier

# --------------------------------------
# Page configuration
# --------------------------------------
st.set_page_config(
    page_title="💳 Fraud Detection Dashboard",
    page_icon="💰",
    layout="wide"
)

# Custom CSS for better visuals
st.markdown("""
    <style>
    .stApp {
        background-color: #f7f9fc;
    }
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E88E5;
        text-align: center;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #444;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# --------------------------------------
# Header
# --------------------------------------
st.markdown("<h1 class='main-title'>💳 Fraud Detection using Machine Learning</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Interactive dashboard to explore, train, and predict fraudulent transactions</p>", unsafe_allow_html=True)
st.markdown("---")

# --------------------------------------
# Tabs
# --------------------------------------
tabs = st.tabs(["🏠 Overview", "📊 Data", "🔍 Correlation", "🧠 Model Training", "🎯 Predict"])

# --------------------------------------
# Tab 1: Overview
# --------------------------------------
with tabs[0]:
    st.subheader("About this Project")
    st.write("""
    This dashboard demonstrates a **Fraud Detection ML pipeline** trained on the IEEE-CIS dataset.  
    It includes:
    - Data Cleaning and Preprocessing  
    - Encoding Categorical Variables  
    - Handling Imbalance using **SMOTE**  
    - Model Training using **XGBoost**  
    - Interactive Evaluation and Predictions  
    """)

    st.info("You can use the sidebar to download the dataset or upload your own file!")

# --------------------------------------
# Sidebar
# --------------------------------------
st.sidebar.header("⚙️ Settings")
download_data = st.sidebar.button("⬇️ Download IEEE Dataset")
upload_data = st.sidebar.file_uploader("Or Upload Your Dataset (Parquet/CSV)", type=["parquet", "csv"])
train_model = st.sidebar.button("🚀 Train Model")
show_corr = st.sidebar.checkbox("Show Correlation Heatmap")
predict_mode = st.sidebar.checkbox("Enable Prediction Mode")

# --------------------------------------
# Data Handling
# --------------------------------------
@st.cache_data
def load_data_from_drive():
    file_id = '1PaIZ1U2f6fDOvI75s52-xLQH5C9yNCxv'
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'ieee_fraud_data.parquet'
    gdown.download(url, output, quiet=False)
    return pd.read_parquet(output, engine='pyarrow')

df = None
if download_data:
    with st.spinner("Downloading dataset..."):
        df = load_data_from_drive()
        st.success("✅ Dataset downloaded and loaded!")

elif upload_data is not None:
    with st.spinner("Loading uploaded data..."):
        if upload_data.name.endswith(".parquet"):
            df = pd.read_parquet(upload_data, engine='pyarrow')
        else:
            df = pd.read_csv(upload_data)
        st.success("✅ Dataset loaded successfully!")

# --------------------------------------
# Tab 2: Data
# --------------------------------------
with tabs[1]:
    if df is not None:
        st.subheader("Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)
        st.write(f"Shape: `{df.shape}`")
        st.write("Column Types:")
        st.write(df.dtypes.value_counts())
    else:
        st.warning("⚠️ Please download or upload a dataset first.")

# --------------------------------------
# Tab 3: Correlation
# --------------------------------------
with tabs[2]:
    if df is not None and show_corr:
        st.subheader("🔍 Correlation Heatmap")
        corr = df.select_dtypes(include=['number']).corr()
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(corr, cmap='coolwarm')
        st.pyplot(fig)
    else:
        st.info("Enable 'Show Correlation Heatmap' from the sidebar.")

# --------------------------------------
# Tab 4: Model Training
# --------------------------------------
with tabs[3]:
    if df is not None and train_model:
        st.subheader("🧠 Model Training and Evaluation")
        with st.spinner("Preprocessing and training the model..."):
            df = df.copy()
            df.set_index('TransactionID', inplace=True)

            # Fill missing numeric and categorical values
            for col in df.select_dtypes(include=['number']).columns:
                df[col].fillna(df[col].median(), inplace=True)
            for col in df.select_dtypes(include=['object']).columns:
                mode = df[col].mode()
                if not mode.empty:
                    df[col].fillna(mode[0], inplace=True)

            # Encode and get dummies
            if 'P_emaildomain' in df.columns:
                le = LabelEncoder()
                df['P_emaildomain'] = le.fit_transform(df['P_emaildomain'])
            if {'ProductCD', 'card4', 'card6', 'M6'}.issubset(df.columns):
                df = pd.get_dummies(df, columns=['ProductCD', 'card4', 'card6', 'M6'], drop_first=True)

            X = df.drop(columns=['isFraud'])
            y = df['isFraud']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            classifier = XGBClassifier(eval_metric='logloss', random_state=42)
            pipeline = ImbPipeline([
                ('scaler', StandardScaler()),
                ('smote', SMOTE(random_state=42)),
                ('classifier', classifier)
            ])
            pipeline.fit(X_train, y_train)

            y_pred = pipeline.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)

            st.success("✅ Model training complete!")

            # Confusion Matrix
            st.subheader("📊 Confusion Matrix")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Fraud', 'Fraud'], yticklabels=['Not Fraud', 'Fraud'])
            st.pyplot(fig)

            # Classification Report
            st.subheader("📈 Classification Report")
            st.dataframe(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose())

            st.balloons()
    else:
        st.info("Click **🚀 Train Model** in the sidebar to begin training.")

# --------------------------------------
# Tab 5: Predict
# --------------------------------------
with tabs[4]:
    if df is not None and predict_mode:
        st.subheader("🎯 Live Fraud Prediction")
        st.markdown("Enter a few feature values to predict whether a transaction is fraudulent.")
        st.info("Note: For simplicity, this uses the trained model's feature schema.")

        sample_data = {}
        for col in df.select_dtypes(include=['number']).columns[:10]:  # show only first 10 numeric features
            sample_data[col] = st.number_input(f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))

        if st.button("Predict Fraud Probability"):
            x_input = pd.DataFrame([sample_data])
            pred = pipeline.predict(x_input)[0]
            st.success("Fraudulent Transaction 🚨" if pred == 1 else "Legitimate Transaction ✅")

    elif not predict_mode:
        st.info("Enable **Prediction Mode** from the sidebar to try live predictions.")
