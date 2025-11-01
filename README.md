💳 Fraud Detection using Machine Learning (IEEE-CIS Dataset)
🚀 Overview

This project focuses on detecting fraudulent transactions using Machine Learning, leveraging the IEEE-CIS Fraud Detection dataset from Kaggle.
Fraud detection is a critical problem in the financial industry, and this project demonstrates how data preprocessing, feature engineering, and model optimization can work together to build an effective fraud detection system.

🧠 Project Objective

The goal of this project is to classify online transactions as fraudulent or genuine using machine learning — specifically, the XGBoost algorithm, which is highly efficient for imbalanced classification problems.

📊 Dataset

Source: IEEE-CIS Fraud Detection Dataset (Kaggle)

This dataset contains transaction-level data, including numerical, categorical, and identity-based features.
Key details:

Rows: ~590K+ transactions

Features: Transaction amount, card details, product codes, device info, and more

Target: isFraud (1 → Fraudulent, 0 → Genuine)

⚙️ Workflow

The project follows a structured data science pipeline:

1️⃣ Data Preprocessing

Handled missing values using median imputation for numeric columns

Dropped irrelevant columns (like TransactionID)

Encoded categorical variables using Label Encoding

Performed outlier analysis to stabilize the distribution

2️⃣ Exploratory Data Analysis (EDA)

Visualized feature distributions and fraud frequency

Identified relationships between transaction amount, time, and fraud likelihood

Examined correlations to detect redundant features

3️⃣ Feature Engineering

Created new derived features like TransactionAmt_to_mean_card1

Normalized skewed numerical features

Balanced the dataset using undersampling/oversampling techniques

4️⃣ Model Training (XGBoost)

Implemented the XGBoost classifier, tuned hyperparameters using GridSearchCV

Used early stopping to avoid overfitting

Trained the model on preprocessed data

5️⃣ Model Evaluation

Evaluated using:

Accuracy

Precision / Recall / F1-score

ROC-AUC Curve

Focused on Recall and F1-score, since false negatives (missed frauds) are costlier than false positives.

📈 Results

Model Used: XGBoost

Accuracy: ~99%

ROC-AUC Score: ~0.97

The model effectively distinguishes between fraudulent and non-fraudulent transactions with minimal false negatives.

🧩 Tech Stack

Language: Python 🐍

Libraries:

pandas, numpy, matplotlib, seaborn — for data processing and visualization

scikit-learn — for preprocessing and evaluation

xgboost — for model training and tuning

joblib — for model saving

🖥️ Folder Structure
Fraud_Detection/
│
├── fraud_detection_model.ipynb    # Main notebook
├── data/                          # Dataset folder (not included due to size)
├── models/                        # Saved models (XGBoost)
├── README.md                      # Project documentation
└── requirements.txt               # Dependencies

⚡ Future Work

🌐 Build a Streamlit-based web app for real-time fraud prediction

📦 Deploy the model using Flask + AWS

📊 Add SHAP explainability to understand feature importance

🧩 Implement deep learning models (like LSTM for temporal features)

🧾 Key Learnings

Handling missing data with appropriate strategies (median imputation)

Understanding the impact of outliers on mean vs. median

Tackling imbalanced datasets effectively

Using XGBoost for high-dimensional, tabular data

🧑‍💻 Author

Rahul Gupta
🎓 B.Tech – Computer Science | SRM Institute of Science and Technology
💼 Aspiring Machine Learning & Backend Engineer
🔗 LinkedIn
 | GitHub

📜 License

This project is licensed under the MIT License — feel free to use, modify, and build upon it for your own learning or projects.
