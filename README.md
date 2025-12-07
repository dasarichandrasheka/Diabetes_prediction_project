# Diabetes_prediction_project
it's a simple ,minimal wed for diabetic prediction 
A Machine Learning web application built using Python, Scikit-Learn, and Streamlit to predict whether a person is likely to have Diabetes based on medical parameters.

This project uses the PIMA Indians Diabetes Dataset from Kaggle and demonstrates the complete ML pipeline — from data preprocessing to model deployment.

🔍 Project Overview

The goal of this project is to build a fast, accurate, and user-friendly diabetes prediction system that can be used by:

Healthcare professionals

Individuals assessing risk

Learning & teaching ML pipelines

Recruiters evaluating ML skills

The app takes patient details as input and predicts “Diabetic” or “Non-Diabetic” using a trained machine learning model.

🎯 Core Purpose of This Project

To demonstrate the end-to-end process of applying Machine Learning to a real-world healthcare dataset:

✔ Data Cleaning & Preprocessing

Handling missing values

Scaling features

Splitting data

✔ Model Training

Random Forest Classifier

Hyperparameter tuning

Handling nonlinear patterns

Performance evaluation (Accuracy, Precision, Recall, F1, ROC-AUC)

✔ Model Saving

Exporting the trained model and scaler using joblib

✔ Deployment

Interactive Streamlit web app

Clean UI for user predictions

End-to-end pipeline integrated with the trained model

This project showcases your skills in ML, software engineering, deployment, and UI design.

🧠 Tech Stack

Python

Pandas, NumPy

Scikit-Learn

Streamlit

Joblib

Matplotlib / Seaborn (optional for EDA)

GitHub / Streamlit Cloud (for deployment)

📊 Dataset

Source: PIMA Indians Diabetes Dataset (Kaggle)

Features used for prediction:

Pregnancies

Glucose Level

Blood Pressure

Skin Thickness

Insulin

BMI

Diabetes Pedigree Function

Age

Target:
Outcome → (1 = Diabetic, 0 = Non-Diabetic)

🏗 Project Structure
📂 diabetes-prediction-app
│── app.py                 # Streamlit front-end
│── train_model.py         # Training script
│── model.pkl              # Saved ML model
│── scaler.pkl             # Saved StandardScaler
│── requirements.txt       # Dependencies
│── README.md              # Project documentation

🖥 How to Run Locally
1️⃣ Clone the repo
git clone https://github.com/your-username/diabetes-prediction.git
cd diabetes-prediction

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Run Streamlit app
streamlit run app.py


Your app will open in the browser at:

http://localhost:8501

🌐 Live Demo

(Replace this link after deployment)

👉 https://your-app-name.streamlit.app

📈 Model Performance
Metric	Value
Accuracy	~85%
Precision	Good
Recall	Good
F1 Score	Stable
ROC-AUC	High

(Random Forest was chosen because it handles:
✔ nonlinear patterns
✔ noisy data
✔ unscaled features
✔ better generalization)

🧩 Key Learning Outcomes

Real-world dataset preprocessing

Selecting the right ML model

Avoiding common ML pitfalls

Pipeline creation (train → save → load → predict)

Building a Streamlit UI

Deploying ML models to cloud

This project is a perfect example of end-to-end ML engineering.


