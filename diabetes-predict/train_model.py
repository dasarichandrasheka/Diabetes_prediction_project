import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
DATA_PATH = "data/diabetes.csv"
MODEL_OUT = "model.joblib"
SCALER_OUT = "scaler.joblib"
METRICS_OUT = "metrics.json"
RANDOM_STATE = 42

df = pd.read_csv(DATA_PATH)
print("LOADED dataset shape:", df.shape)
print(df.head())

zeros_as_missing = ["glucose", "bloodpressure", "skinthickness", "insulin","bmi"]
for col in zeros_as_missing:
    if col in df.columns:
        df[col] = df[col].replace(0, np.nan)

df.fillna(df.median(), inplace=True)

x = df.drop(columns=["Outcome"])
y = df["Outcome"]

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

param_grid = {
    "n_estimators": [100, 200],
    "max_depth":[None, 6, 10],
    "min_samples_split":[2,5]
}
rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)

grid = GridSearchCV(rf, param_grid, cv=5, scoring="roc_auc", verbose=1)
grid.fit(x_train_scaled, y_train)

best_model = grid.best_estimator_
print("best params:", grid.best_params_)

y_pred = best_model.predict(x_test_scaled)
y_proba = best_model.predict_proba(x_test_scaled)[:, 1]

acc = accuracy_score(y_test, y_pred)
print("accuracy:", acc)

metrics = {
    "accurcy": float(accuracy_score(y_test, y_pred)),
    "precision": float(precision_score(y_test, y_pred)),
    "recall": float(recall_score(y_test, y_pred)),
    "f1": float(f1_score(y_test, y_pred)),
    "roc_auc": float(roc_auc_score(y_test, y_proba))
}
print("test mertics:", metrics)

with open(METRICS_OUT, "w") as f:
    json.dump(metrics,  f, indent=2)

joblib.dump(best_model,MODEL_OUT)
joblib.dump(scaler, SCALER_OUT)
print(f"saved model -> {MODEL_OUT}, scaler -> {SCALER_OUT}")

if hasattr(best_model, "feature_impotances_"):
    fi = pd.Series(best_model.feature_importances_, index=x.columns).sort_values(ascending=False)
    plt.figure(figsize=(8,5))
    sns.barplot(x=fi.values, y=fi.index)
    plt.title("feature_importances.png")
    print("saved feature_importances.png")



