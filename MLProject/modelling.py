import os
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import argparse

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, log_loss, confusion_matrix
)


def log_metrics(y_true, y_pred, y_proba):
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
    recall = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')
    auc = roc_auc_score(y_true, y_proba[:, 1])  # <- this is the fix
    loss = log_loss(y_true, y_proba)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", auc)
    mlflow.log_metric("log_loss", loss)
    mlflow.log_metric("true_positive", tp)
    mlflow.log_metric("true_negative", tn)
    mlflow.log_metric("false_positive", fp)
    mlflow.log_metric("false_negative", fn)

    print(f"Run finished. Accuracy: {acc:.4f} | F1 Score: {f1:.4f} | AUC: {auc:.4f}")


def train_and_log_model(model_name, model, mlflow_log_fn, X_train, X_test, y_train, y_test, params):
    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("model_name", model_name)
        for param, value in params.items():
            mlflow.log_param(param, value)
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("n_classes", len(y_train.unique()))

        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)

        log_metrics(y_test, preds, proba)
        mlflow_log_fn(model, "model")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=5)
    args = parser.parse_args()

    n_estimators = args.n_estimators
    max_depth = args.max_depth

    # Load and preprocess the dataset
    data_path = "student_depression_preprocessing.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"File not found: {data_path}")

    df = pd.read_csv(data_path)

    y = df['Depression']
    X = df.drop(columns=[
        'Depression',
        'Have you ever had suicidal thoughts ?_Yes',
        'Have you ever had suicidal thoughts ?_No',
        'Total_Stress', 'Satisfaction_Balance', 'Pressure_Balance',
        'Stress_Balance_Ratio', 'Age_Group', 'CGPA_Category'
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Set common parameters
    n_estimators = 100
    max_depth = 5

    # Train and log Logistic Regression
    logistic_model = LogisticRegression(max_iter=1000)
    train_and_log_model("logistic_regression", logistic_model, mlflow.sklearn.log_model,
                        X_train, X_test, y_train, y_test, {})

    # Train and log Random Forest
    rf_model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    train_and_log_model("random_forest", rf_model, mlflow.sklearn.log_model,
                        X_train, X_test, y_train, y_test,
                        {"n_estimators": n_estimators, "max_depth": max_depth})

    # Train and log XGBoost
    xgb_model = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth,
                              use_label_encoder=False, eval_metric='logloss', random_state=42)
    train_and_log_model("xgboost", xgb_model, mlflow.xgboost.log_model,
                        X_train, X_test, y_train, y_test,
                        {"n_estimators": n_estimators, "max_depth": max_depth})
