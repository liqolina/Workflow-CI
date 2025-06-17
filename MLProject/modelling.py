import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    median_absolute_error, max_error, explained_variance_score
)
from xgboost import XGBRegressor

import mlflow
import mlflow.sklearn
import mlflow.xgboost

from dotenv import load_dotenv

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100

def max_absolute_error(y_true, y_pred):
    return np.max(np.abs(y_true - y_pred))

def log_model_metrics(y_true, y_pred, training_time=None):
    metrics = {
        "r2_score": r2_score(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "mape": mean_absolute_percentage_error(y_true, y_pred),
        "explained_variance": explained_variance_score(y_true, y_pred),
        "median_absolute_error": median_absolute_error(y_true, y_pred),
        "max_absolute_error": max_absolute_error(y_true, y_pred),
        "max_error": max_error(y_true, y_pred),
    }
    if training_time is not None:
        metrics["training_time"] = training_time
    return metrics

def plot_predictions(y_true, y_pred, model_name, output_dir):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'Predicted vs Actual ({model_name})')
    plt.grid(True)

    plot_path = os.path.join(output_dir, f"{model_name}_prediksi.png")
    plt.savefig(plot_path)
    plt.close()
    return plot_path

def train_and_log_model(model, model_name, X_train, X_test, y_train, y_test, feature_names=None, params=None):
    with mlflow.start_run(run_name=model_name) as run:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = log_model_metrics(y_test, y_pred)

        mlflow.log_param("model_type", model_name)
        if params:
            for key, val in params.items():
                mlflow.log_param(key, val)

        for key, val in metrics.items():
            mlflow.log_metric(key, val)

        input_example = X_train[:5]
        if model_name.lower().startswith("xgboost"):
            mlflow.xgboost.log_model(model, model_name, input_example=input_example)
        else:
            mlflow.sklearn.log_model(model, model_name, input_example=input_example)

        output_dir = "Actual_VS_Predicted_Graph"
        os.makedirs(output_dir, exist_ok=True)
        plot_path = plot_predictions(y_test, y_pred, model_name, output_dir)
        mlflow.log_artifact(plot_path)

        # Debug
        print(f"MLflow artifact root: {os.getenv('MLFLOW_ARTIFACT_ROOT', 'mlruns')}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Plot path exists: {os.path.exists(plot_path)}")
        print(f"Run ID: {run.info.run_id}")
        
        # Debug: Cek artifacts di DagsHub
        client = mlflow.tracking.MlflowClient()
        try:
            artifacts = client.list_artifacts(run.info.run_id)
            print(f"Artifacts for run_id {run.info.run_id}: {[a.path for a in artifacts]}")
        except Exception as e:
            print(f"Failed to list artifacts for run_id {run.info.run_id}: {str(e)}")
        
        # Cetak run_id untuk GitHub Actions
        run_id = run.info.run_id
        print(f"MLFLOW_RUN_ID={run_id}")

        print(
            f"{model_name} - "
            f"R²: {metrics['r2_score']:.4f}, RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}, "
            f"MAPE: {metrics['mape']:.2f}%, Explained Variance: {metrics['explained_variance']:.4f}, "
            f"MedAE: {metrics['median_absolute_error']:.4f}, MaxAE: {metrics['max_absolute_error']:.4f}"
        )

def configure_mlflow():
    load_dotenv()
    tracking_uri = 'https://dagshub.com/liqolina/Eksperimen_SML_LutfiAundrieHermawan.mlflow'
    username = 'liqolina'
    token = os.getenv('TOKEN_DAGSHUB')

    if not token:
        raise EnvironmentError("TOKEN_DAGSHUB environment variable is not set.")

    os.environ['MLFLOW_TRACKING_URI'] = tracking_uri
    os.environ['MLFLOW_TRACKING_USERNAME'] = username
    os.environ['MLFLOW_TRACKING_PASSWORD'] = token
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("Student_Depression_Prediction")

def main():
    configure_mlflow()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, 'student_depression_preprocessing.csv')

    if not os.path.exists(data_path):
        raise FileNotFoundError("File 'student_depression_preprocessing.csv' tidak ditemukan.")

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

    models = [
        ("Linear Regression", LinearRegression()),
        ("Random Forest", RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)),
        ("XGBoost", XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42))
    ]

    for model_name, model in models:
        train_and_log_model(model, model_name, X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()
