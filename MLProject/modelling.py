import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from xgboost import XGBClassifier
import mlflow

def log_classification_metrics(y_true, y_pred, training_time):
    """Calculate and return classification metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
        "recall": recall_score(y_true, y_pred, average='weighted'),
        "f1_score": f1_score(y_true, y_pred, average='weighted'),
        "training_time": training_time
    }

def plot_confusion_matrix(y_true, y_pred, model_name, output_dir):
    """Generate and save confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title(f'Confusion Matrix ({model_name})')

    plot_path = os.path.join(output_dir, f"{model_name}_conf_matrix.png")
    plt.savefig(plot_path)
    plt.close()
    return plot_path

def plot_feature_importance(model, feature_names, model_name, output_dir):
    """Plot and save feature importance (if available)."""
    if not hasattr(model, "feature_importances_"):
        return None

    plt.figure(figsize=(10, 6))
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    sns.barplot(x=importances[indices], y=np.array(feature_names)[indices])
    plt.title(f'Feature Importance ({model_name})')
    plt.tight_layout()

    feat_path = os.path.join(output_dir, f"{model_name}_feature_importance.png")
    plt.savefig(feat_path)
    plt.close()
    return feat_path

def train_and_log_classifier(model, model_name, X_train, X_test, y_train, y_test, feature_names, params=None):
    """Train model, log metrics, confusion matrix, and feature importance to MLflow."""
    with mlflow.start_run(run_name=model_name):
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time

        y_pred = model.predict(X_test)
        metrics = log_classification_metrics(y_test, y_pred, training_time)

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

        output_dir = "Confusion_Matrix_Plots"
        os.makedirs(output_dir, exist_ok=True)

        plot_path = plot_confusion_matrix(y_test, y_pred, model_name, output_dir)
        mlflow.log_artifact(plot_path)

        feat_path = plot_feature_importance(model, feature_names, model_name, output_dir)
        if feat_path:
            mlflow.log_artifact(feat_path)

        print(
            f"{model_name} - "
            f"Acc: {metrics['accuracy']:.4f}, Prec: {metrics['precision']:.4f}, "
            f"Recall: {metrics['recall']:.4f}, F1: {metrics['f1_score']:.4f}"
        )

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


def configure_mlflow():
    """Configure MLflow tracking (DagsHub)."""
    load_dotenv()
    tracking_uri = 'https://dagshub.com/liqolina/Workflow-CI.mlflow'
    username = 'liqolina'
    token = os.getenv('DAGSHUB_TOKEN')

    if not token:
        raise EnvironmentError("TOKEN_DAGSHUB environment variable is not set.")

    os.environ['MLFLOW_TRACKING_URI'] = tracking_uri
    os.environ['MLFLOW_TRACKING_USERNAME'] = username
    os.environ['MLFLOW_TRACKING_PASSWORD'] = token
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("Student_Depression_Classification")
    
def main():
    configure_mlflow()
    
    # Load dataset
    data_path = "student_depression_preprocessing.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"File not found: {data_path}")
    
    df = pd.read_csv(data_path)

    # Target & features
    y = df['Depression']  # Should be categorical: 0/1 or similar
    X = df.drop(columns=[
        'Depression', 
        'Have you ever had suicidal thoughts ?_Yes', 
        'Have you ever had suicidal thoughts ?_No',
        'Total_Stress', 'Satisfaction_Balance', 'Pressure_Balance', 
        'Stress_Balance_Ratio', 'Age_Group', 'CGPA_Category'
    ])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define classifiers
    models = [
        ("Logistic Regression", LogisticRegression(max_iter=1000)),
        ("Random Forest", RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)),
        ("XGBoost", XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42, use_label_encoder=False, eval_metric='logloss'))
    ]

    for model_name, model in models:
        train_and_log_classifier(model, model_name, X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()
