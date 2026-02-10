"""
LightGBM training with MLflow logging.

Production deployment (SageMaker):
  The training script would be packaged as a SageMaker training job that:
  1. Reads train/test data from S3
  2. Checks MLflow for champion model
  3. Evaluates champion on new test data
  4. If degradation > threshold: trains new model, logs to MLflow, saves to S3
  5. If performance OK: logs reuse decision, exits early (no training cost)

  This combines the retrain check + training into one job, avoiding a separate
  orchestration step. The job always runs, but exits early when reuse is safe.
"""
import time
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
from mlflow.tracking import MlflowClient
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from config import get_mlflow_tracking_uri, CATEGORICAL_COLS, EXCLUDE_COLS
from registry import get_model_name


def train_model(train_df, test_df, client_config):
    """Train LightGBM, log params + metrics + model artifact to MLflow. Returns (model, metrics)."""
    client_id = client_config["client_id"]
    params = client_config.get("model_params", {})

    feature_cols = [c for c in train_df.columns if c not in EXCLUDE_COLS]
    X_train, y_train = train_df[feature_cols].copy(), train_df["target"]
    X_test, y_test = test_df[feature_cols].copy(), test_df["target"]

    # LightGBM handles categorical features natively
    cat_cols = [c for c in CATEGORICAL_COLS if c in X_train.columns]
    for col in cat_cols:
        X_train[col] = X_train[col].astype("category")
        X_test[col] = X_test[col].astype("category")

    # Simulate production training delay (scaled down)
    train_time = min(7 + (len(X_train) / 1_000_000) * 16, 23)
    print(f"[{client_id}] Training on {len(X_train):,} rows... ({train_time:.0f}s simulated)")
    time.sleep(train_time / 60)

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="auc",
        categorical_feature=cat_cols,
    )

    # Compute metrics on both splits
    def _metrics(X, y, prefix):
        pred, proba = model.predict(X), model.predict_proba(X)[:, 1]
        return {
            f"{prefix}_balanced_accuracy": balanced_accuracy_score(y, pred),
            f"{prefix}_auc": roc_auc_score(y, proba),
            f"{prefix}_precision": precision_score(y, pred, zero_division=0),
            f"{prefix}_recall": recall_score(y, pred, zero_division=0),
            f"{prefix}_f1": f1_score(y, pred, zero_division=0),
        }

    metrics = {**_metrics(X_train, y_train, "train"), **_metrics(X_test, y_test, "test")}

    # Log to MLflow
    mlflow.set_tracking_uri(get_mlflow_tracking_uri())
    mlflow.set_experiment(f"{client_id}_experiment")
    with mlflow.start_run(run_name=f"{client_id}_training"):
        mlflow.log_params({"client_id": client_id, "region": client_config["region"], **params})
        mlflow.log_metrics(metrics)
        model_name = get_model_name(client_id)
        mlflow.lightgbm.log_model(model, name="model", registered_model_name=model_name)

        client = MlflowClient()
        versions = client.search_model_versions(f"name='{model_name}'")
        model_version = str(max(int(v.version) for v in versions))

        client.set_model_version_tag(
            model_name,
            model_version,
            "baseline_balanced_accuracy",
            f"{metrics['test_balanced_accuracy']:.6f}",
        )
        client.set_registered_model_alias(model_name, "champion", model_version)

    print(f"[{client_id}] Training complete — test balanced_accuracy={metrics['test_balanced_accuracy']:.4f}")
    return model, metrics
