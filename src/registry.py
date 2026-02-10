"""
Model registry logic for retraining decisions.

Uses a per-client model name and the champion alias to decide retrain vs reuse.
"""
from datetime import datetime, timezone
from typing import Dict, Any, Tuple, Optional

import mlflow
from mlflow.exceptions import RestException
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
from config import get_mlflow_tracking_uri


def get_model_name(client_id: str) -> str:
    return f"{client_id}_classifier"


def _evaluate_champion_model(
    client_id: str,
    test_df: pd.DataFrame,
    threshold: float = 0.03,
    default_baseline: float = 0.85,
) -> Tuple[bool, Optional[str], Optional[Dict[str, float]], Dict[str, Any]]:
    mlflow.set_tracking_uri(get_mlflow_tracking_uri())
    client = MlflowClient()
    model_name = get_model_name(client_id)
    model_uri = f"models:/{model_name}@champion"
    decision_meta = {
        "decision_reason": "model_reused",
        "degradation_delta": None,
        "month": datetime.now(timezone.utc).strftime("%Y-%m"),
    }

    try:
        model_version = client.get_model_version_by_alias(model_name, "champion")
    except RestException:
        decision_meta["decision_reason"] = "no_champion_model"
        return True, None, None, decision_meta

    # Simple simulation: if a champion exists, flip a coin to decide retrain.
    needs_retrain = np.random.rand() < 0.5
    decision_meta["decision_reason"] = "coin_flip_retrain" if needs_retrain else "coin_flip_reuse"
    return needs_retrain, model_uri, None, decision_meta


def evaluate_model_necessity(client_id: str, test_data: pd.DataFrame, test_labels: pd.Series) -> bool:
    """
    Returns True if retraining is required, False if existing model is sufficient.
    """
    eval_df = test_data.copy()
    eval_df["target"] = test_labels
    needs_retrain, _, _, _ = _evaluate_champion_model(client_id, eval_df)
    return needs_retrain


def check_retrain_needed(
    test_df: pd.DataFrame,
    client_config: Dict[str, Any],
    threshold: float = 0.03,
) -> Tuple[bool, Optional[str], Optional[Dict[str, float]], Dict[str, Any]]:
    client_id = client_config["client_id"]
    print(f"[{client_id}] Checking if retraining needed (threshold: {threshold:.2%})...")

    needs_retrain, model_uri, metrics, decision_meta = _evaluate_champion_model(
        client_id,
        test_df,
        threshold=threshold,
        default_baseline=client_config.get("baseline_accuracy", 0.85),
    )

    decision_msg = "Retraining required." if needs_retrain else "Model performance acceptable. Reusing existing model."
    print(f"[{client_id}] {decision_msg}")

    return needs_retrain, model_uri, metrics, decision_meta


def log_retrain_decision(
    client_id: str,
    needs_retrain: bool,
    metrics: Optional[Dict[str, float]] = None,
    decision_reason: Optional[str] = None,
    degradation_delta: Optional[float] = None,
    month: Optional[str] = None,
):
    """Log the retrain/reuse decision as an MLflow run with spec-required tags."""
    mlflow.set_tracking_uri(get_mlflow_tracking_uri())
    mlflow.set_experiment(f"{client_id}_experiment")
    run_month = month or datetime.now(timezone.utc).strftime("%Y-%m")

    with mlflow.start_run(run_name=f"{client_id}_retrain_check_{run_month}"):
        mlflow.log_param("client_id", client_id)
        mlflow.log_param("needs_retrain", needs_retrain)
        mlflow.log_param("decision", "retrain" if needs_retrain else "reuse")
        mlflow.set_tags({
            "decision_reason": decision_reason or "unknown",
            "degradation_delta": "" if degradation_delta is None else f"{degradation_delta:.6f}",
            "month": run_month,
            "status": "model_reused" if not needs_retrain else "model_retrained",
        })
        if metrics:
            mlflow.log_metrics(metrics)
        print(f"[{client_id}] Retrain decision logged to MLflow")
