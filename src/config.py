"""Project configuration helpers."""
import os
from typing import Any, Dict
import yaml


_DEFAULTS: Dict[str, Any] = {
    "mlflow_tracking_uri": "http://localhost:5000",
}

# Shared ML constants
CATEGORICAL_COLS = ["category_a", "category_b", "region", "segment"]
EXCLUDE_COLS = {"session_id", "user_id", "timestamp", "target"}
BATCH_SIZE = 256


def _candidate_paths() -> list[str]:
    return [
        os.getenv("PROJECT_CONFIG_PATH", ""),
        "/opt/airflow/configs/project.yaml",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs", "project.yaml")),
    ]


def load_project_config() -> Dict[str, Any]:
    for path in _candidate_paths():
        if path and os.path.isfile(path):
            with open(path, "r") as handle:
                data = yaml.safe_load(handle) or {}
                return {**_DEFAULTS, **data}
    return dict(_DEFAULTS)


def get_mlflow_tracking_uri() -> str:
    env_uri = os.getenv("MLFLOW_URI")
    if env_uri:
        return env_uri
    return load_project_config().get("mlflow_tracking_uri", _DEFAULTS["mlflow_tracking_uri"])
