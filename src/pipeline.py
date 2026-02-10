"""End-to-end pipeline runner for a single client. Called by Airflow DAG tasks."""
import mlflow
from data_gen import generate_client_data
from data_prep import prepare_data
from split import split_train_test
from train import train_model
from registry import check_retrain_needed, log_retrain_decision, get_model_name
from simulate import run_monte_carlo_simulation
from config import get_mlflow_tracking_uri


def run_client_pipeline(client_config, skip_simulation=False):
    """Execute the full monthly pipeline for one client. Returns results dict."""
    client_id = client_config["client_id"]
    print(f"\n{'='*60}\nStarting pipeline for {client_id}\n{'='*60}\n")

    # 1. Data extraction
    raw_data = generate_client_data(client_config)

    # 2. Data preparation (DuckDB)
    prepared = prepare_data(raw_data, client_config)

    # 3. Session-aware train/test split
    train_df, test_df = split_train_test(prepared, client_config)

    # 4. Retrain check
    needs_retrain, model_uri, eval_metrics, decision_meta = check_retrain_needed(
        test_df, client_config
    )
    log_retrain_decision(
        client_id,
        needs_retrain,
        eval_metrics,
        decision_reason=decision_meta.get("decision_reason"),
        degradation_delta=decision_meta.get("degradation_delta"),
        month=decision_meta.get("month"),
    )

    # 5. Train or reuse
    if needs_retrain:
        print(f"[{client_id}] Step 5: RETRAINING")
        model, train_metrics = train_model(train_df, test_df, client_config)
    else:
        print(f"[{client_id}] Step 5: REUSING existing model")
        mlflow.set_tracking_uri(get_mlflow_tracking_uri())
        model = mlflow.lightgbm.load_model(model_uri)
        train_metrics = eval_metrics

    # 6. Monte Carlo simulation
    sim_rows = 0
    if not skip_simulation:
        sim_results = run_monte_carlo_simulation(model, prepared, client_config)
        sim_rows = len(sim_results)
    else:
        print(f"[{client_id}] Step 6: simulation SKIPPED")

    print(f"[{client_id}] Pipeline complete.")
    return {
        "client_id": client_id, "needs_retrain": needs_retrain,
        "train_metrics": train_metrics, "simulation_rows": sim_rows,
    }


# ---------------------------------------------------------------------------
# Airflow task wrappers (each regenerates data — in production these read from S3)
# ---------------------------------------------------------------------------

def _gen_data(cfg):
    return prepare_data(generate_client_data(cfg), cfg)


def run_data_prep_task(client_config):
    prepared = _gen_data(client_config)
    return f"Data prep complete: {len(prepared)} rows"


def run_split_task(client_config):
    prepared = _gen_data(client_config)
    train_df, test_df = split_train_test(prepared, client_config)
    return f"Split complete: train={len(train_df)}, test={len(test_df)}"


def run_retrain_check_task(client_config):
    prepared = _gen_data(client_config)
    _, test_df = split_train_test(prepared, client_config)
    needs_retrain, model_uri, metrics, decision_meta = check_retrain_needed(
        test_df, client_config
    )
    log_retrain_decision(
        client_config["client_id"],
        needs_retrain,
        metrics,
        decision_reason=decision_meta.get("decision_reason"),
        degradation_delta=decision_meta.get("degradation_delta"),
        month=decision_meta.get("month"),
    )
    return needs_retrain


def run_train_task(client_config):
    prepared = _gen_data(client_config)
    train_df, test_df = split_train_test(prepared, client_config)
    _, metrics = train_model(train_df, test_df, client_config)
    return f"Training complete: accuracy={metrics['test_balanced_accuracy']:.4f}"


def run_simulate_task(client_config):
    client_id = client_config["client_id"]
    mlflow.set_tracking_uri(get_mlflow_tracking_uri())
    model_name = get_model_name(client_id)
    model_uri = f"models:/{model_name}@champion"
    model = mlflow.lightgbm.load_model(model_uri)

    prepared = _gen_data(client_config)
    result = run_monte_carlo_simulation(model, prepared, client_config)
    return f"Simulation complete: {len(result)} rows"
