"""
Airflow DAG — monthly ML pipeline.

Per client: data_prep >> split >> retrain_check >> [train | skip] >> simulate
Clients run in parallel (max_active_tasks).
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator, BranchPythonOperator
from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.sdk import TaskGroup
import yaml, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pipeline import run_data_prep_task, run_split_task, run_retrain_check_task, run_train_task, run_simulate_task

with open("/opt/airflow/configs/clients.yaml") as f:
    clients = yaml.safe_load(f)["clients"]


def _make_branch(cfg):
    def _decide(**_ctx):
        return f"{cfg['client_id']}.train" if run_retrain_check_task(cfg) else f"{cfg['client_id']}.skip_train"
    return _decide


with DAG(
    "monthly_ml_pipeline",
    default_args={"owner": "ml-platform", "retries": 0},
    description="Monthly ML training + Monte Carlo simulation",
    schedule="0 0 1 * *",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_tasks=5,
    tags=["ml", "monthly"],
) as dag:

    start = EmptyOperator(task_id="start")
    end = EmptyOperator(task_id="end")

    for cfg in clients:
        cid = cfg["client_id"]
        with TaskGroup(group_id=cid) as grp:
            prep      = PythonOperator(task_id="data_prep",     python_callable=run_data_prep_task, op_kwargs={"client_config": cfg})
            split     = PythonOperator(task_id="split",         python_callable=run_split_task,     op_kwargs={"client_config": cfg})
            check     = BranchPythonOperator(task_id="retrain_check", python_callable=_make_branch(cfg))
            train     = PythonOperator(task_id="train",         python_callable=run_train_task,     op_kwargs={"client_config": cfg})
            skip      = EmptyOperator(task_id="skip_train")
            simulate  = PythonOperator(task_id="simulate",      python_callable=run_simulate_task,  op_kwargs={"client_config": cfg},
                                       trigger_rule="none_failed_min_one_success")

            prep >> split >> check >> [train, skip] >> simulate

        start >> grp >> end
