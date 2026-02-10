# Architecture Decision Records


## Orchestration via Airflow (MWAA in prod)

- Context: Need dynamic monthly runs across thousands of clients with branching (retrain vs reuse) and visibility.
- Decision: Use Airflow DAGs with TaskGroups. Production target is MWAA.
- Consequences: Clear Python-native DAG logic and UI but adds managed-service cost.

## Model Registry via MLflow with "champion" alias

- Context: Must track model lineage and retrieve the current model per client reliably.
- Decision: Use MLflow Model Registry. Track per-client model names and use a stable alias ("champion") for the current model.
- Consequences: Easy rollback and stable model URI. Requires MLflow hosting and access control.

## Simulation parallelism via Ray remote tasks

- Context: Monte Carlo simulation is the heaviest step (2M rows x 300 perturbations = 600M predictions) and must scale horizontally.
- Decision: Use `@ray.remote` tasks for chunk-based parallelism. Model broadcast once via `ray.put()` to shared memory (avoiding per-task serialization). Data chunked and distributed as parallel tasks.
- Consequences: Fine-grained task scheduling, zero-copy model sharing across workers. Requires Ray cluster and client version alignment.

## Model training via SageMaker (spot)

- Context: Each client needs a LightGBM model trained monthly; scaling to thousands requires elastic, per-job compute
- Decision: Use SageMaker Training Jobs with spot instances in the client region.
- Consequences: Auto-termination and cost savings, but adds startup latency and AWS-specific integration.

## Retrain decision mechanism

- Context: Monthly retraining is wasteful when degradation is typically low.
- Decision: Evaluate the latest model on the new test set and retrain only if balanced accuracy drops beyond a threshold.
- Consequences: Retraining logic is data-driven and auditable via MLflow.

## Data residency

- Context: Customer data must remain in the origin region.
- Decision: Keep data processing, training, and simulation in-region. Only metadata and metrics cross regions.
- Consequences: Multi-region infrastructure and coordination are required, but compliance is preserved.
