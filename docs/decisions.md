# Architecture Decision Records


## Orchestration via Airflow (MWAA in prod)

- Context: Need dynamic monthly runs across thousands of clients with branching (retrain vs reuse) and visibility.
- Decision: Use Airflow DAGs with TaskGroups. Production target is MWAA.
- Consequences: Clear Python-native DAG logic and UI but adds managed-service cost.

## Model Registry via MLflow with "champion" alias

- Context: Must track model lineage and retrieve the current model per client reliably.
- Decision: Use MLflow Model Registry. Track per-client model names and use a stable alias ("champion") for the current model.
- Consequences: Easy rollback and stable model URI. Requires MLflow hosting and access control.

## Simulation parallelism via Ray

- Context: Monte Carlo simulation is the heaviest step and must scale horizontally.
- Decision: Use Ray for distributed execution and shared model broadcasting (single put, many tasks).
- Consequences: Fast parallelism and local dev parity. Requires version alignment between Ray client and cluster.

## Model training via SageMaker (spot)

- Context: Each client needs a LightGBM model trained monthly; scaling to thousands requires elastic, per-job compute
- Decision: Use SageMaker Training Jobs with spot instances in the client region.
- Consequences: Auto-termination and cost savings, but adds startup latency and AWS-specific integration.

## Retrain decision mechanism

- Context: Monthly retraining is wasteful when degradation is typically low.
- Decision: Evaluate the latest model on the new test set and retrain only if balanced accuracy drops beyond a threshold.
- Consequences: Retrainign logic is data-driven

## Data residency

- Context: Customer data must remain in the origin region.
- Decision: Keep data processing, training, and simulation in-region. Only metadata and metrics cross regions.
- Consequences: Multi-region infrastructure and coordination are required, but compliance is preserved.
