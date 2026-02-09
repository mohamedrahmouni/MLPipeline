# Architecture Documentation

Proposed AWS+Snowflake architecture for scaling the monthly ML pipeline from ~50 runs/day to thousands of runs/day. Follows C4 modeling principles.

---

## C4 Level 1: System Context

```mermaid
graph TB
    DS[Data Scientists] --> System[ML Pipeline System]
    Clients[End Clients] -.data.-> System
    System --> Snowflake[Snowflake Data Platform]
    System --> AWS[AWS Cloud Services]
    System --> MLflow[MLflow Registry]

    Snowflake -.customer data stays in region.-> System

    style System fill:#e1f5ff
    style Snowflake fill:#29b5e8
    style AWS fill:#ff9900
```

- **Snowflake**: Customer data storage and preprocessing. Data stays in origin region.
- **AWS**: Orchestration (MWAA/Airflow), training (SageMaker), simulation (Ray).
- **MLflow**: Central model registry, metrics tracking, retrain decision audit trail.

---

## C4 Level 2: Container Diagram

```mermaid
graph TB
    subgraph "Orchestration"
        MWAA[MWAA / Airflow<br/>DAG Scheduler]
    end

    subgraph "Data Layer - Snowflake regional"
        SF_Prep[Snowflake UDTF<br/>Data Prep + Split]
    end

    subgraph "Compute Layer - AWS regional"
        SageMaker[SageMaker Training<br/>LightGBM on Spot]
        Ray[Ray Cluster<br/>Monte Carlo Simulation]
    end

    subgraph "Storage"
        S3[S3 Regional<br/>Models + Results]
        MLflow[MLflow<br/>Metrics + Registry]
    end

    MWAA -->|trigger prep| SF_Prep
    SF_Prep -->|export train/test| S3
    MWAA -->|launch training| SageMaker
    SageMaker -->|read data| S3
    SageMaker -->|save model| S3
    SageMaker -->|log metrics| MLflow
    MWAA -->|submit simulation| Ray
    Ray -->|load model| S3
    Ray -->|save results| S3
    S3 -->|load results back| SF_Prep

    style MWAA fill:#ff9900
    style SageMaker fill:#ff9900
    style S3 fill:#569a31
    style SF_Prep fill:#29b5e8
    style Ray fill:#028cf0
```

---

## C4 Level 3: Single Client Pipeline

```mermaid
graph TB
    Start([Monthly Trigger]) --> Prep[Data Prep + Split<br/>Snowflake UDTF]
    Prep --> Export[Export to S3]
    Export --> Check{Retrain Check<br/>MLflow Registry}

    Check -->|degraded| Train[Train LightGBM<br/>SageMaker Spot]
    Check -->|OK| Reuse[Load Existing Model<br/>from S3]

    Train --> Save[Save Model to S3<br/>+ Log to MLflow]
    Save --> Simulate
    Reuse --> Simulate[Monte Carlo Simulation<br/>Ray Cluster]

    Simulate --> Results[Save Results<br/>S3 + Snowflake]
    Results --> End([Done])

    style Check fill:#ffd700
    style Train fill:#ff9900
    style Reuse fill:#90ee90
    style Simulate fill:#028cf0
```

**Steps:**

1. **Data Prep + Split** (Snowflake) — UDTF extracts last month's data, imputes nulls, casts types. Session-aware train/test split (no leakage). Output: 31-column train/test sets exported to S3.
2. **Retrain Check** (MLflow) — Load champion model, evaluate on new test set. If performance degraded beyond threshold, retrain; otherwise reuse existing model.
3. **Train** (SageMaker, conditional) — Spot instance trains LightGBM on 1M rows, validates on 500k. Logs metrics (balanced accuracy, AUC, precision, recall, F1) to MLflow. Saves artifact to regional S3.
4. **Monte Carlo Simulation** (Ray) — Sample 2M rows, 300 perturbation evaluations per row. Model broadcast once via `ray.put()` shared memory. Results saved to S3, loaded back into Snowflake.

---

## Old vs New Architecture

| | Old (Dataiku) | New (AWS + Snowflake) |
|---|---|---|
| **Execution** | Sequential, 1 client at a time | Parallel, 50 concurrent clients |
| **Data prep** | S warehouse, sequential | S warehouse, multi-cluster concurrent |
| **Training** | Shared 4 vCPU / 32 GB | Dedicated SageMaker spot instances |
| **Simulation** | 3 joblib processes | Ray cluster, 10-50 nodes |
| **Model strategy** | Always retrain | Retrain only when degraded |
| **Daily capacity** | ~50 clients | ~4,000+ clients |
| **Scaling** | Manual, fixed resources | Auto-scaling, burst on Day 1 |
| **Cost model** | Fixed enterprise license | Pay-per-use |

**Throughput math** (worst case):
- Pipeline per client: ~23 min
- With model reuse (~65% skip training): ~15 min average
- 50 concurrent pipelines: `(24h x 60 / 15) x 50 = 4,800 clients/day`

---

## Local Demo to Production Mapping

| Local (docker-compose) | Production (AWS) |
|---|---|
| Postgres | RDS PostgreSQL |
| Airflow standalone | MWAA (Managed Airflow) |
| MLflow on filesystem | MLflow on ECS + S3 |
| Ray head + 2 workers | Ray on EKS, auto-scaled, spot |
| DuckDB (fake Snowflake) | Snowflake regional warehouses |
| Synthetic data generation | Snowflake UDTF on real data |
| PythonOperator | SageMakerTrainingOperator |
| Local filesystem | S3 regional buckets |
| Manual DAG trigger | EventBridge cron (1st of month) |
