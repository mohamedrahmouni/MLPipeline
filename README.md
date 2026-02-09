# ML Pipeline Migration Showcase

**Dataiku+Snowflake to AWS+Snowflake migration prototype**

A working end-to-end prototype that demonstrates scaling a monthly ML + Monte Carlo simulation pipeline from ~50 runs/day to thousands of runs/day, runnable entirely via `docker compose`.

---

## What's Inside

- **Full pipeline**: Data prep (DuckDB/Snowflake) -> Session-aware split -> Train (LightGBM) -> Monte Carlo simulation (Ray)
- **Intelligent model reuse**: Retrain only when performance degrades; skip training for ~65% of clients
- **Parallel execution**: Multiple clients run concurrently via Airflow TaskGroups
- **Ray-based simulation**: Distributed Monte Carlo with shared model broadcasting via `ray.put()`
- **Airflow orchestration**: Real DAG with BranchPythonOperator for retrain vs reuse decision
- **Baseline comparison**: Old vs new architecture throughput comparison script

---

## Quick Start

**Prerequisites**: Docker & Docker Compose v2.0+, 8GB RAM, uv (Python package manager)

```bash
# Build and start all services
echo "AIRFLOW_UID=$(id -u)" > .env
docker compose build && docker compose up -d

# Wait ~2 min for services to be healthy
docker compose ps

# Get Airflow login credentials
docker compose logs airflow 2>&1 | grep "Password for user"
```

**Service UIs:**
- Airflow: http://localhost:8080
- MLflow: http://localhost:5000
- Ray Dashboard: http://localhost:8265

### Run the Demo

**Option 1: Comparison script** (recommended)

```bash
uv sync
MLFLOW_URI=http://localhost:5000 uv run python baseline/compare.py
```

Shows old (sequential, always retrain) vs new (parallel, intelligent reuse) side by side.

**Option 2: Airflow DAG**

1. Open Airflow UI at http://localhost:8080
2. Enable and trigger the `monthly_ml_pipeline` DAG
3. Watch TaskGroups execute in parallel
4. Check MLflow UI for logged models and metrics

---

## Repository Structure

```
MLPipeline/
├── docker-compose.yml        # Airflow, MLflow, Ray, Postgres
├── Dockerfile.airflow        # Airflow image with ML dependencies
├── configs/
│   ├── clients.yaml          # Client configs (region, model params)
│   └── project.yaml          # MLflow tracking URI
├── dags/
│   └── pipeline_dag.py       # Airflow DAG with branching logic
├── src/
│   ├── data_gen.py           # Synthetic data (31 columns, stands in for Snowflake)
│   ├── data_prep.py          # DuckDB preprocessing (stands in for Snowflake UDTF)
│   ├── split.py              # Session-aware train/test split
│   ├── train.py              # LightGBM training + MLflow logging
│   ├── registry.py           # Retrain decision logic + MLflow audit
│   ├── simulate.py           # Monte Carlo simulation (Ray or local fallback)
│   └── pipeline.py           # End-to-end pipeline runner + Airflow task wrappers
├── baseline/
│   ├── old_pipeline.py       # Dataiku-constrained sequential simulator
│   └── compare.py            # Old vs new comparison
└── docs/
    ├── architecture.md       # C4 diagrams, old vs new comparison
    └── cost_estimate.md      # AWS cost breakdown (~$1,730/month for 2,000 clients)
```

---

## Documentation

- [Architecture](docs/architecture.md) - C4 diagrams, pipeline flow, old vs new comparison
- [Decisions](docs/decisions.md) - ADRs
- [Cost Estimate](docs/cost_estimate.md) - Monthly AWS cost breakdown

---

## Troubleshooting

```bash
# Check service health
docker compose ps

# Airflow logs
docker compose logs airflow

# Restart MLflow if connection errors
docker compose restart mlflow

# DAG not appearing — wait 2 min for parsing, then check
docker compose logs airflow 2>&1 | grep -i error
```
