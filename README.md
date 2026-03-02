# ML Pipeline — Scalable Monthly Training & Monte Carlo Simulation

![Python 3.12](https://img.shields.io/badge/python-3.12-blue)
![Airflow](https://img.shields.io/badge/Airflow-2.x-017CEE?logo=apacheairflow)
![MLflow](https://img.shields.io/badge/MLflow-3.9-0194E2?logo=mlflow)
![Ray](https://img.shields.io/badge/Ray-2.53-028CF0)
![LightGBM](https://img.shields.io/badge/LightGBM-4.6-green)
![Docker Compose](https://img.shields.io/badge/Docker_Compose-ready-2496ED?logo=docker)

An end-to-end ML platform prototype that scales a monthly training + Monte Carlo simulation pipeline from ~50 runs/day to thousands of runs/day. Fully runnable locally via `docker compose`.

### Key Capabilities

- **Parallel client pipelines** — Airflow DAG with per-client TaskGroups running concurrently
- **Intelligent model reuse** — Evaluate champion model on new data; retrain only when degraded (~65% skip training)
- **Distributed Monte Carlo simulation** — Ray cluster with vectorized NumPy broadcasting and `ray.put()` model sharing
- **Full MLOps lifecycle** — MLflow model registry, metrics tracking, retrain/reuse audit trail
- **Production-mapped architecture** — Local Docker services map 1:1 to AWS managed services (SageMaker, MWAA, EKS)

---

## Architecture

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   Airflow (MWAA) │────▶│  SageMaker/Train │────▶│  Ray Simulation  │
│   Orchestration  │     │  LightGBM + MLflow│     │  Monte Carlo     │
└──────────────────┘     └──────────────────┘     └──────────────────┘
         │                        │                        │
         ▼                        ▼                        ▼
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Snowflake/DuckDB│     │  MLflow Registry │     │  S3 / Results    │
│  Data Prep       │     │  Model Versioning│     │  Storage         │
└──────────────────┘     └──────────────────┘     └──────────────────┘
```

**Per-client pipeline:** Data Prep → Session-aware Split → Retrain Check → Train/Reuse → Monte Carlo Simulation

See [docs/architecture.md](docs/architecture.md) for C4 diagrams, vectorization deep-dive, and performance benchmarks.

---

## Tech Stack

| Layer | Local (Docker) | Production (AWS) |
|-------|---------------|-----------------|
| Orchestration | Airflow standalone | MWAA (Managed Airflow) |
| Data Prep | DuckDB | Snowflake UDTF |
| Training | PythonOperator + LightGBM | SageMaker Spot Instances |
| Simulation | Ray (1 head + 3 workers) | Ray on EKS, auto-scaled |
| Model Registry | MLflow on filesystem | MLflow on ECS + S3 |
| Storage | Local filesystem | S3 regional buckets |

---

## Quick Start

**Prerequisites:** Docker & Docker Compose v2.0+, 8 GB RAM, Python 3.12, [uv](https://docs.astral.sh/uv/)

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
| Service | URL |
|---------|-----|
| Airflow | http://localhost:8080 |
| MLflow | http://localhost:5000 |
| Ray Dashboard | http://localhost:8265 |

### Run the Pipeline

**Option 1 — Airflow DAG** (recommended)

1. Open Airflow UI → enable and trigger `monthly_ml_pipeline`
2. Watch per-client TaskGroups execute in parallel
3. Check MLflow for logged models, metrics, and retrain/reuse decisions

**Option 2 — Comparison script**

```bash
uv sync
MLFLOW_URI=http://localhost:5000 uv run python baseline/compare.py
```

Runs legacy (sequential, always-retrain) vs new (parallel, intelligent-reuse, Ray) side by side with real timing.

---

## Project Structure

```
├── docker-compose.yml          # Airflow, MLflow, Ray, Postgres
├── Dockerfile.airflow          # Airflow image with ML deps
├── Dockerfile.ray              # Ray workers with matching ML deps
├── configs/
│   ├── clients.yaml            # Client configs (region, model params)
│   └── project.yaml            # MLflow tracking URI
├── dags/
│   └── pipeline_dag.py         # Airflow DAG with branching logic
├── src/
│   ├── config.py               # Shared configuration & constants
│   ├── data_gen.py             # Synthetic data generator (31-col schema)
│   ├── data_prep.py            # DuckDB preprocessing (Snowflake stand-in)
│   ├── split.py                # Session-aware train/test split
│   ├── train.py                # LightGBM training + MLflow logging
│   ├── registry.py             # Retrain decision logic + audit trail
│   ├── simulate.py             # Distributed Monte Carlo (Ray)
│   └── pipeline.py             # Pipeline runner + Airflow task wrappers
├── baseline/
│   ├── old_pipeline.py         # Legacy constraints documentation
│   ├── compare.py              # Old vs new comparison runner
│   └── README.md               # Comparison usage guide
└── docs/
    ├── architecture.md         # C4 diagrams, vectorization deep-dive
    ├── decisions.md            # Architecture Decision Records
    └── cost_estimate.md        # AWS cost breakdown (~$1,730/mo)
```

---

## Design Highlights

### Intelligent Model Reuse
Each month, the pipeline evaluates the champion model on fresh test data. If performance is acceptable (degradation < threshold), training is skipped entirely — saving compute and time for ~65% of clients.

### Vectorized Monte Carlo Simulation
The simulation avoids Python loops by using NumPy broadcasting to generate all perturbations at once:

```python
# (n_rows, 1, n_features) * (n_rows, n_perturbations, n_features)
# → (n_rows, n_perturbations, n_features) — zero Python loops
improved = num_features[:, np.newaxis, :] * improvement_factors
```

Combined with a single batch `model.predict_proba()` call and Ray distribution across workers, this achieves **~300x speedup** over naive nested loops.

### Ray Model Broadcasting
The trained model (~180 MB) is uploaded once to Ray's object store via `ray.put()`, then shared across all workers via lightweight references — reducing network transfer from 45 GB to 9 GB.

---

## Documentation

- [Architecture](docs/architecture.md) — C4 diagrams, simulation deep-dive, performance benchmarks
- [Decisions](docs/decisions.md) — Architecture Decision Records (ADRs)
- [Cost Estimate](docs/cost_estimate.md) — AWS monthly cost breakdown (~$1,730 for 2,000 clients)

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
