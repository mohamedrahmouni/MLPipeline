# Documentation

Technical documentation for the ML Pipeline project.

---

## Documents

| Document | Description |
|----------|-------------|
| [architecture.md](architecture.md) | C4 diagrams (Context → Container → Component), Monte Carlo vectorization deep-dive, Ray broadcasting, performance benchmarks |
| [decisions.md](decisions.md) | Architecture Decision Records — rationale for Airflow, MLflow, Ray, SageMaker, and data residency choices |
| [cost_estimate.md](cost_estimate.md) | AWS monthly cost breakdown for 2,000 clients (~$1,730/month, ~$0.87/client) |
| [PRESENTATION_OUTLINE.md](PRESENTATION_OUTLINE.md) | Scalability analysis, throughput calculations, regional architecture, and trade-offs |

---

## Architecture at a Glance

**Per-client pipeline (orchestrated by Airflow):**

```
Data Prep (DuckDB / Snowflake)
  → Session-aware Train/Test Split
    → Retrain Check (MLflow champion model)
      → Train (LightGBM + MLflow) or Reuse
        → Monte Carlo Simulation (Ray cluster)
          → Results to S3 / Snowflake
```

**Key design decisions:**
- Airflow for orchestration (not compute) — heavy work on SageMaker and Ray
- Model reuse: ~65% of clients skip training each month
- Ray `ray.put()` for zero-copy model broadcasting across workers
- NumPy broadcasting for vectorized perturbation generation (no Python loops)
- Regional data residency — only metadata crosses regions
