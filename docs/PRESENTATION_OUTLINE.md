# Scalability Analysis & Design Rationale

Detailed breakdown of the problem constraints, solution approach, and production scaling calculations.

---

## Problem Statement & Constraints

### Current Limitations
1. **Sequential execution** → Only ~50 clients/day capacity
2. **Fixed resources** → 8 vCPU, 64 GB RAM (shared across all tasks)
3. **Always retrain** → Wasteful when degradation is only 2-3%/month
4. **Unoptimized simulation** → 3 joblib processes, partial vectorization

### Hard Constraints
1. **Data residency** → Customer data MUST stay in origin region
2. **Scale requirement** → Thousands of clients on Day 1 of month
3. **Monthly cadence** → All models run on first day
4. **Session-aware splitting** → No train/test leakage

### Current Pipeline (Sequential, per client)
```
Data Prep (avg 3, up to 11 min) → Split (avg 1, up to 3 min) → Train (avg 7, up to 23 min) → Simulate (avg 12, up to 20 min)

Average total: ~23 min/client
Worst case:    ~57 min/client
```

---

## Solution Approach Overview

### Key Strategic Decisions
1. **Airflow as orchestrator** → Cron-triggered DAG on Day 1 of month, splits work into per-client TaskGroups, provides high-level execution visibility, logging, and retry management
2. **Elastic compute** → SageMaker (training) + Ray (simulation) on AWS — pay-per-use, scales to zero when idle
3. **Intelligent retraining** → Evaluate champion model on new data; skip training when performance is still acceptable (~65% reuse)
4. **Distributed simulation** → Ray cluster with model broadcasting via `ray.put()` and vectorized NumPy perturbations
5. **Keep Snowflake** → Data prep stays where the data lives (familiar to the team, performant, compliant)

### Architecture Philosophy
- **Orchestrate, don't compute, in Airflow** — Airflow triggers and monitors; heavy work runs on SageMaker and Ray
- **Minimize change** where the current system works (Snowflake prep, LightGBM model)
- **Maximize parallelism** — concurrent TaskGroups, multi-cluster Snowflake, distributed Ray workers
- **Optimize costs** — spot instances, model reuse, auto-scaling to zero on Days 2-30

---

## Proposed Architecture (C4 Context)

```
┌─────────────┐
│Data Scientists│──────► AWS + Snowflake ML Pipeline ◄──── End Clients
└─────────────┘                   │
                                  ▼
                    ┌─────────────┴──────────────┐
                    │                            │
            ┌───────▼──────┐           ┌────────▼──────┐
            │   Snowflake  │           │   AWS Cloud   │
            │ (Data Layer) │           │  (Compute)    │
            │              │           │               │
            │ • Regional   │           │ • MWAA        │
            │ • UDTF prep  │           │ • SageMaker   │
            │ • Compliant  │           │ • Ray/EKS     │
            └──────────────┘           └───────────────┘
                                              │
                                    ┌─────────▼─────────┐
                                    │  MLflow Registry  │
                                    │  (Metrics + Models)│
                                    └───────────────────┘
```

**Key Components:**
- **Snowflake**: Data stays in region, UDTF preprocessing
- **MWAA (Airflow)**: Orchestration with TaskGroups for parallelism
- **SageMaker**: Elastic training on spot instances
- **Ray on EKS**: Distributed Monte Carlo simulation
- **MLflow**: Model registry + metrics tracking + audit trail

---

## 🔄 **Pipeline Flow**

### Single Client Pipeline (Parallel across 50 clients)

```
┌─────────────────────────────────────────────────────────────┐
│  ORCHESTRATION: Airflow/MWAA (DAG runs on Day 1)           │
└────────┬─────────────────────────────────────────┬──────────┘
         │                                         │
    ┌────▼─────┐                              ┌───▼──────┐
    │TaskGroup │ Client 1                     │TaskGroup │ Client N
    │          │                              │          │
    │ ┌────────▼────────┐                    │ (parallel) │
    │ │ 1. Data Prep    │ Snowflake UDTF     │            │
    │ │    + Split      │ (avg 4, up to 14m) │            │
    │ └────────┬────────┘                    │            │
    │          │                              │            │
    │ ┌────────▼────────┐                    │            │
    │ │ 2. Export to S3 │ Train/test data    │            │
    │ │                 │ staged regionally  │            │
    │ └────────┬────────┘                    │            │
    │          │                              │            │
    │ ┌────────▼────────────────────────┐    │            │
    │ │ 3. SageMaker Job                │    │            │
    │ │    Reads data from S3           │    │            │
    │ │    ┌─────────────────────────┐  │    │            │
    │ │    │ a. Load champion model  │  │    │            │
    │ │    │    from MLflow/S3       │  │    │            │
    │ │    │ b. Evaluate on new test │  │    │            │
    │ │    │    data                 │  │    │            │
    │ │    │ c. BRANCH:              │  │    │            │
    │ │    │    degradation ≥ 5%?    │  │    │            │
    │ │    │    YES → Train LightGBM │  │    │            │
    │ │    │    NO  → Reuse champion │  │    │            │
    │ │    │ d. Save model to S3     │  │    │            │
    │ │    │ e. Log decision+metrics │  │    │            │
    │ │    │    to MLflow            │  │    │            │
    │ │    └─────────────────────────┘  │    │            │
    │ └────────┬────────────────────────┘    │            │
    │          │                              │            │
    │ ┌────────▼────────┐                    │            │
    │ │ 4. Monte Carlo  │ Ray cluster        │            │
    │ │    Simulation   │ loads model from S3│            │
    │ │                 │ (avg 12, up to 20m)│            │
    │ └────────┬────────┘                    │            │
    │          │                              │            │
    │ ┌────────▼────────┐                    │            │
    │ │ 5. Save Results │ S3 → Snowflake     │            │
    │ └─────────────────┘                    │            │
    └─────────────────────────────────────────────────────┘
```

**Key flow: S3 is the data handoff layer between all services**
- Snowflake exports train/test data → **S3 (regional)**
- SageMaker reads data from S3, runs retrain check + training **inside the same job**
- SageMaker saves model → S3, logs decision → MLflow
- Ray loads model from S3, runs simulation, saves results → S3
- Results loaded back from S3 → Snowflake

**Critical Innovation: Retrain Check Inside SageMaker**
- The SageMaker job handles both evaluation and training in one step
- Loads champion model, evaluates on new test data from S3
- If balanced accuracy degradation < 5% → **Reuse** (exit early, save compute)
- If degradation ≥ 5% → **Retrain** LightGBM on training data from S3
- Either way: decision + metrics logged to MLflow for audit

---

## 🚀 **Model Reuse**

### The Problem
- Old approach: **Always retrain** every month for every client
- Reality: Model degradation is only 2-3% per month
- Waste: ~700+ training jobs/month that aren't needed

### The Solution: Data-Driven Retraining
```python
def check_retrain_needed(test_df, client_config, threshold=0.05):
    champion_model = load_model(f"models:/{client_id}@champion")
    new_metrics = evaluate(champion_model, test_df)

    if new_metrics['balanced_accuracy'] < (baseline - threshold):
        return True  # Retrain needed
    else:
        return False  # Reuse existing model
```

### Impact
| Metric | Value |
|--------|-------|
| **Reuse rate** | ~65% (based on 2-3% degradation/month) |
| **Training jobs saved** | 1,300 / 2,000 clients |
| **Time saved per reuse** | ~7 min average (skip training), up to ~23 min worst case |
| **Total time saved** | ~150 hours/month (1,300 reuses × 7 min avg) |
| **Cost saved** | ~$45/month on training |


---

## ⚡ **Vectorized Monte Carlo**

### The Challenge
- **2M rows × 300 perturbations** = **600M predictions** per client
- Old: 3 joblib processes, row-by-row loops
- Bottleneck: Model serialization + Python loops

### The Solution: Full Vectorization + Ray Broadcasting

#### Step 1: Broadcast Model Once
```python
# Model sent ONCE to each Ray worker's shared memory
model_ref = ray.put(model)  # Zero-copy sharing across tasks
```

#### Step 2: Generate All Perturbations at Once (NumPy Broadcasting)
```python
# Shape transformations:
num_features: (n_rows, n_features)              # e.g., (10, 25)
improvement_factors: (n_rows, n_pert, n_feat)   # e.g., (10, 300, 25)

# Broadcasting magic: add new axis, multiply
improved = num_features[:, np.newaxis, :] * improvement_factors
# Result: (10, 300, 25) - all perturbations for all rows in parallel!
```

#### Step 3: Single Batch Prediction
```python
# Reshape to flat: (n_rows * n_pert, n_features)
improved_flat = improved.reshape(-1, n_features)  # (3000, 25)

# ONE model call for entire batch
all_preds = model.predict_proba(improved_flat)    # (3000,)

# Reshape back: (n_rows, n_pert)
preds_matrix = all_preds.reshape(n_rows, n_pert)  # (10, 300)
```



## Monte Carlo Simulation Architecture

### Visual: How Vectorization Works

```
┌─────────────────────────────────────────────────────────────────┐
│  RAY CLUSTER (Auto-scaled on EKS)                              │
│                                                                 │
│  ┌──────────────┐                                              │
│  │ Ray Head     │  ◄── Airflow submits job                     │
│  │              │                                              │
│  │  model_ref = ray.put(model)  ← Broadcast ONCE              │
│  └──────┬───────┘                                              │
│         │                                                       │
│         │ Distribute chunks to workers                         │
│    ┌────┴─────┬──────────┬──────────┐                         │
│    │          │          │          │                          │
│ ┌──▼────┐ ┌──▼────┐ ┌──▼────┐ ┌───▼────┐                     │
│ │Worker1│ │Worker2│ │Worker3│ │Worker N│ (10-50 workers)      │
│ │       │ │       │ │       │ │        │                      │
│ │ Model │ │ Model │ │ Model │ │ Model  │ ← Shared memory      │
│ │ (ref) │ │ (ref) │ │ (ref) │ │ (ref)  │   (not copied!)      │
│ └───┬───┘ └───┬───┘ └───┬───┘ └───┬────┘                     │
│     │         │         │         │                            │
│  Chunk1    Chunk2    Chunk3    ChunkN                         │
│  (200 rows)(200 rows)(200 rows)(200 rows)                     │
│     │         │         │         │                            │
│     └─────────┴─────────┴─────────┘                           │
│                   │                                             │
│            Results gathered                                    │
└────────────────────┼───────────────────────────────────────────┘
                     ▼
              ┌──────────────┐
              │ Concatenate  │
              │ All Results  │
              └──────────────┘
                     │
                     ▼
              ┌──────────────┐
              │   S3 Bucket  │
              │   (regional) │
              └──────────────┘
```

**Key Points:**
1. Model broadcast ONCE via `ray.put()` → all workers share read-only reference
2. NumPy broadcasting: `(n_rows, features)` → `(n_rows, n_pert, features)` without loops
3. Single batch prediction for all perturbations
4. Vectorized statistics calculation (mean, std, percentiles)

---

## Scalability Analysis

### Old vs New: Side-by-Side Comparison

| Dimension | Legacy Platform | New (AWS) |
|-----------|--------------|-----------|
| **Execution Model** | Sequential | Parallel (nb_clients TaskGroups) |
| **Data Prep** | S warehouse | S multi-cluster |
| **Training Compute** | Shared 4 vCPU / 32 GB | Dedicated per-client SageMaker |
| **Simulation** | 3 joblib processes | Ray cluster (10-50 nodes) |
| **Model Strategy** | Always retrain | 65% reuse, 35% retrain |
| **Scaling** | Manual, fixed | Auto-scale on Day 1 |
| **Daily Capacity** | ~50 clients | **~3,900 clients** |

### Throughput Calculation
```
Per-client pipeline times (from test spec):
  With training:    avg 23 min  (3 + 1 + 7 + 12)
  With reuse:       avg 16 min  (3 + 1 + ~0 + 12)  ← skip training step
  Blended (65% reuse): 0.65 × 16 + 0.35 × 23 = ~18.5 min average

Old capacity (sequential, single pipeline):
  (24h × 60 min) / 23 min = ~62 clients/day
  (with overhead → ~50 as stated in the spec)

New capacity (50 concurrent pipelines):
  (24h × 60 min / 18.5 min) × 50 parallel = ~3,900 clients/day

Scaling factor: ~78x improvement over current 50/day
```

### Cost Per Client Run
```
Total monthly cost: $1,730 for 2,000 clients
Cost per run: $0.87 per client

Breakdown:
- Snowflake (55%): $950 (data prep + splits)
- MWAA (26%): $450 (orchestration)
- Ray (4%): $75 (simulation)
- MLflow (5%): $85 (tracking)
- SageMaker (1.5%): $25 (700 training jobs with reuse)
- Other (8.5%): $145 (networking, monitoring, S3)
```

**Key insight:** Model reuse saves $45/month in training costs alone!

---

## Data Residency & Compliance

### Regional Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  REGION: US-EAST-1 (Customer A data)                       │
│                                                             │
│  Snowflake ──► S3 (us-east-1) ──► SageMaker (us-east-1)   │
│                                    │                        │
│                                    ▼                        │
│                               Ray (us-east-1)              │
│                                                             │
│  ✅ Data never leaves region                               │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  REGION: EU-CENTRAL-1 (Customer B data)                    │
│                                                             │
│  Snowflake ──► S3 (eu-central-1) ──► SageMaker (eu-cent-1)│
│                                       │                     │
│                                       ▼                     │
│                                  Ray (eu-central-1)        │
│                                                             │
│  ✅ Data never leaves region                               │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  CENTRAL: MLflow (any region)                              │
│                                                             │
│  ✅ Only metadata + metrics cross regions                  │
│  ✅ No customer data, only model performance stats         │
└─────────────────────────────────────────────────────────────┘
```

**Compliance guarantees:**
- Raw data stays in Snowflake (origin region)
- Training/simulation compute in same region
- Only lightweight metadata to central MLflow
- Airflow in main region coordinates (metadata only)


---

## Limitations & Trade-offs

### 1. Operational Complexity — More Services to Learn and Manage
The proposed architecture introduces **5 new services** compared to the legacy platform:

- Today the team works in **one tool** — this proposal asks them to operate **five**
- Each service has its own failure modes, upgrade cycles, and debugging patterns
- Risk: Team may slow down initially while ramping up on new tooling

**Mitigation:** Start with managed services (MWAA, not self-hosted Airflow; EKS with KubeRay, not bare-metal Ray). Invest in runbooks and internal documentation. Plan 2-3 weeks of team onboarding before Phase 1 goes live.


> This solution trades **operational simplicity** for **scalability and cost efficiency**. The legacy setup is easy to operate but cannot scale. This proposal scales to thousands but requires a team comfortable with distributed systems. The question is: **is 50 clients/day acceptable, or is the scale requirement non-negotiable?** If it's non-negotiable, this complexity is the cost of solving it properly.

---
