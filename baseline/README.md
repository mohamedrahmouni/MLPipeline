# Pipeline Comparison

Real comparison between old (Dataiku) and new (AWS) architectures using actual docker-compose services.

## What It Does

**OLD Architecture (Sequential)**:
- Sequential processing (one client at a time)
- Always retrains every model
- Local/sequential Monte Carlo simulation
- No intelligent model reuse

**NEW Architecture (Parallalel)**:
- Parallel processing (multiple clients concurrently)
- Model reuse (checks if retrain needed)
- Distributed Ray cluster for Monte Carlo simulation
- MLflow for model registry and versioning

## Prerequisites

**Python 3.12 required**

```bash
# Ensure you're using Python 3.12
python --version  # Should show 3.12.x

# Install dependencies
uv sync  # Installs ray[client] and other dependencies

# Start docker-compose services (except airflow)
docker compose up -d postgres mlflow ray-head ray-worker-1 ray-worker-2 ray-worker-3

# Wait for services to be ready
docker compose ps
```

## Usage

### Quick Demo (default)
```bash
# Uses 5 clients with 10x smaller data (fast ~2-3 minutes)
uv run python baseline/compare.py

# Monitor Ray tasks in real-time at http://localhost:8265
```

### Custom Scale Factor
```bash
# More clients, smaller data (faster)
uv run python baseline/compare.py --scale 0.05 --max-clients 5

# Fewer clients, larger data (closer to production)
uv run python baseline/compare.py --scale 0.5 --max-clients 2
```

### Full Production Scale
```bash
# All clients, full data volume (slow ~30-60 minutes)
uv run python baseline/compare.py --scale 1.0 --max-clients 0
```

### Adjust Parallelism
```bash
# Use more parallel workers for new architecture
uv run python baseline/compare.py --parallel-workers 10
```

## Scale Factor Explanation

The `--scale` parameter controls data volume:
- `1.0` = Full production scale (200K rows per client)
- `0.1` = 10x smaller (20K rows, 10x faster) **← default**
- `0.05` = 20x smaller (10K rows, 20x faster)
- `0.01` = 100x smaller (2K rows, 100x faster)

Lower scale = faster demo, but still shows relative performance differences.

## Output

The script measures and compares:
- Total wall-clock time
- Throughput (clients/minute)
- Data processing volume
- Model reuse rate
- Overall speedup

Example output:
```
✓ Connected to Ray cluster at ray://localhost:10001
  Ray nodes: 4 (1 head + 3 workers)
  Total CPUs: 24

COMPARISON SUMMARY
======================================================================

  Metric                                           Old          New
  ------------------------------------------------------------------
  Total wall-clock time (s)                       85.2         48.3
  Clients completed                                  5            5
  Throughput (clients/min)                         3.5          6.2
  Total rows processed                          57,500       57,500
  Models retrained                                   5            2
  Models reused                                      0            3
  Reuse rate                                        0%         60%
  ------------------------------------------------------------------
  WALL-CLOCK SPEEDUP                                 —         1.8x
  THROUGHPUT GAIN                                    —         1.8x
  TIME SAVED (s)                                     —         36.9

  Key architectural improvements:
    ✓ Parallel execution (1.8x throughput)
    ✓ Intelligent model reuse (60% skip training)
    ✓ Ray cluster for distributed simulation
    ✓ Efficient data processing with DuckDB
```

## How It Works

**Sequential Mode (Old)**:
1. Runs one client at a time
2. Always performs full training
3. Uses local simulation (no Ray)
4. Measures actual execution time

**Parallel Mode (New)**:
1. Connects to Ray cluster once (avoids race conditions)
2. Runs multiple clients concurrently
3. Checks MLflow registry for existing models
4. Only retrains if model degraded
5. Uses Ray cluster for distributed simulation
6. Each Ray task is named for monitoring (e.g., `client_alpha_sim_chunk_0`)
7. Measures actual execution time

## Ray Task Distribution

Each client's simulation is split into chunks:
- **Chunk size**: 256 rows per task (configurable via `BATCH_SIZE`)
- **1000 rows** → **4 Ray tasks** (chunks 0-3)
- All chunks for a client use that client's model
- Ray efficiently caches models on workers

**Example**: `client_alpha` with 1000 rows:
```
client_alpha → ray.put(model) → ObjectRef
  ├─ client_alpha_sim_chunk_0 (rows 0-255)
  ├─ client_alpha_sim_chunk_1 (rows 256-511)
  ├─ client_alpha_sim_chunk_2 (rows 512-767)
  └─ client_alpha_sim_chunk_3 (rows 768-999)
```

## Troubleshooting

**Ray connection errors**:
```bash
# Check Ray cluster is running
docker compose logs ray-head

# Rebuild Ray containers if needed
docker compose up -d --build ray-head ray-worker-1 ray-worker-2 ray-worker-3

# Check Ray dashboard at http://localhost:8265
```

**MLflow connection errors**:
```bash
# Check MLflow is running
docker compose logs mlflow
docker compose restart mlflow
```

**Python version mismatch**:
```bash
# Must use Python 3.12 to match Ray cluster
rm -rf .venv
uv venv --python 3.12
uv sync
```

**Out of memory**:
```bash
# Reduce scale factor or max clients
uv run python baseline/compare.py --scale 0.05 --max-clients 2
```

## Notes

- The comparison uses **real services** from docker-compose
- All timing is **actual wall-clock time**, not simulated
- First run may be slower (MLflow model initialization)
- Subsequent runs will show model reuse working
- Scale factor only affects data volume, not architectural differences
- **Ray Dashboard** at http://localhost:8265 shows task execution in real-time
- Each Ray task is named (e.g., `client_alpha_sim_chunk_0`) for easy monitoring
- Ray cluster: 1 head + 3 workers = 4 nodes total
- Models are efficiently broadcast using `ray.put()` and cached on workers
