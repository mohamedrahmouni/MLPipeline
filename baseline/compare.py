"""
Side-by-side comparison: Old (Dataiku) vs New (AWS) architecture.

Simulates pipeline timing based on Problem.md production numbers:
  Data prep:   avg 3 min, up to 11 min   (Snowflake UDTF)
  Split:       avg 1 min, up to 3 min    (Snowpark)
  Training:    avg 7 min, up to 23 min   (LightGBM)
  Simulation:  avg 12 min, up to 20 min  (Monte Carlo)

No external services needed (no MLflow, no Ray). Runs in ~10 seconds.

Usage: uv run python baseline/compare.py
"""
import hashlib
import os
import random
import time
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed

# 1 production-minute = SCALE seconds of demo wall-clock.
# Keeps the comparison fast while preserving realistic relative timings.
DEMO_SCALE = 0.05

# Production step durations from Problem.md (minutes)
STEP_SPECS = {
    "data_prep":  (3, 11),   # (avg, max)
    "split":      (1, 3),
    "training":   (7, 23),
    "simulation": (12, 20),
}

# New architecture parameters
REUSE_RATE = 0.65           # 65% of clients skip training
RAY_SIM_SPEEDUP = 2.0      # Ray cluster vs 3 joblib processes
DEMO_PARALLEL = 5           # concurrent clients in demo
PRODUCTION_PARALLEL = 50    # concurrent clients in production


def _step_minutes(step, row_count, max_rows=200_000):
    """Simulate one step's production duration (minutes), scaled by data volume."""
    avg, mx = STEP_SPECS[step]
    size_factor = min(row_count / max_rows, 1.0)
    base = avg + (mx - avg) * size_factor * 0.3
    noise = random.gauss(0, (mx - avg) * 0.1)
    return round(max(avg * 0.8, min(base + noise, mx)), 1)


def _run_client(cfg, *, always_retrain=False, sim_speedup=1.0):
    """Simulate one client's pipeline. Returns step timings in production-minutes."""
    cid = cfg["client_id"]
    rows = cfg["row_count"]

    prep = _step_minutes("data_prep", rows)
    split = _step_minutes("split", rows)

    # Deterministic retrain decision per client (stable across runs)
    stable_seed = int(hashlib.sha256(cid.encode()).hexdigest(), 16) % (2**32)
    rng = random.Random(stable_seed)
    needs_retrain = always_retrain or rng.random() > REUSE_RATE

    train = _step_minutes("training", rows) if needs_retrain else 0.0
    sim = round(_step_minutes("simulation", rows) / sim_speedup, 1)

    total = prep + split + train + sim
    time.sleep(total * DEMO_SCALE)

    action = "RETRAIN" if needs_retrain else "REUSE  "
    print(f"  [{cid:<16}] {action}  {total:5.0f} min"
          f"  (prep={prep:.0f}  split={split:.0f}  train={train:.0f}  sim={sim:.0f})")
    return {
        "client_id": cid, "retrained": needs_retrain, "total_min": total,
        "prep": prep, "split": split, "train": train, "sim": sim,
    }


# ---------------------------------------------------------------------------
# Old architecture: Dataiku — sequential, always retrain, 3 joblib processes
# ---------------------------------------------------------------------------

def run_old(clients):
    print(f"\n{'='*70}")
    print("OLD ARCHITECTURE  (Dataiku — sequential, always retrain)")
    print(f"{'='*70}\n")

    results = [_run_client(c, always_retrain=True, sim_speedup=1.0)
               for c in clients]

    total_min = sum(r["total_min"] for r in results)
    avg_min = total_min / len(results)
    throughput = int(1440 / avg_min)  # 1 client at a time

    print(f"\n  Sequential total: {total_min:.0f} min"
          f"  |  Avg per client: {avg_min:.0f} min"
          f"  |  Max throughput: ~{throughput} clients/day")
    return {
        "total_min": total_min, "avg_min": avg_min,
        "throughput": throughput, "retrained": len(results),
        "reused": 0, "results": results,
    }


# ---------------------------------------------------------------------------
# New architecture: AWS — parallel, intelligent reuse, Ray simulation
# ---------------------------------------------------------------------------

def run_new(clients):
    print(f"\n{'='*70}")
    print("NEW ARCHITECTURE  (AWS — parallel, intelligent reuse, Ray)")
    print(f"{'='*70}\n")

    results = []
    with ThreadPoolExecutor(max_workers=DEMO_PARALLEL) as pool:
        futures = {pool.submit(_run_client, c,
                               always_retrain=False,
                               sim_speedup=RAY_SIM_SPEEDUP): c
                   for c in clients}
        for f in as_completed(futures):
            results.append(f.result())

    avg_min = sum(r["total_min"] for r in results) / len(results)
    throughput = int((1440 / avg_min) * PRODUCTION_PARALLEL)
    retrained = sum(1 for r in results if r["retrained"])
    reused = len(results) - retrained

    print(f"\n  Avg per client: {avg_min:.0f} min"
          f"  |  Projected throughput (@{PRODUCTION_PARALLEL} parallel): ~{throughput} clients/day")
    print(f"  Retrained: {retrained}  |  Reused: {reused}")
    return {
        "avg_min": avg_min, "throughput": throughput,
        "retrained": retrained, "reused": reused, "results": results,
    }


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def print_comparison(old, new, n_clients):
    tp_gain = new["throughput"] / old["throughput"] if old["throughput"] else 0
    time_saved = old["avg_min"] - new["avg_min"]
    reuse_pct = new["reused"] / n_clients * 100

    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"\n  {'Metric':<40} {'Old':>10} {'New':>10}")
    print(f"  {'-'*62}")
    print(f"  {'Avg pipeline time / client (min)':<40} {old['avg_min']:>10.0f} {new['avg_min']:>10.0f}")
    print(f"  {'Max throughput (clients/day)':<40} {old['throughput']:>10,} {new['throughput']:>10,}")
    print(f"  {'Throughput gain':<40} {'—':>10} {tp_gain:>9.0f}x")
    print(f"  {'Models retrained':<40} {old['retrained']:>10} {new['retrained']:>10}")
    print(f"  {'Models reused':<40} {old['reused']:>10} {new['reused']:>10}")
    print(f"  {'Reuse rate':<40} {'0%':>10} {reuse_pct:>9.0f}%")
    print(f"  {'Avg time saved / client (min)':<40} {'—':>10} {time_saved:>9.0f}")
    print()
    print("  Key drivers:")
    print(f"    - Parallelism: {PRODUCTION_PARALLEL} concurrent clients (vs 1 sequential)")
    print(f"    - Model reuse: {reuse_pct:.0f}% skip training (saves 7-23 min each)")
    print(f"    - Ray simulation: {RAY_SIM_SPEEDUP:.0f}x faster than 3 joblib processes")
    print()


if __name__ == "__main__":
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "configs", "clients.yaml")
    with open(cfg_path) as f:
        clients = yaml.safe_load(f)["clients"]

    print(f"\nSimulating {len(clients)} clients"
          f"  (1 production-minute = {DEMO_SCALE}s demo time)\n")

    old = run_old(clients)
    new = run_new(clients)
    print_comparison(old, new, len(clients))
