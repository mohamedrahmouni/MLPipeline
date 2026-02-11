"""
Real comparison: Old (sequential, always-retrain) vs New (parallel, intelligent-reuse, Ray).

This script runs ACTUAL pipelines using your docker-compose services:
  - MLflow for model registry
  - Ray cluster for distributed simulation
  - DuckDB for data prep
  - LightGBM for training

Usage:
  # Run with default demo scale (reduces data by 10x, uses 3 clients)
  uv run python baseline/compare.py

  # Run with custom scale factor
  uv run python baseline/compare.py --scale 0.05

  # Run with all clients (no limit)
  uv run python baseline/compare.py --scale 1.0 --max-clients 0
"""
import argparse
import os
import sys
import time
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pipeline import run_client_pipeline
from data_gen import generate_client_data
from data_prep import prepare_data
from split import split_train_test
from train import train_model
from registry import check_retrain_needed, log_retrain_decision
from simulate import run_monte_carlo_simulation
import mlflow
from config import get_mlflow_tracking_uri

# Simulation factor: reduces data volume for faster demo
# 1.0 = full production scale, 0.1 = 10x smaller (10x faster)
DEFAULT_SCALE = 0.1
DEFAULT_MAX_CLIENTS = 3  # Limit clients for demo (0 = no limit)


def scale_client_config(cfg, scale_factor):
    """Reduce row_count by scale_factor for faster demo runs."""
    scaled = cfg.copy()
    scaled["row_count"] = max(1000, int(cfg["row_count"] * scale_factor))
    return scaled


def run_client_sequential_mode(client_config, scale_factor):
    """
    OLD ARCHITECTURE simulation: sequential, always retrain, no Ray.

    This mimics the old Dataiku approach:
    - Always retrains (ignores intelligent reuse)
    - Uses local simulation instead of Ray cluster
    - Runs one step at a time
    """
    cfg = scale_client_config(client_config, scale_factor)
    client_id = cfg["client_id"]

    print(f"\n[{client_id}] Starting OLD-STYLE pipeline (always retrain, no Ray)...")
    start = time.time()

    # 1. Data extraction
    raw_data = generate_client_data(cfg)

    # 2. Data preparation
    prepared = prepare_data(raw_data, cfg)

    # 3. Train/test split
    train_df, test_df = split_train_test(prepared, cfg)

    # 4. ALWAYS retrain (no intelligent reuse check)
    print(f"[{client_id}] Training model (forced retrain)...")
    model, train_metrics = train_model(train_df, test_df, cfg)

    # 5. Simulation WITHOUT Ray (local/sequential)
    print(f"[{client_id}] Running simulation (local/sequential, no Ray)...")
    # Use a simple local simulation instead of Ray
    sim_results = _run_local_simulation(model, prepared, cfg)

    elapsed = time.time() - start
    print(f"[{client_id}] OLD-STYLE complete in {elapsed:.1f}s")

    return {
        "client_id": client_id,
        "retrained": True,
        "time_seconds": elapsed,
        "rows": len(prepared),
        "sim_rows": len(sim_results),
    }


def _run_local_simulation(model, data_df, client_config, n_perturbations=100):
    """
    Local (non-Ray) Monte Carlo simulation.
    Mimics old approach with reduced perturbations for speed.
    """
    import pandas as pd
    import numpy as np
    from config import EXCLUDE_COLS, CATEGORICAL_COLS

    client_id = client_config["client_id"]

    # Sample smaller subset for demo
    sample = data_df.sample(n=min(500, len(data_df)), random_state=42)

    # Prepare features
    feature_cols = [c for c in sample.columns if c not in EXCLUDE_COLS]
    cat_cols = [c for c in CATEGORICAL_COLS if c in feature_cols]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    # Convert categorical columns
    for col in cat_cols:
        sample[col] = sample[col].astype("category")

    # Base predictions
    base_features = sample[feature_cols].copy()
    base_preds = model.predict_proba(base_features)[:, 1]

    results = []
    for i in range(len(sample)):
        num_values = sample.iloc[i][num_cols].astype(float).values
        noise = np.random.randn(n_perturbations, len(num_cols)) * 0.1
        perturbed_num = num_values + noise

        # Create perturbed dataframe
        perturbed_df = pd.DataFrame(perturbed_num, columns=num_cols)
        for col in cat_cols:
            perturbed_df[col] = sample.iloc[i][col]
            perturbed_df[col] = perturbed_df[col].astype("category")

        perturbed_df = perturbed_df[feature_cols]
        preds = model.predict_proba(perturbed_df)[:, 1]
        d = preds - base_preds[i]

        results.append({
            "session_id": sample.iloc[i]["session_id"],
            "base_probability": base_preds[i],
            "mean_delta": d.mean(),
            "std_delta": d.std(),
        })

    return pd.DataFrame(results)


def run_client_parallel_mode(client_config, scale_factor):
    """
    NEW ARCHITECTURE: intelligent reuse, Ray cluster, parallel-ready.

    This uses the actual new pipeline:
    - Checks if retrain is needed (intelligent reuse)
    - Uses Ray cluster for distributed simulation
    - Can run in parallel with other clients
    """
    cfg = scale_client_config(client_config, scale_factor)
    client_id = cfg["client_id"]

    print(f"\n[{client_id}] Starting NEW-STYLE pipeline (intelligent reuse, Ray)...")
    start = time.time()

    # 1. Data extraction
    raw_data = generate_client_data(cfg)

    # 2. Data preparation
    prepared = prepare_data(raw_data, cfg)

    # 3. Train/test split
    train_df, test_df = split_train_test(prepared, cfg)

    # 4. Intelligent retrain check
    needs_retrain, model_uri, eval_metrics, decision_meta = check_retrain_needed(
        test_df, cfg
    )
    log_retrain_decision(
        client_id,
        needs_retrain,
        eval_metrics,
        decision_reason=decision_meta.get("decision_reason"),
        degradation_delta=decision_meta.get("degradation_delta"),
        month=decision_meta.get("month"),
    )

    # 5. Train or reuse
    if needs_retrain:
        print(f"[{client_id}] Training model (intelligent reuse: needed)...")
        model, train_metrics = train_model(train_df, test_df, cfg)
    else:
        print(f"[{client_id}] Reusing existing model (intelligent reuse: not needed)...")
        mlflow.set_tracking_uri(get_mlflow_tracking_uri())
        model = mlflow.lightgbm.load_model(model_uri)
        train_metrics = eval_metrics

    # 6. Simulation WITH Ray cluster
    print(f"[{client_id}] Running simulation (Ray cluster)...")
    sim_results = run_monte_carlo_simulation(model, prepared, cfg, n_perturbations=100)

    elapsed = time.time() - start
    print(f"[{client_id}] NEW-STYLE complete in {elapsed:.1f}s")

    return {
        "client_id": client_id,
        "retrained": needs_retrain,
        "time_seconds": elapsed,
        "rows": len(prepared),
        "sim_rows": len(sim_results),
    }


def run_sequential_mode(clients, scale_factor):
    """Run all clients sequentially (old architecture simulation)."""
    print(f"\n{'='*70}")
    print("OLD ARCHITECTURE: Sequential, Always-Retrain, No Ray")
    print(f"{'='*70}")

    start = time.time()
    results = []

    for cfg in clients:
        result = run_client_sequential_mode(cfg, scale_factor)
        results.append(result)

    total_time = time.time() - start

    # Calculate metrics
    num_clients = len(results)
    avg_time = sum(r["time_seconds"] for r in results) / num_clients
    total_rows = sum(r["rows"] for r in results)
    total_sim = sum(r["sim_rows"] for r in results)
    retrained = sum(1 for r in results if r["retrained"])
    throughput = num_clients / (total_time / 60) if total_time > 0 else 0  # clients per minute

    print(f"\n{'='*70}")
    print(f"Total wall-clock time: {total_time:.1f}s")
    print(f"Clients completed: {num_clients}/{num_clients}")
    print(f"Throughput: {throughput:.1f} clients/minute")
    print(f"Total data processed: {total_rows:,} rows")
    print(f"Total simulation rows: {total_sim:,}")
    print(f"Models retrained: {retrained} | Models reused: 0")
    print(f"{'='*70}\n")

    return {
        "total_time": total_time,
        "avg_time": avg_time,
        "total_rows": total_rows,
        "total_sim": total_sim,
        "retrained": retrained,
        "reused": 0,
        "num_succeeded": num_clients,
        "num_failed": 0,
        "throughput": throughput,
        "results": results,
    }


def run_parallel_mode(clients, scale_factor, max_workers=5):
    """Run clients in parallel (new architecture)."""
    print(f"\n{'='*70}")
    print(f"NEW ARCHITECTURE: Parallel ({max_workers} workers), Intelligent-Reuse, Ray")
    print(f"{'='*70}")

    # Initialize Ray connection once before parallel execution to avoid race conditions
    import ray
    ray_addr = os.getenv("RAY_ADDRESS", "ray://localhost:10001")
    try:
        ray.init(address=ray_addr, ignore_reinit_error=True)

        # Get cluster resources
        cluster_resources = ray.cluster_resources()
        num_cpus = int(cluster_resources.get("CPU", 0))
        num_nodes = len(ray.nodes())

        print(f"\n✓ Connected to Ray cluster at {ray_addr}")
        print(f"  Ray nodes: {num_nodes} (1 head + {num_nodes-1} workers)")
        print(f"  Total CPUs: {num_cpus}\n")
    except Exception as e:
        print(f"\n✗ WARNING: Could not connect to Ray cluster: {e}")
        print("  Clients will attempt individual connections...\n")

    start = time.time()
    results = []
    failed = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_client_parallel_mode, cfg, scale_factor): cfg
            for cfg in clients
        }

        for future in as_completed(futures):
            cfg = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"\n✗ [{cfg['client_id']}] FAILED: {str(e)[:100]}\n")
                failed.append(cfg['client_id'])

    total_time = time.time() - start

    # Calculate metrics
    num_succeeded = len(results)
    num_failed = len(failed)
    avg_time = sum(r["time_seconds"] for r in results) / num_succeeded if num_succeeded else 0
    total_rows = sum(r["rows"] for r in results)
    total_sim = sum(r["sim_rows"] for r in results)
    retrained = sum(1 for r in results if r["retrained"])
    reused = num_succeeded - retrained
    throughput = num_succeeded / (total_time / 60) if total_time > 0 else 0  # clients per minute

    print(f"\n{'='*70}")
    print(f"Total wall-clock time: {total_time:.1f}s")
    print(f"Clients completed: {num_succeeded}/{len(clients)}" + (f" ({num_failed} failed)" if num_failed > 0 else ""))
    print(f"Throughput: {throughput:.1f} clients/minute")
    print(f"Total data processed: {total_rows:,} rows")
    print(f"Total simulation rows: {total_sim:,}")
    print(f"Models retrained: {retrained} | Models reused: {reused}")
    if failed:
        print(f"Failed clients: {', '.join(failed)}")
    print(f"{'='*70}\n")

    return {
        "total_time": total_time,
        "avg_time": avg_time,
        "total_rows": total_rows,
        "total_sim": total_sim,
        "retrained": retrained,
        "reused": reused,
        "num_succeeded": num_succeeded,
        "num_failed": num_failed,
        "throughput": throughput,
        "results": results,
    }


def print_comparison(old, new, scale_factor):
    """Print side-by-side comparison."""
    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY")
    print(f"{'='*70}\n")

    # Calculate speedup based on total wall-clock time
    speedup = old["total_time"] / new["total_time"] if new["total_time"] > 0 else 0
    time_saved = old["total_time"] - new["total_time"]

    # Calculate throughput gain
    throughput_gain = new["throughput"] / old["throughput"] if old["throughput"] > 0 else 0

    # Calculate reuse rate
    new_total = new["retrained"] + new["reused"]
    reuse_pct = (new["reused"] / new_total * 100) if new_total > 0 else 0

    # Check if comparison is fair
    old_clients = old["num_succeeded"]
    new_clients = new["num_succeeded"]
    fair_comparison = old_clients == new_clients

    print(f"  {'Metric':<40} {'Old':>12} {'New':>12}")
    print(f"  {'-'*66}")
    print(f"  {'Total wall-clock time (s)':<40} {old['total_time']:>12.1f} {new['total_time']:>12.1f}")
    print(f"  {'Clients completed':<40} {old_clients:>12} {new_clients:>12}" + ("" if fair_comparison else " ⚠"))
    print(f"  {'Throughput (clients/min)':<40} {old['throughput']:>12.1f} {new['throughput']:>12.1f}")
    print(f"  {'Total rows processed':<40} {old['total_rows']:>12,} {new['total_rows']:>12,}")
    print(f"  {'Models retrained':<40} {old['retrained']:>12} {new['retrained']:>12}")
    print(f"  {'Models reused':<40} {old['reused']:>12} {new['reused']:>12}")
    print(f"  {'Reuse rate':<40} {'0%':>12} {reuse_pct:>11.1f}%")
    print(f"  {'-'*66}")
    print(f"  {'WALL-CLOCK SPEEDUP':<40} {'—':>12} {speedup:>11.1f}x")
    print(f"  {'THROUGHPUT GAIN':<40} {'—':>12} {throughput_gain:>11.1f}x")
    print(f"  {'TIME SAVED (s)':<40} {'—':>12} {time_saved:>12.1f}")
    print()

    if not fair_comparison:
        print(f"  ⚠ WARNING: Comparison may be unfair - some clients failed in new architecture")
        print()

    print("  Key architectural improvements:")
    print(f"    ✓ Parallel execution ({throughput_gain:.1f}x throughput)")
    print(f"    ✓ Intelligent model reuse ({reuse_pct:.0f}% skip training)")
    print(f"    ✓ Ray cluster for distributed simulation")
    print(f"    ✓ Efficient data processing with DuckDB")
    print()

    print(f"  Scale factor: {scale_factor:.2f} ({1/scale_factor:.0f}x smaller data)")
    print(f"  Full scale test: python baseline/compare.py --scale 1.0 --max-clients 0")
    print()


def main():
    parser = argparse.ArgumentParser(description="Compare old vs new pipeline architectures")
    parser.add_argument(
        "--scale",
        type=float,
        default=DEFAULT_SCALE,
        help=f"Data scale factor (default: {DEFAULT_SCALE}, range: 0.01-1.0). "
             f"Lower = faster demo, higher = closer to production."
    )
    parser.add_argument(
        "--max-clients",
        type=int,
        default=DEFAULT_MAX_CLIENTS,
        help=f"Max clients to test (default: {DEFAULT_MAX_CLIENTS}, 0 = all clients)"
    )
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=5,
        help="Number of parallel workers for new architecture (default: 5)"
    )
    args = parser.parse_args()

    # Load client configs
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "configs", "clients.yaml")
    with open(cfg_path) as f:
        all_clients = yaml.safe_load(f)["clients"]

    # Limit clients for demo
    clients = all_clients[:args.max_clients] if args.max_clients > 0 else all_clients

    print(f"\n{'='*70}")
    print(f"PIPELINE COMPARISON: Old (Dataiku) vs New (AWS)")
    print(f"{'='*70}")
    print(f"  Testing {len(clients)} clients")
    print(f"  Data scale factor: {args.scale:.2f} ({1/args.scale:.1f}x faster)")
    print(f"  Parallel workers: {args.parallel_workers}")
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")

    # Run comparisons
    old = run_sequential_mode(clients, args.scale)
    new = run_parallel_mode(clients, args.scale, max_workers=args.parallel_workers)

    # Print comparison
    print_comparison(old, new, args.scale)


if __name__ == "__main__":
    main()