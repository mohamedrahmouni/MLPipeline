"""
Monte Carlo what-if simulation distributed across a Ray cluster.

Approach:
  1. Sample up to 2M rows from the prepared dataset.
  2. Convert categorical columns to category dtype (LightGBM native handling).
  3. Connect to the Ray cluster via Ray Client (ray:// protocol).
  4. Broadcast the model to every worker once with ray.put().
  5. Chunk the sample and dispatch each chunk as a ray.remote() task.
     Each task:
       a. Predicts base probabilities for all rows at once (vectorised).
       b. Per row: generates N random perturbations, predicts, computes deltas.
       c. Aggregates deltas into summary statistics per row.
  6. Gather results from all tasks into a single pandas DataFrame.
"""
import os
import time
import numpy as np
import pandas as pd
import ray
from config import EXCLUDE_COLS, CATEGORICAL_COLS, BATCH_SIZE


def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare features for prediction, converting categorical columns to category dtype."""
    prepared = df.copy()
    cat_cols = [c for c in CATEGORICAL_COLS if c in prepared.columns]
    for col in cat_cols:
        prepared[col] = prepared[col].astype("category")
    return prepared


@ray.remote
def _perturb_chunk(chunk: pd.DataFrame, model_ref, n_perturbations: int) -> pd.DataFrame:
    """Ray remote task: run perturbation simulation on one chunk of rows.

    The model is fetched from the object store (broadcast via ray.put())
    so it is transferred once per worker, not once per task.

    For each row:
      - Add Gaussian noise (std=0.1) to numeric features, N times.
      - Keep categorical features constant (no perturbation).
      - Predict P(target=1) on each perturbed version.
      - Compute delta = perturbed_pred - base_pred.
      - Summarise deltas (mean, std, min, max, directional rates).
    """
    # Ray auto-dereferences ObjectRefs passed as arguments to remote functions
    model = model_ref
    feature_cols = [c for c in chunk.columns if c not in EXCLUDE_COLS]

    # Separate numeric and categorical features
    cat_cols = [c for c in CATEGORICAL_COLS if c in feature_cols]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    # Vectorised base prediction for the whole chunk at once
    base_features = chunk[feature_cols].copy()
    base_preds = model.predict_proba(base_features)[:, 1]

    results = []
    for i in range(len(chunk)):
        # Extract numeric features and categorical features separately
        num_values = chunk.iloc[i][num_cols].astype(float).values
        cat_values = chunk.iloc[i][cat_cols]

        # Perturb only numeric features
        noise = np.random.randn(n_perturbations, len(num_cols)) * 0.1
        perturbed_num = num_values + noise

        # Create perturbed dataframe with numeric + categorical
        perturbed_df = pd.DataFrame(perturbed_num, columns=num_cols)
        for col in cat_cols:
            perturbed_df[col] = cat_values[col]

        # Convert categorical columns to category dtype for LightGBM
        for col in cat_cols:
            perturbed_df[col] = perturbed_df[col].astype("category")

        # Reorder columns to match original
        perturbed_df = perturbed_df[feature_cols]

        preds = model.predict_proba(perturbed_df)[:, 1]
        d = preds - base_preds[i]

        results.append({
            "session_id": chunk.iloc[i]["session_id"],
            "base_probability": base_preds[i],
            "mean_delta": d.mean(),
            "std_delta": d.std(),
            "min_delta": d.min(),
            "max_delta": d.max(),
            "positive_impact_rate": (d > 0).mean(),
            "negative_impact_rate": (d < 0).mean(),
            "mean_abs_delta": np.abs(d).mean(),
        })

    return pd.DataFrame(results)


def run_monte_carlo_simulation(model, data_df, client_config, n_perturbations=300):
    """Run distributed Monte Carlo simulation via Ray remote tasks.

    Args:
        model: Fitted LightGBM model.
        data_df: Prepared DataFrame with categorical columns.
        client_config: Client configuration dict.
        n_perturbations: Number of perturbations per row.
    """
    client_id = client_config["client_id"]

    # 1. Sample (production: up to 2M rows; demo: capped at 1000 for speed).
    sample = data_df.sample(n=min(len(data_df), 2_000_000), random_state=42)
    sim_sample = _prepare_features(sample.sample(n=min(1000, len(sample)), random_state=42))

    sim_time = min(12 + (len(sample) / 2_000_000) * 8, 20)
    print(f"[{client_id}] Monte Carlo: {len(sim_sample):,} rows x {n_perturbations} perturbations ({sim_time:.0f}s simulated)")
    time.sleep(sim_time / 60)

    # 2. Connect to Ray cluster via Ray Client (if not already connected).
    if not ray.is_initialized():
        ray_addr = os.getenv("RAY_ADDRESS", "ray://localhost:10001")
        ray.init(address=ray_addr, ignore_reinit_error=True)

    # 3. Broadcast model to worker object stores (transferred once per worker).
    model_ref = ray.put(model)

    # 4. Split sample into chunks and dispatch as parallel remote tasks.
    chunks = [sim_sample.iloc[i:i + BATCH_SIZE] for i in range(0, len(sim_sample), BATCH_SIZE)]
    futures = [
        _perturb_chunk.options(name=f"{client_id}_sim_chunk_{i}").remote(chunk, model_ref, n_perturbations)
        for i, chunk in enumerate(chunks)
    ]

    # 5. Gather results from all workers.
    result = pd.concat(ray.get(futures), ignore_index=True)
    print(f"[{client_id}] Ray simulation complete: {result.shape}")
    return result
