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
import numpy as np
import pandas as pd
import ray
from config import EXCLUDE_COLS, CATEGORICAL_COLS, BATCH_SIZE


def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare features for prediction, converting categorical columns to category dtype."""
    cat_cols = [c for c in CATEGORICAL_COLS if c in df.columns]
    for col in cat_cols:
        df[col] = df[col].astype("category")
    return df


@ray.remote
def _perturb_chunk(chunk: pd.DataFrame, model_ref, n_perturbations: int) -> pd.DataFrame:
    """Ray remote task: run what-if improvement simulation on one chunk (VECTORIZED).

    The model is fetched from the object store (broadcast via ray.put())
    so it is transferred once per worker, not once per task.

    Simulates improvements (5-20% feature enhancements) across n_perturbations scenarios.
    Returns 19-column analysis for business decision-making:
      - 4 base metrics
      - 6 improvement scenarios (worst to best case)
      - 3 impact distribution metrics
      - 6 business metrics (upside/downside/confidence)

    Fully vectorized: processes all rows and all perturbations in parallel.
    """
    # Ray auto-dereferences ObjectRefs passed as arguments to remote functions
    model = model_ref
    feature_cols = [c for c in chunk.columns if c not in EXCLUDE_COLS]

    # Separate numeric and categorical features
    cat_cols = [c for c in CATEGORICAL_COLS if c in feature_cols]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    # Vectorised base prediction for the whole chunk at once
    base_preds = model.predict_proba(chunk[feature_cols])[:, 1]

    n_rows = len(chunk)

    # VECTORIZED: Generate all improvements for all rows at once
    # Shape: (n_rows, n_features) -> (n_rows, n_perturbations, n_features)
    num_features = chunk[num_cols].astype(float).values  # (n_rows, n_num_cols)

    # Generate improvement factors: (n_rows, n_perturbations, n_num_cols)
    improvement_factors = 1.0 + np.random.uniform(0.05, 0.20, size=(n_rows, n_perturbations, len(num_cols)))

    # Apply improvements: broadcast multiplication
    improved_features = num_features[:, np.newaxis, :] * improvement_factors  # (n_rows, n_perturbations, n_num_cols)

    # Reshape to (n_rows * n_perturbations, n_num_cols) for prediction
    improved_flat = improved_features.reshape(-1, len(num_cols))

    # Create dataframe with categorical features repeated
    improved_df = pd.DataFrame(improved_flat, columns=num_cols)

    # Repeat categorical features for each perturbation
    for col in cat_cols:
        improved_df[col] = np.repeat(chunk[col].values, n_perturbations)
        improved_df[col] = improved_df[col].astype("category")

    improved_df = improved_df[feature_cols]

    # VECTORIZED: Single prediction call for all rows * all perturbations
    all_preds = model.predict_proba(improved_df)[:, 1]  # (n_rows * n_perturbations,)

    # Reshape back to (n_rows, n_perturbations)
    preds_matrix = all_preds.reshape(n_rows, n_perturbations)

    # VECTORIZED: Calculate deltas for all rows
    deltas = preds_matrix - base_preds[:, np.newaxis]  # (n_rows, n_perturbations)

    # VECTORIZED: Calculate all 19 metrics for all rows at once
    results_dict = {
        # Base (4 cols)
        "session_id": chunk["session_id"].values,
        "base_probability": base_preds,
        "expected_improvement": deltas.mean(axis=1),
        "improvement_std": deltas.std(axis=1),

        # Scenarios (6 cols) - vectorized percentiles
        "worst_case": deltas.min(axis=1),
        "p10_improvement": np.percentile(deltas, 10, axis=1),
        "p25_improvement": np.percentile(deltas, 25, axis=1),
        "median_improvement": np.percentile(deltas, 50, axis=1),
        "p75_improvement": np.percentile(deltas, 75, axis=1),
        "best_case": deltas.max(axis=1),

        # Impact distribution (3 cols)
        "positive_impact_pct": (deltas > 0).mean(axis=1) * 100,
        "negative_impact_pct": (deltas < 0).mean(axis=1) * 100,
        "neutral_impact_pct": (deltas == 0).mean(axis=1) * 100,

        # Business metrics (6 cols)
        "upside_potential": np.maximum(0, deltas.max(axis=1)),
        "downside_risk": np.abs(np.minimum(0, deltas.min(axis=1))),
        "expected_value": base_preds + deltas.mean(axis=1),
        "confidence_lower_90": base_preds + np.percentile(deltas, 10, axis=1),
        "confidence_upper_90": base_preds + np.percentile(deltas, 90, axis=1),
        "volatility_score": deltas.std(axis=1) / (base_preds + 1e-6),
    }

    return pd.DataFrame(results_dict)  # 19 columns, vectorized!


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

    print(f"[{client_id}] Monte Carlo: {len(sim_sample):,} rows x {n_perturbations} perturbations")

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

    # Print summary of what-if analysis results
    print(f"[{client_id}] What-if simulation complete: {result.shape[0]:,} sessions × 19 columns")
    print(f"[{client_id}] Improvement insights (avg across sessions):")
    print(f"  Expected improvement:    {result['expected_improvement'].mean():+.4f}")
    print(f"  Upside potential:        {result['upside_potential'].mean():.4f}")
    print(f"  Downside risk:           {result['downside_risk'].mean():.4f}")
    print(f"  Positive impact:         {result['positive_impact_pct'].mean():.1f}%")

    return result
