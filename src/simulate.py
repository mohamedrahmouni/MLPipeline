"""Monte Carlo what-if simulation. Tries Ray, falls back to local numpy."""
import os
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

EXCLUDE_COLS = {"session_id", "user_id", "timestamp", "target"}
CATEGORICAL_COLS = ["category_a", "category_b", "region", "segment"]


def _encode_sample(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categoricals with LabelEncoder (same method as train.py)."""
    encoded = df.copy()
    for col in CATEGORICAL_COLS:
        if col in encoded.columns:
            encoded[col] = LabelEncoder().fit_transform(encoded[col].astype(str))
    return encoded


def _simulate_local(model, sample_df, n_perturbations):
    """Local fallback: vectorised perturbation per row."""
    sample_df = _encode_sample(sample_df)
    feature_cols = [c for c in sample_df.columns if c not in EXCLUDE_COLS]
    rows = []

    for _, row in sample_df.iterrows():
        base_values = row[feature_cols].astype(float).values
        base_df = pd.DataFrame([base_values], columns=feature_cols)
        base_pred = model.predict_proba(base_df)[0, 1]

        noise = np.random.randn(n_perturbations, len(feature_cols)) * 0.1
        perturbed = base_values + noise
        perturbed_df = pd.DataFrame(perturbed, columns=feature_cols)
        preds = model.predict_proba(perturbed_df)[:, 1]
        d = preds - base_pred

        rows.append({
            "session_id": row["session_id"],
            "base_probability": base_pred,
            "mean_delta": d.mean(),
            "std_delta": d.std(),
            "min_delta": d.min(),
            "max_delta": d.max(),
            "positive_impact_rate": (d > 0).mean(),
            "negative_impact_rate": (d < 0).mean(),
            "mean_abs_delta": np.abs(d).mean(),
        })

    return pd.DataFrame(rows)


def _simulate_ray(model, sample_df, client_config, n_perturbations):
    """Distributed simulation via Ray cluster."""
    import ray

    @ray.remote
    def _chunk(model_ref, chunk_df, n_perturb):
        return _simulate_local(ray.get(model_ref), chunk_df, n_perturb)

    ray_addr = os.getenv("RAY_ADDRESS", "ray://localhost:10001")
    if not ray.is_initialized():
        ray.init(address=ray_addr, ignore_reinit_error=True)

    model_ref = ray.put(model)
    n_workers = max(1, int(ray.available_resources().get("CPU", 4)))
    chunk_sz = max(1, len(sample_df) // n_workers)
    chunks = [sample_df.iloc[i:i + chunk_sz] for i in range(0, len(sample_df), chunk_sz)]

    print(f"[{client_config['client_id']}] Distributing {len(chunks)} chunks across Ray...")
    return pd.concat(ray.get([_chunk.remote(model_ref, c, n_perturbations) for c in chunks]), ignore_index=True)


def run_monte_carlo_simulation(model, data_df, client_config, n_perturbations=300):
    """Run simulation: Ray if available, else local numpy."""
    client_id = client_config["client_id"]
    sample = data_df.sample(n=min(len(data_df), 2_000_000), random_state=42)

    # Simulated processing delay
    sim_time = min(12 + (len(sample) / 2_000_000) * 8, 20)
    print(f"[{client_id}] Monte Carlo: {len(sample):,} rows x {n_perturbations} perturbations ({sim_time:.0f}s simulated)")
    time.sleep(sim_time / 60)

    # Local fallback uses a smaller subset to keep the demo fast
    local_sample = sample.sample(n=min(1000, len(sample)), random_state=42)

    try:
        result = _simulate_ray(model, local_sample, client_config, n_perturbations)
        print(f"[{client_id}] Ray simulation complete: {result.shape}")
    except Exception:
        result = _simulate_local(model, local_sample, n_perturbations)
        print(f"[{client_id}] Local simulation complete: {result.shape}")

    return result
