"""Fake data generation — 31-column schema mimicking production Snowflake extract."""
import time
import numpy as np
import pandas as pd


def generate_client_data(client_config):
    """Generate fake client data with 2% nulls for imputation testing."""
    row_count = client_config["row_count"]
    client_id = client_config["client_id"]

    # Simulate Snowflake query time (scaled down for demo)
    query_time = min(3 + (row_count / 100_000) * 8, 11)
    print(f"[{client_id}] Extracting {row_count:,} rows... ({query_time:.0f}s simulated)")
    time.sleep(query_time / 60)

    np.random.seed(hash(client_id) % (2**32))
    n_sessions = max(100, row_count // 1000)

    data = {
        "session_id": np.random.randint(1, n_sessions + 1, size=row_count),
        "user_id": np.random.randint(1000, 100_000, size=row_count),
        "timestamp": pd.date_range("2024-01-01", periods=row_count, freq="1min"),
        # 15 numeric features
        **{f"feature_{i}": np.random.randn(row_count) * (i % 5 + 1) for i in range(1, 16)},
        # 8 boolean features
        **{col: np.random.randint(0, 2, row_count) for col in [
            "is_active", "is_premium", "has_feature_a", "has_feature_b",
            "is_verified", "opt_in_marketing", "mobile_user", "desktop_user",
        ]},
        # 4 categorical features
        "category_a": np.random.choice(["cat1", "cat2", "cat3", "cat4"], row_count),
        "category_b": np.random.choice(["type_x", "type_y", "type_z"], row_count),
        "region": np.random.choice(["north", "south", "east", "west"], row_count),
        "segment": np.random.choice(["seg_1", "seg_2", "seg_3"], row_count),
        # Target
        "target": np.random.randint(0, 2, row_count),
    }

    df = pd.DataFrame(data)

    # Sprinkle 2% nulls across feature columns
    nullable = [c for c in df.columns if c not in ("session_id", "user_id", "timestamp", "target")]
    mask = np.random.rand(row_count, len(nullable)) < 0.02
    df[nullable] = df[nullable].mask(mask)

    print(f"[{client_id}] Extraction complete: {df.shape}")
    return df
