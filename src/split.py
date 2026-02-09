"""Session-aware train/test split — no session spans both sets."""
import time
import numpy as np
import pandas as pd


def split_train_test(df, client_config, test_size=0.3):
    """Split by session_id so no session leaks across train/test."""
    client_id = client_config["client_id"]

    split_time = min(1 + (len(df) / 100_000) * 2, 3)
    print(f"[{client_id}] Splitting train/test... ({split_time:.0f}s simulated)")
    time.sleep(split_time / 60)

    sessions = df["session_id"].unique()
    np.random.seed(42 + hash(client_id) % 1000)
    shuffled = np.random.permutation(sessions)

    test_sessions = set(shuffled[: int(len(sessions) * test_size)])
    mask = df["session_id"].isin(test_sessions)

    train_df, test_df = df[~mask].copy(), df[mask].copy()
    print(f"[{client_id}] Split: train={len(train_df):,} ({len(train_df)/len(df)*100:.0f}%)  "
          f"test={len(test_df):,} ({len(test_df)/len(df)*100:.0f}%)")
    return train_df, test_df
