"""Data preparation using DuckDB (stands in for Snowflake UDTF)."""
import time
import duckdb


def prepare_data(df, client_config):
    """Impute nulls (median / 0 / 'unknown'), cast types, filter dates — all via DuckDB SQL."""
    client_id = client_config["client_id"]

    prep_time = min(1 + (len(df) / 100_000) * 2, 3)
    print(f"[{client_id}] Preprocessing... ({prep_time:.0f}s simulated)")
    time.sleep(prep_time / 60)

    con = duckdb.connect(":memory:")
    con.register("raw_data", df)

    result = con.execute("""
    SELECT
        session_id, user_id, timestamp,

        -- numeric: median imputation
        COALESCE(feature_1,  median(feature_1)  OVER ()) AS feature_1,
        COALESCE(feature_2,  median(feature_2)  OVER ()) AS feature_2,
        COALESCE(feature_3,  median(feature_3)  OVER ()) AS feature_3,
        COALESCE(feature_4,  median(feature_4)  OVER ()) AS feature_4,
        COALESCE(feature_5,  median(feature_5)  OVER ()) AS feature_5,
        COALESCE(feature_6,  median(feature_6)  OVER ()) AS feature_6,
        COALESCE(feature_7,  median(feature_7)  OVER ()) AS feature_7,
        COALESCE(feature_8,  median(feature_8)  OVER ()) AS feature_8,
        COALESCE(feature_9,  median(feature_9)  OVER ()) AS feature_9,
        COALESCE(feature_10, median(feature_10) OVER ()) AS feature_10,
        COALESCE(feature_11, median(feature_11) OVER ()) AS feature_11,
        COALESCE(feature_12, median(feature_12) OVER ()) AS feature_12,
        COALESCE(feature_13, median(feature_13) OVER ()) AS feature_13,
        COALESCE(feature_14, median(feature_14) OVER ()) AS feature_14,
        COALESCE(feature_15, median(feature_15) OVER ()) AS feature_15,

        -- booleans: fill with 0
        COALESCE(is_active, 0)::INTEGER        AS is_active,
        COALESCE(is_premium, 0)::INTEGER       AS is_premium,
        COALESCE(has_feature_a, 0)::INTEGER    AS has_feature_a,
        COALESCE(has_feature_b, 0)::INTEGER    AS has_feature_b,
        COALESCE(is_verified, 0)::INTEGER      AS is_verified,
        COALESCE(opt_in_marketing, 0)::INTEGER AS opt_in_marketing,
        COALESCE(mobile_user, 0)::INTEGER      AS mobile_user,
        COALESCE(desktop_user, 0)::INTEGER     AS desktop_user,

        -- categoricals: fill with 'unknown'
        COALESCE(category_a, 'unknown') AS category_a,
        COALESCE(category_b, 'unknown') AS category_b,
        COALESCE(region, 'unknown')     AS region,
        COALESCE(segment, 'unknown')    AS segment,

        COALESCE(target, 0)::INTEGER AS target
    FROM raw_data
    WHERE timestamp >= '2024-01-01'
    """).fetchdf()

    con.close()
    print(f"[{client_id}] Preprocessing complete: {result.shape}")
    return result
