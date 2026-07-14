import numpy as np
import pandas as pd
from typing import List, Tuple, Optional


STATIC_FEATURES = [
    "TERMINAL_TYPE_ID",
    "ODBC_FILE_SIZE_B",
    "DB_PAGE_SIZE",
    "CLIENT_EXE_SIZE_B",
    "HOME_TOTAL_MB",
    "HOME1_TOTAL_MB",
    "PLATFORM_enc",
    "ASA_VERSION_enc",
]

DYNAMIC_FEATURES = [
    "LAST_TKT_NUMBER_int",
    "TOTAL_TKT_ISSUED_YESTERDAY",
    "TOTAL_TKT_ISSUED_TODAY",
    "NETWORK_LINK",
    "HOME_USED_MB",
    "HOME1_USED_MB",
    "DB_FILE_SIZE_MB",
    "DB_FILE_SIZE_IN_PAGES",
    "DB_FREE_PAGES",
    "DB_MRR_LOG_DIR_SIZE_MB",
    "DB_CRASH_LOG_DIR_SIZE_MB",
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "home_used_pct",
    "home1_used_pct",
    "db_free_pct",
    "os_uptime_hours",
    "db_uptime_hours",
    "log_exist_hours",
    "ticket_volume_ratio",
    "home_used_roll_mean_12",
    "home_used_roll_std_12",
    "db_free_roll_mean_12",
    "db_free_roll_std_12",
    "mrr_log_roll_mean_12",
    "crash_log_roll_mean_12",
    "home_used_diff",
    "db_free_diff",
    "netlink_stable_hours",
    "home_used_zscore",
    "db_free_zscore",
]


def engineer_static_features(static_df: pd.DataFrame) -> pd.DataFrame:
    df = static_df.copy()
    platforms = df["PLATFORM"].astype("category").cat.codes
    asa_vers = df["ASA_VERSION"].astype("category").cat.codes
    df["PLATFORM_enc"] = platforms
    df["ASA_VERSION_enc"] = asa_vers
    return df


def engineer_dynamic_features(
    dynamic_df: pd.DataFrame,
    static_df: pd.DataFrame,
    window_size: int = 12,
) -> pd.DataFrame:
    df = dynamic_df.copy()
    static_map = static_df.set_index("TERMINAL_CODE")[
        ["HOME_TOTAL_MB", "HOME1_TOTAL_MB", "PLATFORM"]
    ].to_dict("index")

    df["EXECUTION_TIME"] = pd.to_datetime(df["EXECUTION_TIME"])
    df = df.sort_values(["TERMINAL_CODE", "EXECUTION_TIME"])

    df["LAST_TKT_NUMBER_int"] = (
        pd.to_numeric(df["LAST_TKT_NUMBER"], errors="coerce").fillna(0).astype(int)
    )

    ts = df["EXECUTION_TIME"]
    df["hour_sin"] = np.sin(2 * np.pi * ts.dt.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * ts.dt.hour / 24)
    df["day_of_week_sin"] = np.sin(2 * np.pi * ts.dt.dayofweek / 7)
    df["day_of_week_cos"] = np.cos(2 * np.pi * ts.dt.dayofweek / 7)

    df["home_used_pct"] = 0.0
    df["home1_used_pct"] = 0.0
    for tc in df["TERMINAL_CODE"].unique():
        if tc in static_map:
            ht = static_map[tc]["HOME_TOTAL_MB"]
            h1t = static_map[tc]["HOME1_TOTAL_MB"]
            mask = df["TERMINAL_CODE"] == tc
            df.loc[mask, "home_used_pct"] = (
                df.loc[mask, "HOME_USED_MB"] / ht * 100
            )
            df.loc[mask, "home1_used_pct"] = (
                df.loc[mask, "HOME1_USED_MB"] / h1t * 100
            )

    df["db_free_pct"] = np.where(
        df["DB_FILE_SIZE_IN_PAGES"] > 0,
        df["DB_FREE_PAGES"] / df["DB_FILE_SIZE_IN_PAGES"] * 100,
        0,
    )

    now_ref = df["EXECUTION_TIME"].max()
    df["os_uptime_hours"] = (
        now_ref - pd.to_datetime(df["OS_UP_SINCE"])
    ).dt.total_seconds() / 3600
    df["db_uptime_hours"] = (
        now_ref - pd.to_datetime(df["DB_UP_SINCE"])
    ).dt.total_seconds() / 3600
    df["log_exist_hours"] = (
        now_ref - pd.to_datetime(df["CLIENT_LOG_EXIST_SINCE"])
    ).dt.total_seconds() / 3600

    df["ticket_volume_ratio"] = np.where(
        df["TOTAL_TKT_ISSUED_YESTERDAY"] > 0,
        df["TOTAL_TKT_ISSUED_TODAY"] / df["TOTAL_TKT_ISSUED_YESTERDAY"],
        0,
    )

    agg_cols = {
        "HOME_USED_MB": "home_used",
        "DB_FREE_PAGES": "db_free",
        "DB_MRR_LOG_DIR_SIZE_MB": "mrr_log",
        "DB_CRASH_LOG_DIR_SIZE_MB": "crash_log",
    }

    for col, prefix in agg_cols.items():
        df[f"{prefix}_roll_mean_{window_size}"] = df.groupby("TERMINAL_CODE")[
            col
        ].transform(lambda x: x.rolling(window_size, min_periods=1).mean())
        if prefix in ("home_used", "db_free"):
            df[f"{prefix}_roll_std_{window_size}"] = (
                df.groupby("TERMINAL_CODE")[col].transform(
                    lambda x: x.rolling(window_size, min_periods=1).std()
                )
            )
            df[f"{prefix}_roll_std_{window_size}"] = df[
                f"{prefix}_roll_std_{window_size}"
            ].fillna(0)

    df["home_used_diff"] = df.groupby("TERMINAL_CODE")["HOME_USED_MB"].diff().fillna(0)
    df["db_free_diff"] = df.groupby("TERMINAL_CODE")["DB_FREE_PAGES"].diff().fillna(0)

    def _netlink_stable(series):
        stable = 0
        result = np.zeros(len(series))
        for i, val in enumerate(series):
            if val == 1:
                stable += 1
            else:
                stable = 0
            result[i] = stable
        return result

    df["netlink_stable_hours"] = (
        df.groupby("TERMINAL_CODE")["NETWORK_LINK"]
        .transform(_netlink_stable)
    )

    for col, prefix in [("HOME_USED_MB", "home_used"), ("DB_FREE_PAGES", "db_free")]:
        mean_val = df.groupby("TERMINAL_CODE")[col].transform("mean")
        std_val = df.groupby("TERMINAL_CODE")[col].transform("std").replace(0, 1)
        df[f"{prefix}_zscore"] = (df[col] - mean_val) / std_val

    return df


def create_sequences(
    dynamic_feat_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    static_feat_df: pd.DataFrame,
    sequence_length: int = 48,
    stride: int = 12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sequences = []
    fault_targets = []
    rul_targets = []

    label_map = labels_df.set_index(["TERMINAL_CODE", "EXECUTION_TIME"])[
        ["fault_label", "rul_days"]
    ].to_dict("index")

    static_map = static_feat_df.set_index("TERMINAL_CODE")[
        STATIC_FEATURES
    ].to_dict("index")

    for tc, group in dynamic_feat_df.groupby("TERMINAL_CODE"):
        group = group.sort_values("EXECUTION_TIME")
        group_values = group[DYNAMIC_FEATURES].values
        group_times = group["EXECUTION_TIME"].values
        group_codes = group["TERMINAL_CODE"].values

        static_vec = static_map.get(tc)
        if static_vec is None:
            continue
        static_arr = np.array([static_vec[f] for f in STATIC_FEATURES])

        for i in range(0, len(group_values) - sequence_length + 1, stride):
            seq = group_values[i : i + sequence_length]
            target_time = group_times[i + sequence_length - 1]
            target_code = group_codes[i + sequence_length - 1]

            key = (target_code, pd.Timestamp(target_time))
            if key in label_map:
                label_row = label_map[key]
                sequences.append(
                    np.concatenate([seq, np.tile(static_arr, (sequence_length, 1))], axis=1)
                )
                fault_targets.append(label_row["fault_label"])
                rul_targets.append(label_row["rul_days"])

    return (
        np.array(sequences, dtype=np.float32),
        np.array(fault_targets, dtype=np.float32),
        np.array(rul_targets, dtype=np.float32),
    )


def get_feature_dim() -> int:
    return len(DYNAMIC_FEATURES) + len(STATIC_FEATURES)
