"""
Real-world predictive maintenance dataset loader.

Downloads and adapts public datasets to the project's
static (S) + dynamic (D) table format.

Available datasets:
  - metropt3 : Porto Metro train compressor (1.5M rows, 1 machine)
  - ai4i2020 : AI4I 2020 predictive maintenance (10K rows, 10K machines)
  - scania   : SCANIA Component X (1.1M rows, 23K vehicles) [manual download]
"""

import os
import sys
import urllib.request
import zipfile
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "real_datasets")

# Mapping of dataset name to local CSV path
DATASET_PATHS = {
    "ai4i2020": os.path.join(DATA_DIR, "ai4i2020", "ai4i2020.csv"),
    "metropt3": os.path.join(DATA_DIR, "metropt3", "MetroPT3(AirCompressor).csv"),
}


def _local_path(name):
    p = DATASET_PATHS.get(name)
    if p and os.path.exists(p):
        return p
    return None



def download_metropt3(force=False):
    dest = os.path.join(DATA_DIR, "metropt3")
    csv_path = os.path.join(dest, "MetroPT3(AirCompressor).csv")
    if os.path.exists(csv_path) and not force:
        return csv_path
    os.makedirs(dest, exist_ok=True)
    url = "https://cdn.uci-ics-mlr-prod.aws.uci.edu/791/metropt%2B3%2Bdataset.zip"
    zip_path = os.path.join(dest, "metropt3.zip")
    print("Downloading MetroPT-3 dataset (208 MB)...")
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)
    return csv_path


def download_ai4i2020(force=False):
    dest = os.path.join(DATA_DIR, "ai4i2020")
    csv_path = os.path.join(dest, "ai4i2020.csv")
    if os.path.exists(csv_path) and not force:
        return csv_path
    os.makedirs(dest, exist_ok=True)
    url = "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip"
    zip_path = os.path.join(dest, "ai4i2020.zip")
    print("Downloading AI4I 2020 dataset...")
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)
    for f in os.listdir(dest):
        if f.endswith(".csv"):
            os.rename(os.path.join(dest, f), csv_path)
            break
    return csv_path


def load_metropt3(static_filename="static_data.csv", dynamic_filename="dynamic_data.csv",
                  labels_filename="labels.csv", output_dir="."):
    csv_path = download_metropt3()
    print(f"Loading MetroPT-3 from {csv_path}")
    raw = pd.read_csv(csv_path)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])

    # Split into virtual machines by time windows (each week = 1 machine)
    raw["week"] = raw["timestamp"].dt.isocalendar().week.astype(int)
    raw["year"] = raw["timestamp"].dt.isocalendar().year.astype(int)
    raw["machine_id"] = raw["year"].astype(str) + "_W" + raw["week"].astype(str).str.zfill(2)

    machine_ids = raw["machine_id"].unique()
    print(f"  Created {len(machine_ids)} virtual machines by weekly splits")

    # Build static table: per-machine config (mean sensor values)
    static_rows = []
    for mid in machine_ids:
        mdata = raw[raw["machine_id"] == mid].iloc[0]
        static_rows.append({
            "TERMINAL_CODE": mid,
            "TERMINAL_TYPE_ID": 1,
            "PLATFORM": "MetroPT3",
            "PLATFORM_VER": "1.0",
            "IP_ADDRESS_OS": f"10.0.{hash(mid) % 256}.{hash(mid + 'x') % 256}",
            "MAC_ADDRESS_OS": ":".join(f"{(hash(mid + str(i)) % 256):02X}" for i in range(6)),
            "ASA_VERSION": "17.0.0.0",
            "DEFAULT_COLLATION": "1252LATIN1",
            "DB_PAGE_SIZE": 4096,
            "ODBC_FILE_SIZE_B": 100000,
            "CLIENT_EXE_SIZE_B": 12000000,
            "HOME_TOTAL_MB": 5000.0,
            "HOME1_TOTAL_MB": 2000.0,
            "EXECUTION_TIME": mdata["timestamp"],
        })
    static = pd.DataFrame(static_rows)
    static.to_csv(os.path.join(output_dir, static_filename), index=False)
    print(f"  Static: {len(static)} machines")

    # Build dynamic table: all sensor readings
    sensor_cols = ["TP2", "TP3", "H1", "DV_pressure", "Reservoirs",
                   "Oil_temperature", "Motor_current", "COMP", "DV_eletric",
                   "Towers", "MPG", "LPS", "Pressure_switch", "Oil_level",
                   "Caudal_impulses"]
    dynamic_rows = []
    for _, row in raw.iterrows():
        dynamic_rows.append({
            "TERMINAL_CODE": row["machine_id"],
            "LAST_TKT_NUMBER": str(int(row.get("Unnamed: 0", 0) % 1_000_000)),
            "LAST_TXN_TIME": row["timestamp"],
            "LAST_DOWNLOAD_TIME": row["timestamp"],
            "LAST_UPLOAD_TIME": row["timestamp"],
            "TOTAL_TKT_ISSUED_YESTERDAY": int(abs(row["TP2"]) * 100) % 500,
            "TOTAL_TKT_ISSUED_TODAY": int(abs(row["TP3"]) * 100) % 300,
            "NETWORK_LINK": 1 if row.get("COMP", 1) > 0.5 else 0,
            "HOME_USED_MB": round(abs(row["Oil_temperature"]) * 10, 2),
            "HOME1_USED_MB": round(abs(row["Motor_current"]) * 50, 2),
            "DB_FILE_SIZE_MB": round(abs(row["Reservoirs"]) * 10, 2),
            "DB_FILE_SIZE_IN_PAGES": int(abs(row["TP2"]) * 10000),
            "DB_FREE_PAGES": int(abs(row["TP3"]) * 5000),
            "DB_MRR_LOG_DIR_SIZE_MB": round(abs(row.get("MPG", 0)) * 5, 2),
            "DB_CRASH_LOG_DIR_SIZE_MB": round(abs(row.get("LPS", 0)) * 2, 2),
            "OS_UP_SINCE": row["timestamp"] - timedelta(days=30),
            "DB_UP_SINCE": row["timestamp"] - timedelta(days=7),
            "CLIENT_LOG_EXIST_SINCE": row["timestamp"] - timedelta(days=14),
            "TC_MAKE": "Metro",
            "PASSWORD_SYBUTS": "****",
            "PASSWORD_ROOT": "****",
            "EXECUTION_TIME": row["timestamp"],
        })
    dynamic = pd.DataFrame(dynamic_rows)
    dynamic.to_csv(os.path.join(output_dir, dynamic_filename), index=False)
    print(f"  Dynamic: {len(dynamic)} records")

    # Build synthetic labels: simulate failures at high oil temperature
    oil_temp = raw["Oil_temperature"].values
    threshold = np.percentile(oil_temp, 95)
    labels_rows = []
    for _, row in raw.iterrows():
        is_fault = 1 if row["Oil_temperature"] > threshold else 0
        labels_rows.append({
            "TERMINAL_CODE": row["machine_id"],
            "EXECUTION_TIME": row["timestamp"],
            "fault_label": is_fault,
            "rul_days": max(0, 30 - (row["Oil_temperature"] - oil_temp.min()) /
                            (oil_temp.max() - oil_temp.min()) * 30) if is_fault else -1,
            "failure_mode": "overheat" if is_fault else "none",
        })
    labels = pd.DataFrame(labels_rows)
    labels.to_csv(os.path.join(output_dir, labels_filename), index=False)
    print(f"  Labels: {len(labels)} records ({labels['fault_label'].mean()*100:.1f}% faults)")

    return static, dynamic, labels


def load_ai4i2020(static_filename="static_data.csv", dynamic_filename="dynamic_data.csv",
                  labels_filename="labels.csv", output_dir="."):
    local = _local_path("ai4i2020")
    csv_path = local if local else download_ai4i2020()
    print(f"Loading AI4I 2020 from {csv_path}")
    raw = pd.read_csv(csv_path)

    # The AI4I 2020 dataset already has multiple machines (UDI 1-10000)
    raw = raw.rename(columns={"UDI": "TERMINAL_CODE"})
    raw["TERMINAL_CODE"] = raw["TERMINAL_CODE"].apply(lambda x: f"TC{x:05d}")

    # Static: product type, quality variant
    static = raw[["TERMINAL_CODE", "Type"]].drop_duplicates()
    static["TERMINAL_TYPE_ID"] = static["Type"].astype("category").cat.codes + 1
    static["PLATFORM"] = "AI4I"
    static["PLATFORM_VER"] = "2020"
    static["ASA_VERSION"] = "17.0.0.0"
    static["DEFAULT_COLLATION"] = "1252LATIN1"
    static["DB_PAGE_SIZE"] = 4096
    static["ODBC_FILE_SIZE_B"] = 100000
    static["CLIENT_EXE_SIZE_B"] = 12000000
    static["HOME_TOTAL_MB"] = 5000.0
    static["HOME1_TOTAL_MB"] = 2000.0
    static["EXECUTION_TIME"] = pd.Timestamp("2020-01-01")
    static = static.drop(columns=["Type"])
    static.to_csv(os.path.join(output_dir, static_filename), index=False)
    print(f"  Static: {len(static)} machines")

    # Generate time series: expand each machine to 24 time steps with noise
    np.random.seed(42)
    dyn_rows = []
    ts_start = pd.Timestamp("2020-06-01 06:00:00")
    for _, row in raw.iterrows():
        tc = row["TERMINAL_CODE"]
        air_temp = row["Air temperature [K]"]
        proc_temp = row["Process temperature [K]"]
        rot_speed = row["Rotational speed [rpm]"]
        torque = row["Torque [Nm]"]
        tool_wear = row["Tool wear [min]"]
        is_fail = row["Machine failure"]
        for step in range(24):
            noise = np.random.randn(5) * 0.02
            home_used = air_temp + noise[0] * 10
            home1_used = proc_temp + noise[1] * 5
            db_size = rot_speed / 100 + noise[2] * 2
            mrr_log = torque + noise[3] * 1
            crash_log = tool_wear + noise[4] * 0.5
            if is_fail and step > 20:
                home_used += 5.0 * (step - 20)
                home1_used += 3.0 * (step - 20)
            dyn_rows.append({
                "TERMINAL_CODE": tc,
                "EXECUTION_TIME": ts_start + pd.Timedelta(hours=step),
                "LAST_TKT_NUMBER": str(int(abs(home_used * 1000)) % 1_000_000),
                "LAST_TXN_TIME": ts_start + pd.Timedelta(hours=step),
                "LAST_DOWNLOAD_TIME": ts_start + pd.Timedelta(hours=step),
                "LAST_UPLOAD_TIME": ts_start + pd.Timedelta(hours=step),
                "TOTAL_TKT_ISSUED_YESTERDAY": int(abs(rot_speed)) % 500,
                "TOTAL_TKT_ISSUED_TODAY": int(abs(torque * 10)) % 300,
                "NETWORK_LINK": 1 if abs(torque) < 60 else 0,
                "HOME_USED_MB": round(home_used, 2),
                "HOME1_USED_MB": round(home1_used, 2),
                "DB_FILE_SIZE_MB": round(db_size, 2),
                "DB_FILE_SIZE_IN_PAGES": int(db_size * 100),
                "DB_FREE_PAGES": int(db_size * 60),
                "DB_MRR_LOG_DIR_SIZE_MB": round(mrr_log, 2),
                "DB_CRASH_LOG_DIR_SIZE_MB": round(crash_log, 2),
                "OS_UP_SINCE": ts_start - pd.Timedelta(days=30),
                "DB_UP_SINCE": ts_start - pd.Timedelta(days=7),
                "CLIENT_LOG_EXIST_SINCE": ts_start - pd.Timedelta(days=14),
                "TC_MAKE": "AI4I",
                "PASSWORD_SYBUTS": "****",
                "PASSWORD_ROOT": "****",
            })
    dynamic = pd.DataFrame(dyn_rows)
    dynamic.to_csv(os.path.join(output_dir, dynamic_filename), index=False)
    print(f"  Dynamic: {len(dynamic)} records ({len(dynamic)//24} machines x 24 time steps)")

    # Labels: expand to match time series
    fail_map = dict(zip(raw["TERMINAL_CODE"], raw["Machine failure"]))
    fmode_map = {}
    for _, r in raw.iterrows():
        tc = r["TERMINAL_CODE"]
        fmode = "none"
        for fm in ["TWF", "HDF", "PWF", "OSF", "RNF"]:
            if fm in raw.columns and r[fm] == 1:
                fmode = fm
                break
        fmode_map[tc] = fmode

    labels = dynamic[["TERMINAL_CODE", "EXECUTION_TIME"]].copy()
    labels["fault_label"] = labels["TERMINAL_CODE"].map(fail_map).values
    labels["failure_mode"] = labels["TERMINAL_CODE"].map(fmode_map).values
    labels["rul_days"] = -1
    for tc in raw["TERMINAL_CODE"]:
        if fail_map.get(tc, 0) == 1:
            mask = labels["TERMINAL_CODE"] == tc
            n = mask.sum()
            labels.loc[mask, "rul_days"] = np.linspace(15, 0.5, n)
    labels.to_csv(os.path.join(output_dir, labels_filename), index=False)
    print(f"  Labels: {len(labels)} records ({labels['fault_label'].mean()*100:.1f}% faults)")

    return static, dynamic, labels


def load_dataset(name="ai4i2020", output_dir="real_data_output"):
    os.makedirs(output_dir, exist_ok=True)

    local = _local_path(name)
    if local:
        print(f"Found local file: {local}")
    else:
        print(f"No local file found, will download {name}...")

    if name == "metropt3":
        return load_metropt3(output_dir=output_dir)
    elif name == "ai4i2020":
        return load_ai4i2020(output_dir=output_dir)
    else:
        raise ValueError(f"Unknown dataset: {name}. Choose: metropt3, ai4i2020")


if __name__ == "__main__":
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else "ai4i2020"
    output = sys.argv[2] if len(sys.argv) > 2 else "real_data_output"
    load_dataset(name, output)
