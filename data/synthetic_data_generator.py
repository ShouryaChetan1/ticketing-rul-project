import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import os
from tqdm import tqdm


TERMINAL_TYPES = {
    1: {"platform": "Windows", "versions": ["10.0.19044", "10.0.22621", "11.0.22000"]},
    2: {"platform": "Linux", "versions": ["4.15.0", "5.4.0", "5.10.0", "6.1.0"]},
    3: {"platform": "Windows", "versions": ["10.0.17763", "10.0.18363"]},
}

TC_MAKES = ["HCL", "Wipro", "Dell", "HP", "Lenovo"]
ASA_VERSIONS = ["16.0.0.0", "17.0.0.0", "17.0.0.1", "17.0.0.2"]
COLLATIONS = ["1252LATIN1", "1256ARABIC", "1252BIN", "UTF8"]

FAILURE_MODES = [
    "disk_exhaustion",
    "db_full",
    "db_corruption",
    "memory_leak",
    "log_growth",
    "network_degradation",
]


def generate_static_data(num_machines: int = 20000) -> pd.DataFrame:
    records = []
    base_date = datetime(2024, 1, 1)

    for i in range(1, num_machines + 1):
        tc = f"TC{i:06d}"
        type_id = random.choices(list(TERMINAL_TYPES.keys()), weights=[0.5, 0.3, 0.2])[0]
        plat_info = TERMINAL_TYPES[type_id]
        platform = plat_info["platform"]
        platform_ver = random.choice(plat_info["versions"])

        ip = f"10.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}"
        mac = ":".join(f"{random.randint(0, 255):02X}" for _ in range(6))

        odbc_size = random.randint(50000, 200000)
        db_page_size = random.choice([2048, 4096, 8192, 16384, 32768])
        asa_ver = random.choice(ASA_VERSIONS)
        collation = random.choice(COLLATIONS)
        client_exe_size = random.randint(8000000, 25000000)
        home_total = round(random.uniform(500.0, 5000.0), 2)
        home1_total = round(random.uniform(200.0, 2000.0), 2)

        exec_time = base_date + timedelta(days=random.randint(0, 30), hours=random.randint(0, 23))

        records.append({
            "TERMINAL_CODE": tc,
            "TERMINAL_TYPE_ID": type_id,
            "PLATFORM": platform,
            "PLATFORM_VER": platform_ver,
            "IP_ADDRESS_OS": ip,
            "MAC_ADDRESS_OS": mac,
            "ODBC_FILE_SIZE_B": odbc_size,
            "DB_FILE_PATH_NAME": f"/data/{tc}/database.db",
            "DB_LOG_FILE_PATH_NAME": f"/data/{tc}/database.log",
            "ASA_VERSION": asa_ver,
            "DEFAULT_COLLATION": collation,
            "DB_PAGE_SIZE": db_page_size,
            "CLIENT_EXE_SIZE_B": client_exe_size,
            "HOME_TOTAL_MB": home_total,
            "HOME1_TOTAL_MB": home1_total,
            "EXECUTION_TIME": exec_time,
        })

    return pd.DataFrame(records)


def assign_degradation(machines: pd.DataFrame, failure_rate: float = 0.15) -> dict:
    assigned = {}
    for tc in machines["TERMINAL_CODE"]:
        if random.random() < failure_rate:
            fm = random.choice(FAILURE_MODES)
            deg_days = random.randint(5, 120)
            assigned[tc] = {
                "failure_mode": fm,
                "days_to_failure": deg_days,
                "start_of_degradation": None,
            }
        else:
            assigned[tc] = None
    return assigned


def simulate_dynamic_data_fast(
    static_df: pd.DataFrame,
    degradation_map: dict,
    days_of_history: int = 180,
    interval_minutes: int = 15,
    samples_per_machine: int = 2000,
    seed: int = 42,
) -> tuple:  # returns (DataFrame, updated_deg_map)
    np.random.seed(seed)
    random.seed(seed)

    end_date = datetime(2025, 1, 1, 0, 0, 0)
    start_date = end_date - timedelta(days=days_of_history)

    machine_list = static_df.to_dict("records")
    all_rows = []

    for machine in tqdm(machine_list, desc="Generating dynamic data"):
        tc = machine["TERMINAL_CODE"]
        home_total = machine["HOME_TOTAL_MB"]
        home1_total = machine["HOME1_TOTAL_MB"]
        db_page_size = machine["DB_PAGE_SIZE"]
        fault_info = degradation_map.get(tc)

        n = samples_per_machine
        t_offset_hours = np.sort(np.random.uniform(0, days_of_history * 24, n))
        timestamps = [start_date + timedelta(hours=h) for h in t_offset_hours]

        base_home_used = random.uniform(50.0, home_total * 0.4)
        base_home1_used = random.uniform(20.0, home1_total * 0.3)
        db_file_size_mb_base = random.uniform(10.0, 200.0)
        total_pages = int((db_file_size_mb_base * 1024 * 1024) / db_page_size)
        free_pages = int(total_pages * random.uniform(0.3, 0.7))
        mrr_log_size = random.uniform(0.5, 20.0)
        crash_log_size = random.uniform(0.1, 5.0)
        last_tkt = random.randint(100000, 999999)

        os_uptime = start_date - timedelta(days=random.randint(1, 60))
        db_uptime = start_date - timedelta(days=random.randint(1, 30))

        failure_happens = fault_info is not None
        if failure_happens:
            deg_days = fault_info["days_to_failure"]
            failure_date = end_date - timedelta(days=random.randint(1, 5))
            deg_start = failure_date - timedelta(days=deg_days)
            fault_info["start_of_degradation"] = deg_start
            fm = fault_info["failure_mode"]
        else:
            deg_start = None
            failure_date = None
            fm = None

        for j, t in enumerate(timestamps):
            hour = t.hour + t.minute / 60.0
            vol_factor = _ticket_volume_pattern(hour)
            if t.weekday() >= 5:
                vol_factor *= 0.4

            tkt_yest = int(np.random.poisson(max(1, 150 * vol_factor)))
            tkt_today = int(np.random.poisson(max(1, 120 * vol_factor)))
            net_link = 1 if random.random() < 0.95 else 0

            home_used = base_home_used + np.random.normal(0, 2.0)
            home1_used = base_home1_used + np.random.normal(0, 1.0)
            db_size = db_file_size_mb_base + np.random.normal(0, 0.5)
            fp = free_pages + int(np.random.normal(0, 10))
            mrr_log = max(0, mrr_log_size + np.random.normal(0, 0.3))
            crash_log = max(0, crash_log_size + np.random.normal(0, 0.1))

            if failure_happens and t >= deg_start and failure_date:
                progress = min(max((t - deg_start) / (failure_date - deg_start), 0.0), 1.0)

                if fm == "disk_exhaustion":
                    home_used += progress * (home_total * 0.55)
                    home1_used += progress * (home1_total * 0.55)
                elif fm == "db_full":
                    fp = max(0, int(fp * (1.0 - progress * 0.95)))
                    db_size += progress * 50.0
                elif fm == "db_corruption":
                    if progress > 0.7:
                        db_size += progress * 30.0
                        fp = int(fp * (1.0 - progress * 0.5))
                    net_link = 0 if progress > 0.9 else net_link
                elif fm == "memory_leak":
                    home_used += progress * (home_total * 0.45)
                elif fm == "log_growth":
                    mrr_log += progress * 200.0
                    crash_log += progress * 50.0
                elif fm == "network_degradation":
                    if progress > 0.3:
                        net_link = 0 if random.random() < progress * 0.5 else net_link

            home_used = min(max(home_used, 0), home_total)
            home1_used = min(max(home1_used, 0), home1_total)
            db_size = max(0.1, db_size)
            fp = max(0, fp)
            mrr_log = max(0, mrr_log)
            crash_log = max(0, crash_log)

            last_tkt += random.randint(0, 5)
            last_txn = t - timedelta(seconds=random.randint(0, 300))
            last_dl = t - timedelta(seconds=random.randint(0, 600))
            last_ul = t - timedelta(seconds=random.randint(0, 600))

            all_rows.append({
                "TERMINAL_CODE": tc,
                "LAST_TKT_NUMBER": str(last_tkt),
                "LAST_TXN_TIME": last_txn,
                "LAST_DOWNLOAD_TIME": last_dl,
                "LAST_UPLOAD_TIME": last_ul,
                "TOTAL_TKT_ISSUED_YESTERDAY": tkt_yest,
                "TOTAL_TKT_ISSUED_TODAY": tkt_today,
                "NETWORK_LINK": net_link,
                "HOME_USED_MB": round(home_used, 2),
                "HOME1_USED_MB": round(home1_used, 2),
                "OS_UP_SINCE": os_uptime,
                "DB_UP_SINCE": db_uptime,
                "DB_FILE_SIZE_MB": round(db_size, 2),
                "DB_FILE_SIZE_IN_PAGES": total_pages,
                "DB_FREE_PAGES": fp,
                "DB_PURGING_UPTO": t - timedelta(hours=random.randint(0, 48)),
                "DB_MRR_LOG_DIR_SIZE_MB": round(mrr_log, 2),
                "DB_CRASH_LOG_DIR_SIZE_MB": round(crash_log, 2),
                "CLIENT_LOG_EXIST_SINCE": t - timedelta(days=random.randint(1, 30)),
                "TC_MAKE": random.choice(TC_MAKES),
                "PASSWORD_SYBUTS": "****",
                "PASSWORD_ROOT": "****",
                "EXECUTION_TIME": t,
            })

    return pd.DataFrame(all_rows), degradation_map


def _ticket_volume_pattern(hour: float) -> float:
    if 6 <= hour < 10:
        return 1.5 + (hour - 6) * 0.3
    elif 10 <= hour < 14:
        return 2.0 + (hour - 10) * 0.1
    elif 14 <= hour < 18:
        return 1.5 - (hour - 14) * 0.1
    elif 18 <= hour < 22:
        return 0.8 - (hour - 18) * 0.1
    else:
        return max(0.1, 0.3 - (hour % 24) * 0.02)


def build_labels(static_df, dynamic_df, degradation_map):
    labels = []
    end_date = dynamic_df["EXECUTION_TIME"].max()

    for tc in tqdm(static_df["TERMINAL_CODE"], desc="Building labels"):
        info = degradation_map.get(tc)
        machine_data = dynamic_df[dynamic_df["TERMINAL_CODE"] == tc].sort_values("EXECUTION_TIME")

        if info is None:
            for _, row in machine_data.iterrows():
                labels.append({
                    "TERMINAL_CODE": tc,
                    "EXECUTION_TIME": row["EXECUTION_TIME"],
                    "fault_label": 0,
                    "rul_days": -1,
                    "failure_mode": "none",
                })
        else:
            failure_date = info["start_of_degradation"] + timedelta(days=info["days_to_failure"])
            for _, row in machine_data.iterrows():
                t = row["EXECUTION_TIME"]
                if t >= info["start_of_degradation"]:
                    remaining = (failure_date - t).total_seconds() / 86400.0
                    fault_label = 1 if remaining <= info["days_to_failure"] else 0
                    rul = max(0, remaining)
                else:
                    fault_label = 0
                    rul = -1
                labels.append({
                    "TERMINAL_CODE": tc,
                    "EXECUTION_TIME": t,
                    "fault_label": fault_label,
                    "rul_days": round(rul, 4),
                    "failure_mode": info["failure_mode"],
                })

    return pd.DataFrame(labels)


def generate_all(
    num_machines: int = 20000,
    days_of_history: int = 180,
    failure_rate: float = 0.15,
    samples_per_machine: int = 1500,
    output_dir: str = ".",
) -> tuple:
    os.makedirs(output_dir, exist_ok=True)

    print(f"Generating static data for {num_machines} machines...")
    static = generate_static_data(num_machines)
    static.to_csv(os.path.join(output_dir, "static_data.csv"), index=False)
    print(f"  -> {len(static)} records")

    print("Assigning degradation profiles...")
    deg_map = assign_degradation(static, failure_rate)
    n_failing = sum(1 for v in deg_map.values() if v is not None)
    print(f"  -> {n_failing} machines with failure profiles")

    print(f"Generating dynamic data ({days_of_history} days, ~{samples_per_machine} samples/machine)...")
    dynamic, deg_map = simulate_dynamic_data_fast(static, deg_map, days_of_history,
                                                   samples_per_machine=samples_per_machine)
    dynamic.to_csv(os.path.join(output_dir, "dynamic_data.csv"), index=False)
    print(f"  -> {len(dynamic)} records")

    print("Building labels...")
    labels = build_labels(static, dynamic, deg_map)
    labels.to_csv(os.path.join(output_dir, "labels.csv"), index=False)
    print(f"  -> {len(labels)} label records")

    fm_counts = labels[labels["fault_label"] == 1]["failure_mode"].value_counts()
    for fm, cnt in fm_counts.items():
        print(f"  {fm}: {cnt}")

    return static, dynamic, labels, deg_map


if __name__ == "__main__":
    generate_all(num_machines=500, output_dir="..", samples_per_machine=1000)
