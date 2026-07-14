import os
import sys
import json
import asyncio
import io
import base64
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.transformer_rul import TimeSeriesTransformer, count_parameters
from features.feature_engineering import (
    engineer_static_features,
    engineer_dynamic_features,
    create_sequences,
    DYNAMIC_FEATURES,
    STATIC_FEATURES,
)
from inference.predict import RULPredictor
from training.train import train_epoch, eval_model

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
MODEL_DIRS = {
    "model_quick": os.path.join(PROJECT_ROOT, "model_quick"),
    "model_med": os.path.join(PROJECT_ROOT, "model_med"),
    "saved_models": os.path.join(PROJECT_ROOT, "saved_models"),
}

app = FastAPI(title="Predictive Maintenance Dashboard", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")


def _load_csv(path):
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def _json_safe(val):
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    return val


@app.get("/", response_class=HTMLResponse)
async def index():
    with open(os.path.join(os.path.dirname(__file__), "templates", "index.html"), encoding="utf-8") as f:
        return f.read()


@app.get("/api/status")
async def api_status():
    return {
        "status": "ok",
        "project": "Predictive Maintenance for Railway Ticketing Machines",
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/models")
async def list_models():
    available = {}
    for name, path in MODEL_DIRS.items():
        config_path = os.path.join(path, "config.json")
        model_path = os.path.join(path, "best_model.pt")
        history_path = os.path.join(path, "training_history.json")
        if os.path.exists(config_path) and os.path.exists(model_path):
            with open(config_path) as f:
                config = json.load(f)
            available[name] = {
                "path": path,
                "config": {k: _json_safe(v) for k, v in config.items()},
                "has_history": os.path.exists(history_path),
                "size_mb": round(os.path.getsize(model_path) / (1024 * 1024), 2),
            }
    return {"models": available, "default": "model_quick" if "model_quick" in available else list(available.keys())[0] if available else None}


@app.get("/api/model/{model_name}/history")
async def get_training_history(model_name: str):
    if model_name not in MODEL_DIRS:
        raise HTTPException(404, f"Model dir '{model_name}' not found")
    history_path = os.path.join(MODEL_DIRS[model_name], "training_history.json")
    if not os.path.exists(history_path):
        raise HTTPException(404, "No training history found")
    with open(history_path) as f:
        history = json.load(f)
    return {"history": history}


@app.get("/api/data/overview")
async def data_overview():
    static = _load_csv(os.path.join(PROJECT_ROOT, "static_data.csv"))
    dynamic = _load_csv(os.path.join(PROJECT_ROOT, "dynamic_data.csv"))
    labels = _load_csv(os.path.join(PROJECT_ROOT, "labels.csv"))
    predictions = _load_csv(os.path.join(PROJECT_ROOT, "predictions.csv"))

    overview = {
        "static_machines": len(static) if static is not None else 0,
        "dynamic_records": len(dynamic) if dynamic is not None else 0,
        "labels_records": len(labels) if labels is not None else 0,
        "predictions_count": len(predictions) if predictions is not None else 0,
    }

    if predictions is not None and len(predictions) > 0:
        overview["at_risk"] = int(predictions["is_fault_risk"].sum())
        overview["avg_fault_prob"] = round(float(predictions["fault_probability"].mean()), 4)
        overview["top_risk"] = predictions.head(10).to_dict(orient="records")
        for r in overview["top_risk"]:
            r["fault_probability"] = round(float(r["fault_probability"]), 4)
            r["predicted_rul_days"] = round(float(r["predicted_rul_days"]), 2)

    if labels is not None and len(labels) > 0:
        overview["total_faults"] = int(labels["fault_label"].sum())
        overview["fault_rate"] = round(float(labels["fault_label"].mean() * 100), 2)
        if "failure_mode" in labels.columns:
            fm_counts = labels[labels["fault_label"] == 1]["failure_mode"].value_counts()
            overview["failure_modes"] = {str(k): int(v) for k, v in fm_counts.items()}

    return overview


@app.get("/api/data/machines")
async def list_machines(search: Optional[str] = Query(None)):
    static = _load_csv(os.path.join(PROJECT_ROOT, "static_data.csv"))
    predictions = _load_csv(os.path.join(PROJECT_ROOT, "predictions.csv"))

    if static is None:
        return {"machines": []}

    machines = static.to_dict(orient="records")
    for m in machines:
        for k in list(m.keys()):
            m[k] = _json_safe(m[k])

    if predictions is not None:
        pred_map = predictions.set_index("TERMINAL_CODE").to_dict("index")
        for m in machines:
            tc = m["TERMINAL_CODE"]
            if tc in pred_map:
                m["fault_probability"] = round(float(pred_map[tc]["fault_probability"]), 4)
                m["predicted_rul_days"] = round(float(pred_map[tc]["predicted_rul_days"]), 2)
                m["is_fault_risk"] = bool(pred_map[tc]["is_fault_risk"])
            else:
                m["fault_probability"] = None
                m["predicted_rul_days"] = None
                m["is_fault_risk"] = None

    if search:
        machines = [m for m in machines if search.lower() in str(m.get("TERMINAL_CODE", "")).lower()]

    return {"machines": machines, "total": len(machines)}


@app.get("/api/data/labels")
async def get_labels():
    labels = _load_csv(os.path.join(PROJECT_ROOT, "labels.csv"))
    if labels is None:
        return {"labels": []}
    return {"labels": labels.to_dict(orient="records")}


@app.post("/api/predict")
async def run_prediction(model_name: str = "model_quick", seq_len: int = 24):
    if model_name not in MODEL_DIRS:
        raise HTTPException(404, f"Model '{model_name}' not found")

    model_dir = MODEL_DIRS[model_name]
    config_path = os.path.join(model_dir, "config.json")
    model_path = os.path.join(model_dir, "best_model.pt")

    if not os.path.exists(config_path) or not os.path.exists(model_path):
        raise HTTPException(400, f"Model files not found in {model_dir}")

    static_path = os.path.join(PROJECT_ROOT, "static_data.csv")
    dynamic_path = os.path.join(PROJECT_ROOT, "dynamic_data.csv")

    if os.path.exists(static_path) and os.path.exists(dynamic_path):
        static_df = pd.read_csv(static_path)
        dynamic_df = pd.read_csv(dynamic_path)
    else:
        sys.path.insert(0, PROJECT_ROOT)
        from data.synthetic_data_generator import generate_all
        static_df, dynamic_df, _, _ = generate_all(
            num_machines=50, days_of_history=30,
            samples_per_machine=300, output_dir=PROJECT_ROOT,
        )

    try:
        predictor = RULPredictor(model_path, config_path, device="cpu")
        predictions = predictor.predict(static_df, dynamic_df, seq_len)
        predictions.to_csv(os.path.join(PROJECT_ROOT, "predictions.csv"), index=False)

        n_risk = int(predictions["is_fault_risk"].sum())
        return {
            "success": True,
            "predictions_count": len(predictions),
            "at_risk": n_risk,
            "at_risk_pct": round(100 * n_risk / len(predictions), 1) if len(predictions) > 0 else 0,
            "sample": predictions.head(20).to_dict(orient="records"),
        }
    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {str(e)}")


@app.post("/api/train")
async def run_training(
    num_machines: int = 100,
    days: int = 30,
    epochs: int = 5,
    seq_len: int = 24,
    stride: int = 8,
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 3,
):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _train_sync, {
        "num_machines": num_machines, "days": days, "epochs": epochs,
        "seq_len": seq_len, "stride": stride, "d_model": d_model,
        "nhead": nhead, "num_layers": num_layers,
    })
    return result


def _train_sync(config: dict) -> dict:
    device = torch.device("cpu")
    from data.synthetic_data_generator import generate_all

    static_df, dynamic_df, labels_df, _ = generate_all(
        num_machines=config["num_machines"], days_of_history=config["days"],
        failure_rate=0.15, samples_per_machine=300,
        output_dir=os.path.join(PROJECT_ROOT, "data_quick"),
    )

    static_feat = engineer_static_features(static_df)
    dynamic_feat = engineer_dynamic_features(dynamic_df, static_df)

    X, y_fault, y_rul = create_sequences(
        dynamic_feat, labels_df, static_feat,
        sequence_length=config["seq_len"], stride=config["stride"],
    )

    from sklearn.model_selection import train_test_split
    X_train, X_val, yf_train, yf_val, yr_train, yr_val = train_test_split(
        X, y_fault, y_rul, test_size=0.2, random_state=42, stratify=y_fault
    )

    from torch.utils.data import DataLoader, TensorDataset
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(yf_train), torch.tensor(yr_train)),
        batch_size=16, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val), torch.tensor(yf_val), torch.tensor(yr_val)),
        batch_size=16, shuffle=False,
    )

    model = TimeSeriesTransformer(
        input_dim=X.shape[2], d_model=config["d_model"], nhead=config["nhead"],
        num_encoder_layers=config["num_layers"], dim_feedforward=config["d_model"] * 4,
        dropout=0.1,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)

    history = []
    for epoch in range(1, config["epochs"] + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, device)
        val_metrics, _, _ = eval_model(model, val_loader, device)
        history.append({
            "epoch": epoch,
            "train_loss": round(float(train_metrics["loss"]), 4),
            "val_fault_accuracy": round(float(val_metrics["fault_accuracy"]), 4),
            "val_fault_f1": round(float(val_metrics["fault_f1"]), 4),
            "val_fault_precision": round(float(val_metrics["fault_precision"]), 4),
            "val_fault_recall": round(float(val_metrics["fault_recall"]), 4),
            "val_rul_mae": round(float(val_metrics["rul_mae"]), 4),
        })

    model_dir = os.path.join(PROJECT_ROOT, "model_quick")
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_dir, "best_model.pt"))
    torch.save(model.state_dict(), os.path.join(model_dir, "final_model.pt"))
    with open(os.path.join(model_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    config_out = {
        "input_dim": X.shape[2], "d_model": config["d_model"], "nhead": config["nhead"],
        "num_layers": config["num_layers"], "n_parameters": count_parameters(model),
        "num_machines": config["num_machines"], "epochs": config["epochs"],
    }
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(config_out, f, indent=2)

    return {
        "success": True,
        "model_dir": model_dir,
        "history": history,
        "final_metrics": history[-1] if history else {},
    }


@app.websocket("/ws/train")
async def websocket_train(websocket: WebSocket):
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        loop = asyncio.get_event_loop()

        def train_with_ws(config: dict, ws_sender):
            import sys as _sys
            device = torch.device("cpu")
            from data.synthetic_data_generator import generate_all

            static_df, dynamic_df, labels_df, _ = generate_all(
                num_machines=config["num_machines"], days_of_history=config["days"],
                failure_rate=0.15, samples_per_machine=300,
                output_dir=os.path.join(PROJECT_ROOT, "data_quick"),
            )
            ws_sender({"type": "log", "message": f"Generated data for {config['num_machines']} machines"})

            static_feat = engineer_static_features(static_df)
            dynamic_feat = engineer_dynamic_features(dynamic_df, static_df)

            X, y_fault, y_rul = create_sequences(
                dynamic_feat, labels_df, static_feat,
                sequence_length=config["seq_len"], stride=config["stride"],
            )
            ws_sender({"type": "log", "message": f"Created {len(X)} sequences (seq_len={config['seq_len']})"})

            from sklearn.model_selection import train_test_split
            X_train, X_val, yf_train, yf_val, yr_train, yr_val = train_test_split(
                X, y_fault, y_rul, test_size=0.2, random_state=42, stratify=y_fault
            )

            from torch.utils.data import DataLoader, TensorDataset
            train_loader = DataLoader(
                TensorDataset(torch.tensor(X_train), torch.tensor(yf_train), torch.tensor(yr_train)),
                batch_size=16, shuffle=True,
            )
            val_loader = DataLoader(
                TensorDataset(torch.tensor(X_val), torch.tensor(yf_val), torch.tensor(yr_val)),
                batch_size=16, shuffle=False,
            )

            model = TimeSeriesTransformer(
                input_dim=X.shape[2], d_model=config["d_model"], nhead=config["nhead"],
                num_encoder_layers=config["num_layers"], dim_feedforward=config["d_model"] * 4,
                dropout=0.1,
            ).to(device)

            optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
            ws_sender({"type": "log", "message": f"Model created: {count_parameters(model):,} parameters"})

            history = []
            for epoch in range(1, config["epochs"] + 1):
                train_metrics = train_epoch(model, train_loader, optimizer, device)
                val_metrics, _, _ = eval_model(model, val_loader, device)
                epoch_data = {
                    "epoch": epoch,
                    "train_loss": round(float(train_metrics["loss"]), 4),
                    "val_fault_accuracy": round(float(val_metrics["fault_accuracy"]), 4),
                    "val_fault_f1": round(float(val_metrics["fault_f1"]), 4),
                    "val_fault_precision": round(float(val_metrics["fault_precision"]), 4),
                    "val_fault_recall": round(float(val_metrics["fault_recall"]), 4),
                    "val_rul_mae": round(float(val_metrics["rul_mae"]), 4),
                }
                history.append(epoch_data)
                ws_sender({"type": "epoch", "data": epoch_data})

            model_dir = os.path.join(PROJECT_ROOT, "model_quick")
            os.makedirs(model_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(model_dir, "best_model.pt"))
            torch.save(model.state_dict(), os.path.join(model_dir, "final_model.pt"))
            with open(os.path.join(model_dir, "training_history.json"), "w") as f:
                json.dump(history, f, indent=2)

            config_out = {
                "input_dim": X.shape[2], "d_model": config["d_model"], "nhead": config["nhead"],
                "num_layers": config["num_layers"], "n_parameters": count_parameters(model),
                "num_machines": config["num_machines"], "epochs": config["epochs"],
            }
            with open(os.path.join(model_dir, "config.json"), "w") as f:
                json.dump(config_out, f, indent=2)

            ws_sender({"type": "complete", "history": history, "final_metrics": history[-1] if history else {}})

        async def send_msg(msg):
            try:
                await websocket.send_json(msg)
            except Exception:
                pass

        def ws_sender(msg):
            import asyncio as _a
            try:
                coro = send_msg(msg)
                fut = asyncio.run_coroutine_threadsafe(coro, loop)
                fut.result(timeout=5)
            except Exception:
                pass

        await loop.run_in_executor(None, train_with_ws, data, ws_sender)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
