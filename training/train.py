import os
import sys
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.synthetic_data_generator import generate_all
from features.feature_engineering import (
    engineer_static_features,
    engineer_dynamic_features,
    create_sequences,
)
from models.transformer_rul import TimeSeriesTransformer, count_parameters


def train_epoch(model, loader, optimizer, device, pos_weight=None):
    model.train()
    total_loss = 0
    total_fault = 0
    total_rul = 0
    n_batches = 0

    for batch_x, batch_fault, batch_rul in loader:
        batch_x = batch_x.to(device)
        batch_fault = batch_fault.to(device)
        batch_rul = batch_rul.to(device)

        optimizer.zero_grad()
        outputs = model(batch_x)
        losses = model.compute_loss(
            outputs["fault_logits"],
            outputs["rul_pred"],
            batch_fault,
            batch_rul,
            pos_weight=pos_weight,
        )
        losses["total_loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += losses["total_loss"].item()
        total_fault += losses["fault_loss"].item()
        total_rul += losses["rul_loss"].item()
        n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "fault_loss": total_fault / n_batches,
        "rul_loss": total_rul / n_batches,
    }


def eval_model(model, loader, device):
    model.eval()
    all_fault_preds = []
    all_rul_preds = []
    all_fault_true = []
    all_rul_true = []

    with torch.no_grad():
        for batch_x, batch_fault, batch_rul in loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            probs = torch.softmax(outputs["fault_logits"], dim=1)
            fault_preds = probs[:, 1].cpu().numpy()
            rul_preds = outputs["rul_pred"].cpu().numpy()

            all_fault_preds.extend(fault_preds)
            all_rul_preds.extend(rul_preds)
            all_fault_true.extend(batch_fault.numpy())
            all_rul_true.extend(batch_rul.numpy())

    all_fault_preds = np.array(all_fault_preds)
    all_rul_true = np.array(all_rul_true)
    all_fault_true = np.array(all_fault_true)
    all_rul_preds = np.array(all_rul_preds)

    fault_binary = (all_fault_preds > 0.5).astype(int)

    metrics = {
        "fault_accuracy": accuracy_score(all_fault_true, fault_binary),
        "fault_precision": precision_score(all_fault_true, fault_binary, zero_division=0),
        "fault_recall": recall_score(all_fault_true, fault_binary, zero_division=0),
        "fault_f1": f1_score(all_fault_true, fault_binary, zero_division=0),
    }

    rul_mask = all_fault_true > 0.5
    if rul_mask.sum() > 0:
        metrics["rul_mae"] = mean_absolute_error(all_rul_true[rul_mask], all_rul_preds[rul_mask])
        metrics["rul_rmse"] = np.sqrt(mean_squared_error(all_rul_true[rul_mask], all_rul_preds[rul_mask]))
    else:
        metrics["rul_mae"] = 0.0
        metrics["rul_rmse"] = 0.0

    return metrics, all_fault_preds, all_rul_preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-machines", type=int, default=20000)
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--failure-rate", type=float, default=0.15)
    parser.add_argument("--seq-len", type=int, default=48)
    parser.add_argument("--stride", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--rul-alpha", type=float, default=1.0)
    parser.add_argument("--data-dir", type=str, default="data_output")
    parser.add_argument("--model-dir", type=str, default="saved_models")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cpu") if args.cpu else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    print("=" * 60)
    print("GENERATING SYNTHETIC DATA")
    print("=" * 60)
    static_df, dynamic_df, labels_df, deg_map = generate_all(
        num_machines=args.num_machines,
        days_of_history=args.days,
        failure_rate=args.failure_rate,
        output_dir=args.data_dir,
    )

    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)
    static_feat = engineer_static_features(static_df)
    dynamic_feat = engineer_dynamic_features(dynamic_df, static_df)

    print(f"Dynamic features shape: {dynamic_feat.shape}")
    print(f"Feature columns ({len(DYNAMIC_FEATURES)}): {DYNAMIC_FEATURES}")
    print(f"Static feature columns ({len(STATIC_FEATURES)}): {STATIC_FEATURES}")

    print("\n" + "=" * 60)
    print("CREATING SEQUENCES")
    print("=" * 60)
    X, y_fault, y_rul = create_sequences(
        dynamic_feat, labels_df, static_feat,
        sequence_length=args.seq_len, stride=args.stride,
    )
    print(f"Sequences shape: {X.shape}")
    print(f"Fault labels: {y_fault.sum():.0f} positive / {len(y_fault)} total ({y_fault.mean()*100:.1f}%)")
    print(f"RUL labels (faulty only): mean={y_rul[y_fault>0.5].mean():.1f} days, "
          f"std={y_rul[y_fault>0.5].std():.1f} days")

    X_train, X_val, yf_train, yf_val, yr_train, yr_val = train_test_split(
        X, y_fault, y_rul, test_size=0.2, random_state=args.seed, stratify=y_fault
    )

    train_dataset = TensorDataset(
        torch.tensor(X_train), torch.tensor(yf_train), torch.tensor(yr_train)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val), torch.tensor(yf_val), torch.tensor(yr_val)
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    print("\n" + "=" * 60)
    print("BUILDING TRANSFORMER MODEL (LLM)")
    print("=" * 60)
    input_dim = X.shape[2]
    model = TimeSeriesTransformer(
        input_dim=input_dim,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_layers,
        dim_feedforward=args.d_model * 4,
        dropout=0.1,
        rul_alpha=args.rul_alpha,
    ).to(device)

    print(f"Input features per timestep: {input_dim}")
    print(f"Model parameters: {count_parameters(model):,}")
    print(f"Sequence length: {args.seq_len}")
    print(f"Architecture: {args.num_layers} encoder layers, {args.nhead} heads, d_model={args.d_model}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)
    best_val_f1 = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_metrics = train_epoch(model, train_loader, optimizer, device)
        val_metrics, _, _ = eval_model(model, val_loader, device)

        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]

        elapsed = time.time() - t0
        history.append({
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_fault_loss": train_metrics["fault_loss"],
            "train_rul_loss": train_metrics["rul_loss"],
            **{f"val_{k}": v for k, v in val_metrics.items()},
        })

        print(
            f"Epoch {epoch:2d}/{args.epochs} | "
            f"Loss: {train_metrics['loss']:.4f} (F: {train_metrics['fault_loss']:.4f} R: {train_metrics['rul_loss']:.4f}) | "
            f"Val Acc: {val_metrics['fault_accuracy']:.4f} F1: {val_metrics['fault_f1']:.4f} "
            f"RUL MAE: {val_metrics['rul_mae']:.2f}d | "
            f"LR: {lr_now:.2e} | {elapsed:.1f}s"
        )

        if val_metrics["fault_f1"] > best_val_f1:
            best_val_f1 = val_metrics["fault_f1"]
            torch.save(model.state_dict(), os.path.join(args.model_dir, "best_model.pt"))
            print(f"  -> New best model saved (F1={best_val_f1:.4f})")

    torch.save(model.state_dict(), os.path.join(args.model_dir, "final_model.pt"))
    with open(os.path.join(args.model_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    with open(os.path.join(args.model_dir, "config.json"), "w") as f:
        config = vars(args)
        config["input_dim"] = input_dim
        config["n_parameters"] = count_parameters(model)
        json.dump(config, f, indent=2)

    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)
    model.load_state_dict(
        torch.load(os.path.join(args.model_dir, "best_model.pt"), map_location=device)
    )
    final_metrics, fault_probs, rul_preds = eval_model(model, val_loader, device)

    for k, v in final_metrics.items():
        print(f"  {k}: {v:.4f}")

    print("\nDone!")


if __name__ == "__main__":
    from features.feature_engineering import DYNAMIC_FEATURES, STATIC_FEATURES
    main()
