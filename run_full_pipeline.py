#!/usr/bin/env python
"""
End-to-end pipeline for LLM-based Predictive Maintenance
for Railway Ticketing Machines.
"""
import os
import sys
import json
import argparse

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))

from data.synthetic_data_generator import generate_all
from features.feature_engineering import (
    engineer_static_features,
    engineer_dynamic_features,
    create_sequences,
    DYNAMIC_FEATURES,
    STATIC_FEATURES,
)
from models.transformer_rul import TimeSeriesTransformer, count_parameters
from training.train import train_epoch, eval_model
from evaluation.visualize import (
    plot_training_history,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_rul_scatter,
    plot_failure_mode_breakdown,
    generate_report,
)

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


def run_pipeline(
    num_machines: int = 500,
    days: int = 60,
    failure_rate: float = 0.15,
    samples_per_machine: int = 500,
    epochs: int = 10,
    seq_len: int = 24,
    stride: int = 6,
    batch_size: int = 32,
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 3,
    lr: float = 1e-3,
    rul_alpha: float = 1.0,
    data_dir: str = "data_output",
    model_dir: str = "saved_models",
    viz_dir: str = "viz_output",
    seed: int = 42,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)

    print("\n[1/6] GENERATING SYNTHETIC DATA...")
    static_df, dynamic_df, labels_df, deg_map = generate_all(
        num_machines=num_machines,
        days_of_history=days,
        failure_rate=failure_rate,
        samples_per_machine=samples_per_machine,
        output_dir=data_dir,
    )

    print("\n[2/6] ENGINEERING FEATURES...")
    static_feat = engineer_static_features(static_df)
    dynamic_feat = engineer_dynamic_features(dynamic_df, static_df)
    print(f"  Dynamic features used ({len(DYNAMIC_FEATURES)}): {DYNAMIC_FEATURES}")
    print(f"  Static features used ({len(STATIC_FEATURES)}): {STATIC_FEATURES}")

    print("\n[3/6] CREATING SEQUENCES...")
    X, y_fault, y_rul = create_sequences(
        dynamic_feat, labels_df, static_feat,
        sequence_length=seq_len, stride=stride,
    )
    print(f"  Sequences shape: {X.shape}")
    print(f"  Fault rate: {y_fault.mean()*100:.1f}%")

    X_train, X_val, yf_train, yf_val, yr_train, yr_val = train_test_split(
        X, y_fault, y_rul, test_size=0.2, random_state=seed, stratify=y_fault
    )

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train), torch.tensor(yf_train), torch.tensor(yr_train)
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_val), torch.tensor(yf_val), torch.tensor(yr_val)
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    print("\n[4/6] BUILDING TRANSFORMER MODEL...")
    input_dim = X.shape[2]
    model = TimeSeriesTransformer(
        input_dim=input_dim,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_layers,
        dim_feedforward=d_model * 4,
        dropout=0.1,
        rul_alpha=rul_alpha,
    ).to(device)

    print(f"  Input dim: {input_dim}, Params: {count_parameters(model):,}")
    print(f"  Architecture: {num_layers} enc layers, {nhead} heads, d_model={d_model}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Normalize RUL to [0,1] for stable multi-task training
    rul_max = max(y_rul[y_fault > 0.5].max(), 1.0)
    yr_train_norm = yr_train / rul_max
    yr_val_norm = yr_val / rul_max

    from torch.utils.data import WeightedRandomSampler
    train_labels_t = torch.tensor(yf_train).long()
    class_counts = train_labels_t.bincount()
    class_weights = 1.0 / class_counts.float()
    sample_weights = class_weights[train_labels_t]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_dataset = TensorDataset(
        torch.tensor(X_train), train_labels_t, torch.tensor(yr_train_norm)
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_val), torch.tensor(yf_val), torch.tensor(yr_val_norm)
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    print(f"\n[5/6] TRAINING ({epochs} epochs)...")
    best_val_f1 = 0.0
    best_val_loss = float("inf")
    history = []

    print(f"  Class-balanced sampling enabled (fault rate={y_fault.mean()*100:.1f}%)")
    print(f"  RUL normalized by {rul_max:.1f} (max RUL in train set)")

    for epoch in range(1, epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, device)
        val_metrics, val_fault_probs, val_rul_preds = eval_model(model, val_loader, device)
        # Denormalize RUL for reporting
        val_rul_preds_denorm = val_rul_preds * rul_max
        val_metrics["rul_mae"] *= rul_max
        val_metrics["rul_rmse"] *= rul_max
        scheduler.step()

        history.append({
            "epoch": epoch,
            "train_loss": float(train_metrics["loss"]),
            "train_fault_loss": float(train_metrics["fault_loss"]),
            "train_rul_loss": float(train_metrics["rul_loss"]),
            **{f"val_{k}": float(v) for k, v in val_metrics.items()},
        })

        log = (
            f"  Epoch {epoch:2d}/{epochs} | "
            f"Loss {train_metrics['loss']:.4f} | "
            f"Acc {val_metrics['fault_accuracy']:.3f} "
            f"F1 {val_metrics['fault_f1']:.3f} "
            f"RUL MAE {val_metrics['rul_mae']:.2f}d"
        )
        print(log)

        if val_metrics["fault_f1"] > best_val_f1:
            best_val_f1 = val_metrics["fault_f1"]
            torch.save(model.state_dict(), os.path.join(model_dir, "best_model.pt"))
            print(f"    -> Best model saved (F1={best_val_f1:.3f})")

    torch.save(model.state_dict(), os.path.join(model_dir, "final_model.pt"))
    with open(os.path.join(model_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    config = dict(
        num_machines=num_machines,
        days=days,
        failure_rate=failure_rate,
        samples_per_machine=samples_per_machine,
        seq_len=seq_len,
        stride=stride,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        input_dim=input_dim,
        n_parameters=count_parameters(model),
    )
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    y_val_all = []
    yr_val_all = []
    for _, yf, yr in val_loader:
        y_val_all.extend(yf.numpy())
        yr_val_all.extend(yr.numpy())
    y_val = np.array(y_val_all)
    yr_val = np.array(yr_val_all)

    print(f"\n[6/6] FINAL EVALUATION...")
    best_path = os.path.join(model_dir, "best_model.pt")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
        print(f"  Loaded best model (F1={best_val_f1:.3f})")
    else:
        print("  No best model saved, using final model")
    final_metrics, fault_probs, rul_preds_norm = eval_model(model, val_loader, device)
    rul_preds = rul_preds_norm * rul_max

    # Recompute metrics with denormalized RUL
    fault_binary = (fault_probs > 0.5).astype(int)
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    rul_mask = y_val > 0.5
    if rul_mask.sum() > 0:
        final_metrics["rul_mae"] = mean_absolute_error(
            yr_val[rul_mask] * rul_max, rul_preds[rul_mask]
        )
        final_metrics["rul_rmse"] = np.sqrt(
            mean_squared_error(yr_val[rul_mask] * rul_max, rul_preds[rul_mask])
        )

    print("\n  Final Metrics:")
    for k, v in final_metrics.items():
        print(f"    {k}: {v:.4f}")

    print("\n  Generating visualizations...")
    plot_training_history(os.path.join(model_dir, "training_history.json"), viz_dir)
    plot_confusion_matrix(y_val, fault_binary, viz_dir)
    plot_roc_curve(y_val, fault_probs, viz_dir)

    rul_true = y_val[y_val > 0.5]
    rul_pred_plot = rul_preds[y_val > 0.5]
    if len(rul_true) > 0:
        plot_rul_scatter(rul_true * rul_max, rul_pred_plot, viz_dir)

    plot_failure_mode_breakdown(labels_df, viz_dir)
    generate_report(final_metrics, viz_dir)

    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE")
    print(f"  Model:     {model_dir}")
    print(f"  Data:      {data_dir}")
    print(f"  Reports:   {viz_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Quick test (100 machines)")
    parser.add_argument("--medium", action="store_true", help="Medium test (500 machines)")
    parser.add_argument("--output", default=None, help="Output directory prefix")
    args = parser.parse_args()

    prefix = args.output or ""

    if args.quick:
        run_pipeline(
            num_machines=100, days=30, samples_per_machine=300,
            epochs=15, seq_len=24, stride=8,
            batch_size=16, d_model=64, nhead=4, num_layers=3,
            lr=5e-4, rul_alpha=0.1,
            data_dir=f"{prefix}data_quick", model_dir=f"{prefix}model_quick",
            viz_dir=f"{prefix}viz_quick",
        )
    elif args.medium:
        run_pipeline(
            num_machines=500, days=60, samples_per_machine=500,
            epochs=15, seq_len=24, stride=6,
            batch_size=32, d_model=128, nhead=4, num_layers=3,
            lr=5e-4, rul_alpha=0.1,
            data_dir=f"{prefix}data_med", model_dir=f"{prefix}model_med",
            viz_dir=f"{prefix}viz_med",
        )
    else:
        print("Usage:")
        print("  python run_full_pipeline.py --quick   (100 machines, 5 epochs)")
        print("  python run_full_pipeline.py --medium  (500 machines, 10 epochs)")
        print("\nFor full 20K-machine training, edit run_full_pipeline.py directly or import in a script.")
