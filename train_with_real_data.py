#!/usr/bin/env python
"""
Train predictive maintenance model on real-world datasets.

Downloads and prepares datasets automatically, then runs
the full training pipeline.

Usage:
    python train_with_real_data.py --dataset ai4i2020 --subset 200
    python train_with_real_data.py --dataset ai4i2020            (all 10K machines)
    python train_with_real_data.py --dataset metropt3
"""
import os, sys, json, argparse, numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(__file__))
from data.real_data_loader import load_dataset
from features.feature_engineering import (engineer_static_features, engineer_dynamic_features, create_sequences)
from models.transformer_rul import TimeSeriesTransformer, count_parameters
from training.train import train_epoch, eval_model
from evaluation.visualize import (plot_training_history, plot_confusion_matrix, plot_roc_curve, plot_rul_scatter, generate_report)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='ai4i2020', choices=['ai4i2020', 'metropt3'])
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--seq-len', type=int, default=12)
    parser.add_argument('--stride', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--subset', type=int, default=0, help='Use only N machines (0=all)')
    parser.add_argument('--output', default='output_real')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    data_dir = os.path.join(args.output, 'data')
    model_dir = os.path.join(args.output, 'model')
    viz_dir = os.path.join(args.output, 'viz')
    for d in [data_dir, model_dir, viz_dir]:
        os.makedirs(d, exist_ok=True)

    print(f'\n[1/5] LOADING {args.dataset.upper()} DATASET')
    static, dynamic, labels = load_dataset(args.dataset, output_dir=data_dir)

    if args.subset > 0:
        codes = dynamic['TERMINAL_CODE'].unique()[:args.subset]
        dynamic = dynamic[dynamic['TERMINAL_CODE'].isin(codes)]
        static = static[static['TERMINAL_CODE'].isin(codes)]
        labels = labels[labels['TERMINAL_CODE'].isin(codes)]
        print(f'  Using subset: {len(codes)} machines')

    print(f'\n[2/5] FEATURE ENGINEERING')
    static_feat = engineer_static_features(static)
    dynamic_feat = engineer_dynamic_features(dynamic, static)
    print(f'  Dynamic features: {dynamic_feat.shape}')

    print(f'\n[3/5] CREATING SEQUENCES')
    X, y_fault, y_rul = create_sequences(dynamic_feat, labels, static_feat,
                                          sequence_length=args.seq_len, stride=args.stride)
    print(f'  Sequences: {X.shape}, fault rate: {y_fault.mean()*100:.1f}%')
    if len(X) < 50:
        print('  Too few sequences. Exiting.')
        return

    y_bin = (y_fault > 0.5).astype(int)
    X_train, X_val, yf_train, yf_val, yr_train, yr_val = train_test_split(
        X, y_bin, y_rul, test_size=0.2, random_state=42, stratify=y_bin)

    rul_max = max(yr_train[yf_train > 0.5].max(), 1.0)
    yr_train_norm = yr_train / rul_max
    yr_val_norm = yr_val / rul_max

    train_labels_t = torch.tensor(yf_train).long()
    class_counts = train_labels_t.bincount()
    sample_weights = 1.0 / class_counts.float()
    sampler = WeightedRandomSampler(sample_weights[train_labels_t], len(train_labels_t), replacement=True)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), train_labels_t, torch.tensor(yr_train_norm)),
        batch_size=args.batch_size, sampler=sampler)
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val), torch.tensor(yf_val).long(), torch.tensor(yr_val_norm)),
        batch_size=args.batch_size, shuffle=False)

    print(f'\n[4/5] BUILDING & TRAINING MODEL')
    model = TimeSeriesTransformer(
        input_dim=X.shape[2], d_model=128, nhead=8, num_encoder_layers=4,
        dim_feedforward=512, dropout=0.1, rul_alpha=0.1).to(device)
    print(f'  Parameters: {count_parameters(model):,}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_f1, history = 0, []
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, device)
        val_metrics, fp, rp = eval_model(model, val_loader, device)
        val_metrics['rul_mae'] *= rul_max
        val_metrics['rul_rmse'] *= rul_max
        scheduler.step()

        history.append({'epoch': epoch, 'train_loss': float(train_metrics['loss']),
                        **{f'val_{k}': float(v) for k, v in val_metrics.items()}})

        log = (f'  Epoch {epoch:2d}/{args.epochs} | Loss {train_metrics["loss"]:.4f} | '
               f'Acc {val_metrics["fault_accuracy"]:.3f} F1 {val_metrics["fault_f1"]:.3f} '
               f'RUL MAE {val_metrics["rul_mae"]:.2f}d')
        print(log)
        if val_metrics['fault_f1'] > best_f1:
            best_f1 = val_metrics['fault_f1']
            torch.save(model.state_dict(), os.path.join(model_dir, 'best_model.pt'))
            print(f'    -> Saved (F1={best_f1:.3f})')

    torch.save(model.state_dict(), os.path.join(model_dir, 'final_model.pt'))
    with open(os.path.join(model_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    config = dict(dataset=args.dataset, epochs=args.epochs, input_dim=X.shape[2],
                  n_parameters=count_parameters(model))
    with open(os.path.join(model_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    print(f'\n[5/5] EVALUATION')
    best_path = os.path.join(model_dir, 'best_model.pt')
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))

    y_val = np.array([yf.numpy() for _, yf, _ in val_loader]).ravel()
    final_metrics, fault_probs, rul_preds = eval_model(model, val_loader, device)
    final_metrics['rul_mae'] *= rul_max
    final_metrics['rul_rmse'] *= rul_max

    print('\nFinal Metrics:')
    for k, v in final_metrics.items():
        print(f'  {k}: {v:.4f}')

    plot_training_history(os.path.join(model_dir, 'training_history.json'), viz_dir)
    plot_confusion_matrix(y_val, (fault_probs > 0.5).astype(int), viz_dir)
    plot_roc_curve(y_val, fault_probs, viz_dir)
    plot_rul_scatter(y_val[y_val > 0.5], rul_preds[y_val > 0.5] * rul_max, viz_dir)
    generate_report(final_metrics, viz_dir)
    print(f'\nDone! Output in {args.output}/')


if __name__ == '__main__':
    main()
