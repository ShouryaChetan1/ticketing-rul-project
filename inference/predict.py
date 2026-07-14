import os
import sys
import json
import argparse

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.transformer_rul import TimeSeriesTransformer
from features.feature_engineering import (
    engineer_static_features,
    engineer_dynamic_features,
    DYNAMIC_FEATURES,
    STATIC_FEATURES,
)


class RULPredictor:
    def __init__(
        self,
        model_path: str,
        config_path: str,
        device: str = "cpu",
    ):
        with open(config_path) as f:
            self.config = json.load(f)

        self.device = torch.device(device)
        self.model = TimeSeriesTransformer(
            input_dim=self.config["input_dim"],
            d_model=self.config.get("d_model", 128),
            nhead=self.config.get("nhead", 8),
            num_encoder_layers=self.config.get("num_layers", 4),
            dim_feedforward=self.config.get("d_model", 128) * 4,
            dropout=0.1,
        ).to(self.device)

        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True)
        )
        self.model.eval()

    def predict(
        self,
        static_df: pd.DataFrame,
        dynamic_df: pd.DataFrame,
        sequence_length: int = 48,
    ) -> pd.DataFrame:
        static_feat = engineer_static_features(static_df)
        dynamic_feat = engineer_dynamic_features(dynamic_df, static_df)

        results = []
        with torch.no_grad():
            used_machines = set(dynamic_feat["TERMINAL_CODE"].unique())
            static_index = static_feat.set_index("TERMINAL_CODE")

            for tc in used_machines:
                machine_dynamic = dynamic_feat[dynamic_feat["TERMINAL_CODE"] == tc] \
                    .sort_values("EXECUTION_TIME")

                machine_static = static_index.loc[tc]
                static_vec = np.array([machine_static[f] for f in STATIC_FEATURES])

                if len(machine_dynamic) < sequence_length:
                    continue

                full_seq = machine_dynamic[DYNAMIC_FEATURES].values
                latest_features = full_seq[-sequence_length:]

                seq_with_static = np.concatenate(
                    [latest_features, np.tile(static_vec, (sequence_length, 1))],
                    axis=1,
                )
                seq_tensor = (
                    torch.tensor(seq_with_static, dtype=torch.float32)
                    .unsqueeze(0)
                    .to(self.device)
                )

                outputs = self.model(seq_tensor)
                probs = torch.softmax(outputs["fault_logits"], dim=1)
                fault_prob = probs[0, 1].item()
                rul_pred = outputs["rul_pred"][0].item()

                latest_time = machine_dynamic["EXECUTION_TIME"].iloc[-1]

                results.append(
                    {
                        "TERMINAL_CODE": tc,
                        "EXECUTION_TIME": latest_time,
                        "fault_probability": round(fault_prob, 4),
                        "predicted_rul_days": round(max(rul_pred, 0), 2),
                        "is_fault_risk": fault_prob > 0.5,
                    }
                )

        return pd.DataFrame(results).sort_values("fault_probability", ascending=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--static", default=None, help="Path to static data CSV (generated if not provided)")
    parser.add_argument("--dynamic", default=None, help="Path to dynamic data CSV (generated if not provided)")
    parser.add_argument("--model-path", default="model_quick/best_model.pt")
    parser.add_argument("--config-path", default="model_quick/config.json")
    parser.add_argument("--output", default="predictions.csv")
    parser.add_argument("--seq-len", type=int, default=24)
    parser.add_argument("--num-machines", type=int, default=50, help="Machines to generate if no data provided")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    if args.static and args.dynamic:
        print(f"Loading static data from {args.static}")
        static_df = pd.read_csv(args.static)
        print(f"Loading dynamic data from {args.dynamic}")
        dynamic_df = pd.read_csv(args.dynamic)
    else:
        print(f"No data files provided. Generating synthetic data for {args.num_machines} machines...")
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from data.synthetic_data_generator import generate_all
        static_df, dynamic_df, _, _ = generate_all(
            num_machines=args.num_machines, days_of_history=30,
            samples_per_machine=300, output_dir="."
        )

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    predictor = RULPredictor(args.model_path, args.config_path, device)

    print("Running predictions...")
    predictions = predictor.predict(static_df, dynamic_df, args.seq_len)
    predictions.to_csv(args.output, index=False)

    n_risk = predictions["is_fault_risk"].sum()
    print(f"\nPredictions saved to {args.output}")
    print(f"Machines at fault risk: {n_risk}/{len(predictions)} ({100*n_risk/len(predictions):.1f}%)")
    print(f"Top at-risk machines:")
    print(predictions.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
