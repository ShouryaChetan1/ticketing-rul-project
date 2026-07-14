import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class TimeSeriesTransformer(nn.Module):
    """
    Transformer-based "LLM" for predictive maintenance.
    
    Takes a sequence of dynamic sensor readings (with static features
    tiled per timestep) and outputs:
      - fault_logits: binary classification (fault / no fault)
      - rul_pred: regression (remaining useful life in days)
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 200,
        rul_alpha: float = 1.0,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        self.seq_pool = nn.AdaptiveAvgPool1d(1)

        self.fault_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2),
        )

        self.rul_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
        )

        self.rul_alpha = rul_alpha

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> dict:
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.dropout(x)

        memory = self.encoder(x)

        pooled = memory.transpose(1, 2)
        pooled = self.seq_pool(pooled).squeeze(-1)

        fault_logits = self.fault_head(pooled)
        rul_pred = self.rul_head(pooled).squeeze(-1)

        result = {
            "fault_logits": fault_logits,
            "rul_pred": rul_pred,
        }

        return result

    def compute_loss(
        self,
        fault_logits: torch.Tensor,
        rul_pred: torch.Tensor,
        fault_labels: torch.Tensor,
        rul_labels: torch.Tensor,
        pos_weight: float = None,
    ) -> dict:
        if pos_weight is not None:
            weight = torch.tensor([1.0, pos_weight], device=fault_logits.device)
            fault_loss = nn.functional.cross_entropy(
                fault_logits, fault_labels.long(), weight=weight
            )
        else:
            fault_loss = nn.functional.cross_entropy(
                fault_logits, fault_labels.long()
            )

        rul_mask = fault_labels > 0.5
        if rul_mask.sum() > 0:
            rul_loss = nn.functional.mse_loss(
                rul_pred[rul_mask], rul_labels[rul_mask]
            )
        else:
            rul_loss = torch.tensor(0.0, device=rul_pred.device)

        total_loss = fault_loss + self.rul_alpha * rul_loss

        return {
            "total_loss": total_loss,
            "fault_loss": fault_loss,
            "rul_loss": rul_loss,
        }


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
