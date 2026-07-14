# Predictive Maintenance for Railway Ticketing Machines

LLM-inspired Transformer model for fault prediction and Remaining Useful Life (RUL) estimation in railway ticketing machines.

## Features

- **Transformer-based model** — predicts fault probability and remaining useful life from sensor time-series data
- **Synthetic data generator** — creates realistic ticketing machine sensor data with configurable failure modes
- **Real dataset support** — AI4I 2020, MetroPT-3, SCANIA (via `train_with_real_data.py`)
- **Full training pipeline** — data generation → feature engineering → sequence creation → training → evaluation → visualization
- **Web dashboard** — FastAPI + JS SPA with live training via WebSocket

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the web dashboard
python start_frontend.py
# Open http://localhost:8000
```

Or with Docker:

```bash
docker compose up --build
# Open http://localhost:8000
```

## Usage

### Web Dashboard (Recommended)

Open the dashboard in your browser for an interactive experience:

| Page | Description |
|------|-------------|
| **Dashboard** | Overview stats, at-risk machines, failure mode distribution |
| **Machines** | Browse/search all machines with fault probability & RUL |
| **Predictions** | Run inference on trained models |
| **Training** | Configure hyperparameters and train with real-time WebSocket progress |
| **Analytics** | View training history and model performance charts |

### Command Line

```bash
# Quick training (100 machines)
python run_full_pipeline.py --quick

# Medium training (500 machines)
python run_full_pipeline.py --medium

# Train on real datasets
python train_with_real_data.py --dataset ai4i2020
python train_with_real_data.py --dataset metropt3

# Run inference with trained model
python inference/predict.py --model-path model_quick/best_model.pt --config-path model_quick/config.json
```

## Project Structure

```
├── frontend/                  # Web dashboard (FastAPI + JS)
│   ├── main.py               # API server with WebSocket training
│   ├── templates/index.html   # SPA frontend
│   └── static/                # CSS & JS assets
├── data/                      # Data generation & loading
│   ├── synthetic_data_generator.py
│   └── real_data_loader.py
├── features/                  # Feature engineering
│   └── feature_engineering.py
├── models/                    # Transformer model
│   └── transformer_rul.py
├── training/                  # Training loop
│   └── train.py
├── evaluation/                # Visualization & reporting
│   └── visualize.py
├── inference/                 # Prediction/inference
│   └── predict.py
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Model Architecture

The `TimeSeriesTransformer` uses:
- **Positional encoding** for temporal awareness
- **Transformer encoder layers** with multi-head self-attention
- **Adaptive average pooling** over the sequence dimension
- **Dual output heads**: fault classification + RUL regression
- Configurable: `d_model`, `nhead`, `num_layers`, `dropout`

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | SPA dashboard |
| GET | `/api/status` | Health check |
| GET | `/api/models` | List trained models |
| GET | `/api/model/{name}/history` | Training history |
| GET | `/api/data/overview` | Data statistics |
| GET | `/api/data/machines` | Machine list |
| POST | `/api/predict` | Run inference |
| POST | `/api/train` | Train model |
| WS | `/ws/train` | Live training progress |
