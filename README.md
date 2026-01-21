# Project Classification with OpenAI Embeddings

Train and batch-predict project classifications (Sector, Subcategory, Type) using OpenAI embeddings and lightweight classifiers.

## Features

- **OpenAI Embeddings**: Uses `text-embedding-3-small` (configurable) for semantic representations
- **Gated Hierarchical Classification**:
  - Sector classifier (always predicts)
  - Subcategory classifier (skipped if Sector=Miscellaneous or confidence below threshold)
  - Type classifier (skipped if confidence below threshold)
- **Batch Processing**: Rate-limit friendly with exponential backoff retries
- **Reproducibility**: Deterministic seeds, data hashing, model versioning

## Project Structure

```
ml_project/
├── train.py                 # Training entry point
├── predict_xlsx.py          # Batch prediction entry point
├── src/
│   ├── __init__.py
│   ├── openai_embeddings.py # Batch embed with retries
│   ├── preprocess.py        # Data loading and normalization
│   ├── train_models.py      # Model training utilities
│   ├── metrics.py           # Evaluation and reporting
│   └── infer.py             # Inference utilities
├── artifacts/               # Saved models (per run_id)
├── reports/                 # Confusion matrices
├── data/
│   ├── raw/                 # Input data
│   └── processed/           # Processed data
├── requirements.txt
├── .env.example
└── README.md
```

## Installation

```bash
# Create virtual environment (Python 3.11+)
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your OpenAI API key
```

## Training

### Input Format

Training XLSX must have columns:
- `id`: Unique identifier
- `Sector`: Primary category (required)
- `subcategory`: Secondary category (optional)
- `type`: Tertiary category (optional)
- `Description`: Text description for embedding

### Command

```bash
python train.py --train_xlsx ./data/raw/2025_classification_training_set.xlsx
```

### Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--train_xlsx` | (required) | Path to training XLSX |
| `--embedding_model` | `text-embedding-3-small` | OpenAI embedding model |
| `--batch_size` | `100` | Texts per API call |
| `--output_dir` | `./artifacts` | Model artifacts directory |
| `--reports_dir` | `./reports` | Evaluation reports directory |
| `--subcategory_threshold` | `0.60` | Confidence threshold for subcategory |
| `--type_threshold` | `0.60` | Confidence threshold for type |

### Output

- `./artifacts/<run_id>/`: Model files and metadata
  - `sector_model.joblib`, `sector_encoder.joblib`
  - `subcategory_model.joblib`, `subcategory_encoder.joblib`
  - `type_model.joblib`, `type_encoder.joblib`
  - `metadata.json`: Configuration and metrics
- `./reports/`: Confusion matrices (CSV)

## Batch Prediction

### Input Format

Scoring XLSX must have columns:
- `id`: Unique identifier
- `description`: Text description for embedding

### Command

```bash
python predict_xlsx.py \
    --model_dir ./artifacts/20240115_120000 \
    --input_xlsx input.xlsx \
    --output_xlsx output.xlsx
```

### Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_dir` | (required) | Path to artifacts directory |
| `--input_xlsx` | (required) | Input XLSX path |
| `--output_xlsx` | (required) | Output XLSX path |
| `--batch_size` | `100` | Texts per API call |
| `--subcategory_threshold` | (from model) | Override confidence threshold |
| `--type_threshold` | (from model) | Override confidence threshold |

### Output Columns

| Column | Description |
|--------|-------------|
| `id` | Original ID |
| `description` | Original description |
| `pred_sector` | Predicted sector |
| `pred_sector_conf` | Sector confidence (0-1) |
| `pred_subcategory` | Predicted subcategory (or null) |
| `pred_subcategory_conf` | Subcategory confidence (0-1) |
| `pred_type` | Predicted type (or null) |
| `pred_type_conf` | Type confidence (0-1) |
| `notes` | Gating/threshold notes |
| `model_version` | Run ID of model used |

## Gating Rules

1. **Sector**: Always predicted
2. **Subcategory**:
   - Skipped if `pred_sector == "Miscellaneous"` (note added)
   - Skipped if confidence < threshold (note added)
3. **Type**:
   - Skipped if confidence < threshold (note added)

## Example Workflow

```bash
# 1. Train models
python train.py --train_xlsx ./data/raw/2025_classification_training_set.xlsx

# 2. Check artifacts
ls ./artifacts/

# 3. Run predictions
python predict_xlsx.py \
    --model_dir ./artifacts/20260119_143022 \
    --input_xlsx ./data/raw/new_projects.xlsx \
    --output_xlsx ./data/processed/predictions.xlsx
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key (required) |

## License

MIT
