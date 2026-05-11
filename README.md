# Supervised Project Classification

Train and evaluate supervised sector classification using OpenAI embeddings and reproducible holdout sampling.

## Supported Workflow (Supervised Only)

- Train supervised sector bundle: `train_supervised_sector.py`
- Run end-to-end evaluation: `run_test_suite.py`
- Batch predict from existing supervised bundle: `predict_supervised.py`
- Validate predictions: `validate_ai.py`

Legacy GPT-only and legacy predictor routes are archived.

## Project Structure

```text
ml_project/
├── train_supervised_sector.py
├── predict_supervised.py
├── run_test_suite.py
├── create_test_input.py
├── validate_ai.py
├── src/
│   ├── openai_embeddings.py
│   ├── preprocess.py
│   ├── supervised_candidates.py
│   ├── supervised_runtime.py
│   └── train_models.py
├── artifacts/
├── reports/
├── data/
├── requirements.txt
└── .env.example
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Set `OPENAI_API_KEY` in `.env`.

## Training

### Input

Training dataset (XLSX/CSV) must include:
- `id`
- `Sector`
- `Description`
- optional: `subcategory`, `type`, `tag`

### Command

```bash
python train_supervised_sector.py \
  --train_xlsx ./data/raw/2025_classification_training_set.xlsx \
  --embedding_model text-embedding-3-large \
  --candidate mlp_embedding_rule
```

### Output

Saved in `./artifacts/<run_id>_supervised/`:
- `sector_supervised_bundle.joblib`
- `metadata.json`
- `split_indices.npy` (canonical held-out 10% indices)

## End-to-End Evaluation

```bash
python run_test_suite.py \
  --skip_train \
  --model_dir ./artifacts/<run_id>_supervised \
  --train_xlsx ./data/raw/2025_classification_training_set.xlsx \
  --sample_size 1000 \
  --sample_source stratified_test \
  --random_mode
```

Notes:
- `--sample_source stratified_test` samples only from the saved holdout split when `split_indices.npy` exists.
- `--random_mode` changes sample draw each run while staying inside the same holdout pool.

## Batch Prediction

```bash
python predict_supervised.py \
  --model_dir ./artifacts/<run_id>_supervised \
  --input_xlsx ./data/test/<run_id>/test_input.xlsx \
  --output_xlsx ./data/test/<run_id>/test_predictions.xlsx
```

Output columns include:
- `pred_sector`, `pred_sector_conf`
- `top_3_predicted_sectors`, `top_3_predicted_probs`
- `prediction_source`, `model_version`, `candidate_name`
- `pred_tag` (currently defaults to `none`)

## Validation

```bash
python validate_ai.py \
  --predictions ./data/test/<run_id>/test_predictions.xlsx \
  --truth ./data/test/<run_id>/test_ground_truth.xlsx \
  --output-dir ./reports \
  --run-id <run_id>
```

## Reproducibility Guarantees

- Deterministic split seed for training holdout
- Persisted holdout indices in artifact (`split_indices.npy`)
- Dataset hash compatibility checks in `run_test_suite.py`

## Retention Policy

Active directories keep latest run per family; older runs are archived under `archive/`.
