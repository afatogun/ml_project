"""
Script to clean likely mislabeled 'Miscellaneous' rows from the training data.
Workflow:
1. Split data 50/50 (stratified by sector)
2. PASS 1: Train on first 50%, predict on second 50% → flag mislabeled in second 50%
3. PASS 2: Train on second 50%, predict on first 50% → flag mislabeled in first 50%
4. Combine flagged rows from both passes
5. Remove all flagged rows from the full dataset
6. Save cleaned dataset and report
"""
from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.openai_embeddings import batch_embed
from src.infer import predict_with_confidence
from src.train_models import train_all_classifiers, SEED
from src.preprocess import load_training_data
import os

TRAIN_XLSX = './data/raw/2025_classification_training_set.xlsx'
CLEANED_XLSX = './data/raw/2025_classification_training_set.cleaned.xlsx'
REMOVED_CSV = './data/raw/misc_removed.csv'
CONFIDENCE_THRESHOLD = 0.8

print("="*60)
print("CLEANING MISLABELED MISCELLANEOUS ROWS")
print("="*60)

# 1. Load data
print("\n[1/6] Loading data...")
df = load_training_data(TRAIN_XLSX)
print(f"  Total rows: {len(df)}")

# 2. Stratified 50/50 split
print("\n[2/6] Creating 50/50 split...")
split_a, split_b = train_test_split(
    df, test_size=0.5, random_state=SEED, stratify=df['sector']
)
print(f"  Split A: {len(split_a)} rows")
print(f"  Split B: {len(split_b)} rows")

# ==== PASS 1: Train on A, predict on B ====
print("\n[3/6] PASS 1: Training on Split A, predicting on Split B...")
print("  Computing embeddings for Split A...")
embeddings_a = batch_embed(split_a['description'].tolist())
print("  Training classifiers...")
train_splits_a = {
    'train': {'df': split_a, 'embeddings': embeddings_a},
    'test': {'df': split_a, 'embeddings': embeddings_a},
}
classifiers_a = train_all_classifiers(train_splits_a, classifier_type='lightgbm')

print("  Computing embeddings for Split B...")
embeddings_b = batch_embed(split_b['description'].tolist())
print("  Predicting on Split B...")
sector_preds_b, sector_confs_b = predict_with_confidence(
    classifiers_a['sector']['model'],
    classifiers_a['sector']['encoder'],
    embeddings_b
)

split_b_results = split_b.copy()
split_b_results['pred_sector'] = sector_preds_b
split_b_results['pred_sector_conf'] = sector_confs_b

# Flag Miscellaneous rows with confident non-misc prediction in Split B
misc_mask_b = split_b_results['sector'].str.lower() == 'miscellaneous'
confident_mask_b = (split_b_results['pred_sector'].str.lower() != 'miscellaneous') & \
                   (split_b_results['pred_sector_conf'] >= CONFIDENCE_THRESHOLD)
flagged_b = split_b_results[misc_mask_b & confident_mask_b]
print(f"  Flagged from Split B: {len(flagged_b)} rows")

# ==== PASS 2: Train on B, predict on A ====
print("\n[4/6] PASS 2: Training on Split B, predicting on Split A...")
print("  Training classifiers on Split B...")
train_splits_b = {
    'train': {'df': split_b, 'embeddings': embeddings_b},
    'test': {'df': split_b, 'embeddings': embeddings_b},
}
classifiers_b = train_all_classifiers(train_splits_b, classifier_type='lightgbm')

print("  Predicting on Split A...")
sector_preds_a, sector_confs_a = predict_with_confidence(
    classifiers_b['sector']['model'],
    classifiers_b['sector']['encoder'],
    embeddings_a
)

split_a_results = split_a.copy()
split_a_results['pred_sector'] = sector_preds_a
split_a_results['pred_sector_conf'] = sector_confs_a

# Flag Miscellaneous rows with confident non-misc prediction in Split A
misc_mask_a = split_a_results['sector'].str.lower() == 'miscellaneous'
confident_mask_a = (split_a_results['pred_sector'].str.lower() != 'miscellaneous') & \
                   (split_a_results['pred_sector_conf'] >= CONFIDENCE_THRESHOLD)
flagged_a = split_a_results[misc_mask_a & confident_mask_a]
print(f"  Flagged from Split A: {len(flagged_a)} rows")

# ==== Combine flagged rows from both passes ====
print("\n[5/6] Combining flagged rows from both passes...")
all_flagged = pd.concat([flagged_a, flagged_b], ignore_index=True)
print(f"  Total flagged for removal: {len(all_flagged)} rows")

# Remove flagged rows from full dataset
cleaned_df = df[~df['id'].isin(all_flagged['id'])]

# Save cleaned data and report
print("\n[6/6] Saving results...")
cleaned_df.to_excel(CLEANED_XLSX, index=False)
all_flagged.to_csv(REMOVED_CSV, index=False)

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Original rows:         {len(df)}")
print(f"Flagged for removal:   {len(all_flagged)}")
print(f"Cleaned rows:          {len(cleaned_df)}")
print(f"Reduction:             {len(all_flagged)/len(df)*100:.2f}%")
print(f"\nCleaned data saved to: {CLEANED_XLSX}")
print(f"Removed rows saved to: {REMOVED_CSV}")
print("\nNext step: Train on the cleaned dataset with:")
print(f"  python train.py --train_xlsx {CLEANED_XLSX}")
