"""Candidate supervised sector models for benchmarking GPT replacement."""

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import f1_score

import lightgbm as lgb

from .classifier_features import extract_classifier_features
from .metrics import export_sector_diagnostics, evaluate_classifier
from .train_models import SEED


class TfidfSectorClassifier:
    """Word+char TF-IDF sector classifier with logistic regression."""

    def __init__(self) -> None:
        self.word_vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 3),
            min_df=2,
            max_features=75000,
            strip_accents="unicode",
            sublinear_tf=True,
        )
        self.char_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            max_features=60000,
            sublinear_tf=True,
        )
        self.model = LogisticRegression(
            max_iter=3000,
            random_state=SEED,
            class_weight="balanced",
            solver="saga",
            multi_class="auto",
        )

    def fit(self, texts: List[str], y: np.ndarray) -> "TfidfSectorClassifier":
        word_matrix = self.word_vectorizer.fit_transform(texts)
        char_matrix = self.char_vectorizer.fit_transform(texts)
        features = hstack([word_matrix, char_matrix], format="csr")
        self.model.fit(features, y)
        return self

    def _transform(self, texts: List[str]):
        word_matrix = self.word_vectorizer.transform(texts)
        char_matrix = self.char_vectorizer.transform(texts)
        return hstack([word_matrix, char_matrix], format="csr")

    def predict(self, texts: List[str]) -> np.ndarray:
        return self.model.predict(self._transform(texts))

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        return self.model.predict_proba(self._transform(texts))


class TfidfReducedHybridClassifier:
    """Reduced TF-IDF + rule features + dense embeddings for hybrid Candidate D."""

    def __init__(self, n_components: int = 256) -> None:
        self.word_vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 3),
            min_df=2,
            max_features=75000,
            strip_accents="unicode",
            sublinear_tf=True,
        )
        self.char_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            max_features=60000,
            sublinear_tf=True,
        )
        self.svd = TruncatedSVD(n_components=n_components, random_state=SEED)
        self.model = LogisticRegression(
            max_iter=4000,
            random_state=SEED,
            class_weight="balanced",
            solver="lbfgs",
            multi_class="auto",
        )

    def fit(self, texts: List[str], dense_embeddings: np.ndarray, rule_features: np.ndarray, y: np.ndarray) -> "TfidfReducedHybridClassifier":
        sparse_features = self._fit_sparse(texts)
        reduced_features = self.svd.fit_transform(sparse_features)
        design_matrix = np.hstack([dense_embeddings, reduced_features, rule_features])
        self.model.fit(design_matrix, y)
        return self

    def _fit_sparse(self, texts: List[str]):
        word_matrix = self.word_vectorizer.fit_transform(texts)
        char_matrix = self.char_vectorizer.fit_transform(texts)
        return hstack([word_matrix, char_matrix], format="csr")

    def _transform_sparse(self, texts: List[str]):
        word_matrix = self.word_vectorizer.transform(texts)
        char_matrix = self.char_vectorizer.transform(texts)
        return hstack([word_matrix, char_matrix], format="csr")

    def _build_matrix(self, texts: List[str], dense_embeddings: np.ndarray, rule_features: np.ndarray) -> np.ndarray:
        sparse_features = self._transform_sparse(texts)
        reduced_features = self.svd.transform(sparse_features)
        return np.hstack([dense_embeddings, reduced_features, rule_features])

    def predict(self, texts: List[str], dense_embeddings: np.ndarray, rule_features: np.ndarray) -> np.ndarray:
        return self.model.predict(self._build_matrix(texts, dense_embeddings, rule_features))

    def predict_proba(self, texts: List[str], dense_embeddings: np.ndarray, rule_features: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(self._build_matrix(texts, dense_embeddings, rule_features))


def train_embedding_lgbm_sector_model(
    train_embeddings: np.ndarray,
    y_train: np.ndarray,
) -> Tuple[Any, LabelEncoder]:
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_train)

    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=15,
        num_leaves=64,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=SEED,
        class_weight="balanced",
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(train_embeddings, y_encoded)
    return model, encoder


def train_tfidf_logistic_sector_model(
    train_texts: List[str],
    y_train: np.ndarray,
) -> Tuple[TfidfSectorClassifier, LabelEncoder]:
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_train)

    model = TfidfSectorClassifier()
    model.fit(train_texts, y_encoded)
    return model, encoder


def train_embedding_rule_lgbm_sector_model(
    train_embeddings: np.ndarray,
    train_texts: List[str],
    y_train: np.ndarray,
) -> Tuple[Any, LabelEncoder]:
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_train)

    rule_frame = build_rule_feature_frame(train_texts)
    train_matrix = np.hstack([train_embeddings, rule_frame.to_numpy(dtype=np.float32)])

    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=15,
        num_leaves=64,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=SEED,
        class_weight="balanced",
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(train_matrix, y_encoded)
    return model, encoder


def build_rule_feature_frame(descriptions: List[str]) -> pd.DataFrame:
    rows = [extract_classifier_features(description) for description in descriptions]
    frame = pd.DataFrame(rows).fillna(False)
    for column in frame.columns:
        frame[column] = frame[column].astype(int)
    return frame


def train_mlp_sector_model(
    train_embeddings: np.ndarray,
    train_texts: List[str],
    y_train: np.ndarray,
) -> Tuple[Any, LabelEncoder, StandardScaler]:
    """2-layer MLP on L2-normalised embeddings + rule features."""
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_train)

    rule_frame = build_rule_feature_frame(train_texts)
    train_matrix = np.hstack([train_embeddings, rule_frame.to_numpy(dtype=np.float32)])

    scaler = StandardScaler()
    train_matrix = scaler.fit_transform(train_matrix)

    model = MLPClassifier(
        hidden_layer_sizes=(1024, 512, 256),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=256,
        learning_rate_init=1e-3,
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        random_state=SEED,
        verbose=False,
    )
    model.fit(train_matrix, y_encoded)
    return model, encoder, scaler


def _make_lgbm() -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
    )


def _make_mlp() -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=(1024, 512, 256),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=256,
        learning_rate_init=1e-3,
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        random_state=SEED,
        verbose=False,
    )


def train_stacked_sector_model(
    train_embeddings: np.ndarray,
    train_texts: List[str],
    y_train: np.ndarray,
) -> Tuple[Any, Any, Any, LabelEncoder, StandardScaler]:
    """Stacked ensemble: MLP + LightGBM base learners, LogisticRegression meta-learner.

    Uses a 20% stratified holdout to generate unbiased meta-features for the
    meta-learner, then retrains both base models on the full training set.
    """
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_train)

    rule_frame = build_rule_feature_frame(train_texts)
    rule_matrix = rule_frame.to_numpy(dtype=np.float32)
    n_rule = rule_matrix.shape[1]
    n_emb = train_embeddings.shape[1]

    scaler = StandardScaler()
    mlp_full = scaler.fit_transform(np.hstack([train_embeddings, rule_matrix]))
    lgbm_full = pd.DataFrame(
        np.hstack([train_embeddings, rule_matrix]),
        columns=[f"f{i}" for i in range(n_emb + n_rule)],
    )

    # --- Holdout split for meta-feature generation ---
    idx = np.arange(len(y_encoded))
    idx_base, idx_meta = train_test_split(
        idx, test_size=0.20, random_state=SEED, stratify=y_encoded
    )

    base_mlp = _make_mlp()
    base_mlp.fit(mlp_full[idx_base], y_encoded[idx_base])

    base_lgbm = _make_lgbm()
    base_lgbm.fit(lgbm_full.iloc[idx_base], y_encoded[idx_base])

    mlp_meta_probs = base_mlp.predict_proba(mlp_full[idx_meta])
    lgbm_meta_probs = base_lgbm.predict_proba(lgbm_full.iloc[idx_meta])
    meta_features = np.hstack([mlp_meta_probs, lgbm_meta_probs])

    meta_model = LogisticRegression(max_iter=1000, C=1.0, random_state=SEED)
    meta_model.fit(meta_features, y_encoded[idx_meta])

    # --- Retrain base models on full training data ---
    final_mlp = _make_mlp()
    final_mlp.fit(mlp_full, y_encoded)

    final_lgbm = _make_lgbm()
    final_lgbm.fit(lgbm_full, y_encoded)

    return final_mlp, final_lgbm, meta_model, encoder, scaler


def train_tfidf_reduced_rule_hybrid_sector_model(
    train_texts: List[str],
    train_embeddings: np.ndarray,
    y_train: np.ndarray,
) -> Tuple[TfidfReducedHybridClassifier, LabelEncoder]:
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_train)

    model = TfidfReducedHybridClassifier()
    rule_frame = build_rule_feature_frame(train_texts)
    model.fit(train_texts, train_embeddings, rule_frame.to_numpy(dtype=np.float32), y_encoded)
    return model, encoder


def _metrics_from_labels(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    return {
        "accuracy": round(float(np.mean(y_pred == y_true)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "n_samples": len(y_true),
    }


def _align_probabilities(source_encoder: LabelEncoder, target_encoder: LabelEncoder, probabilities: np.ndarray) -> np.ndarray:
    aligned = np.zeros((probabilities.shape[0], len(target_encoder.classes_)), dtype=np.float32)
    target_lookup = {label: idx for idx, label in enumerate(target_encoder.classes_)}
    for source_idx, label in enumerate(source_encoder.classes_):
        aligned[:, target_lookup[label]] = probabilities[:, source_idx]
    return aligned


def _build_ensemble_probabilities(
    embedding_probabilities: np.ndarray,
    tfidf_probabilities: np.ndarray,
    hybrid_probabilities: np.ndarray,
) -> np.ndarray:
    return (embedding_probabilities + tfidf_probabilities + hybrid_probabilities) / 3.0


def benchmark_sector_candidates(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
    reports_dir: str,
) -> Dict[str, Dict[str, Any]]:
    """Train first-wave supervised sector candidates and return metrics."""
    os_safe_reports_dir = reports_dir

    train_texts = train_df["description"].fillna("").astype(str).tolist()
    test_texts = test_df["description"].fillna("").astype(str).tolist()
    y_train = train_df["sector"].values
    y_test = test_df["sector"].values

    results: Dict[str, Dict[str, Any]] = {}

    embedding_model, embedding_encoder = train_embedding_lgbm_sector_model(train_embeddings, y_train)
    embedding_metrics = evaluate_classifier(embedding_model, embedding_encoder, test_embeddings, y_test)
    embedding_metrics["diagnostics"] = export_sector_diagnostics(
        embedding_model,
        embedding_encoder,
        test_embeddings,
        test_df,
        os_safe_reports_dir,
        "test_embedding_lgbm",
    )
    results["embedding_lgbm"] = embedding_metrics

    tfidf_model, tfidf_encoder = train_tfidf_logistic_sector_model(train_texts, y_train)
    tfidf_predictions = tfidf_model.predict(test_texts)
    tfidf_labels = tfidf_encoder.inverse_transform(tfidf_predictions)
    tfidf_metrics = _metrics_from_labels(y_test, tfidf_labels)
    tfidf_metrics["diagnostics"] = export_sector_diagnostics(
        tfidf_model,
        tfidf_encoder,
        test_texts,
        test_df,
        os_safe_reports_dir,
        "test_tfidf_logistic",
    )
    results["tfidf_logistic"] = tfidf_metrics

    test_rule_frame = build_rule_feature_frame(test_texts)

    hybrid_model, hybrid_encoder = train_embedding_rule_lgbm_sector_model(train_embeddings, train_texts, y_train)
    hybrid_test_matrix = np.hstack([test_embeddings, test_rule_frame.to_numpy(dtype=np.float32)])
    hybrid_metrics = evaluate_classifier(hybrid_model, hybrid_encoder, hybrid_test_matrix, y_test)
    hybrid_metrics["diagnostics"] = export_sector_diagnostics(
        hybrid_model,
        hybrid_encoder,
        hybrid_test_matrix,
        test_df,
        os_safe_reports_dir,
        "test_embedding_rule_lgbm",
    )
    results["embedding_rule_lgbm"] = hybrid_metrics

    mlp_model, mlp_encoder, mlp_scaler = train_mlp_sector_model(train_embeddings, train_texts, y_train)
    mlp_test_matrix = mlp_scaler.transform(
        np.hstack([test_embeddings, test_rule_frame.to_numpy(dtype=np.float32)])
    )
    mlp_metrics = evaluate_classifier(mlp_model, mlp_encoder, mlp_test_matrix, y_test)
    mlp_metrics["diagnostics"] = export_sector_diagnostics(
        mlp_model,
        mlp_encoder,
        mlp_test_matrix,
        test_df,
        os_safe_reports_dir,
        "test_mlp_embedding_rule",
    )
    results["mlp_embedding_rule"] = mlp_metrics

    # Candidate F: stacked MLP + LightGBM
    stack_mlp, stack_lgbm, stack_meta, stack_encoder, stack_scaler = train_stacked_sector_model(
        train_embeddings, train_texts, y_train
    )
    stack_rule = test_rule_frame.to_numpy(dtype=np.float32)
    stack_n_emb = test_embeddings.shape[1]
    stack_n_rule = stack_rule.shape[1]
    stack_mlp_test = stack_scaler.transform(np.hstack([test_embeddings, stack_rule]))
    stack_lgbm_test = pd.DataFrame(
        np.hstack([test_embeddings, stack_rule]),
        columns=[f"f{i}" for i in range(stack_n_emb + stack_n_rule)],
    )
    stack_mlp_probs = stack_mlp.predict_proba(stack_mlp_test)
    stack_lgbm_probs = stack_lgbm.predict_proba(stack_lgbm_test)
    stack_meta_probs = stack_meta.predict_proba(np.hstack([stack_mlp_probs, stack_lgbm_probs]))
    stack_preds = stack_encoder.inverse_transform(stack_meta_probs.argmax(axis=1))
    stack_metrics = _metrics_from_labels(y_test, stack_preds)
    stack_metrics["diagnostics"] = export_sector_diagnostics(
        ProbabilityEnsembleWrapper(stack_meta_probs),
        stack_encoder,
        test_embeddings,
        test_df,
        os_safe_reports_dir,
        "test_stacked_mlp_lgbm",
    )
    results["stacked_mlp_lgbm"] = stack_metrics

    candidate_d_model, candidate_d_encoder = train_tfidf_reduced_rule_hybrid_sector_model(
        train_texts,
        train_embeddings,
        y_train,
    )
    candidate_d_predictions = candidate_d_model.predict(
        test_texts,
        test_embeddings,
        test_rule_frame.to_numpy(dtype=np.float32),
    )
    candidate_d_labels = candidate_d_encoder.inverse_transform(candidate_d_predictions)
    candidate_d_metrics = _metrics_from_labels(y_test, candidate_d_labels)
    candidate_d_test_matrix = candidate_d_model._build_matrix(
        test_texts,
        test_embeddings,
        test_rule_frame.to_numpy(dtype=np.float32),
    )
    candidate_d_metrics["diagnostics"] = export_sector_diagnostics(
        candidate_d_model.model,
        candidate_d_encoder,
        candidate_d_test_matrix,
        test_df,
        os_safe_reports_dir,
        "test_candidate_d_tfidf_svd_rule",
    )
    results["candidate_d_tfidf_svd_rule"] = candidate_d_metrics

    embedding_probabilities = embedding_model.predict_proba(test_embeddings)
    tfidf_probabilities = _align_probabilities(
        tfidf_encoder,
        embedding_encoder,
        tfidf_model.predict_proba(test_texts),
    )
    hybrid_probabilities = _align_probabilities(
        candidate_d_encoder,
        embedding_encoder,
        candidate_d_model.predict_proba(
            test_texts,
            test_embeddings,
            test_rule_frame.to_numpy(dtype=np.float32),
        ),
    )
    ensemble_probabilities = _build_ensemble_probabilities(
        embedding_probabilities,
        tfidf_probabilities,
        hybrid_probabilities,
    )
    ensemble_predictions = ensemble_probabilities.argmax(axis=1)
    ensemble_labels = embedding_encoder.inverse_transform(ensemble_predictions)
    ensemble_metrics = _metrics_from_labels(y_test, ensemble_labels)
    ensemble_metrics["diagnostics"] = export_sector_diagnostics(
        ProbabilityEnsembleWrapper(ensemble_probabilities),
        embedding_encoder,
        test_embeddings,
        test_df,
        os_safe_reports_dir,
        "test_candidate_e_ensemble",
    )
    results["candidate_e_probability_ensemble"] = ensemble_metrics

    return results


class ProbabilityEnsembleWrapper:
    """Thin wrapper so diagnostics can consume precomputed probabilities."""

    def __init__(self, probabilities: np.ndarray) -> None:
        self._probabilities = probabilities

    def predict(self, _unused) -> np.ndarray:
        return self._probabilities.argmax(axis=1)

    def predict_proba(self, _unused) -> np.ndarray:
        return self._probabilities