"""Random Forest. Xử lý imbalance bằng class_weight."""
import json

import numpy as np
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

import config
from src.data_loader import load_data
from src.preprocess import build_preprocessor
from src.split import time_split
from src.threshold import choose_threshold

PARAM_GRID = {}


def build_model(class_weight: str | None = "balanced_subsample") -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=300, class_weight=class_weight, random_state=config.SEED, n_jobs=-1)


def run_ablation() -> dict:
    """Lưu vào config.ablation_path("rf")."""
    # Cùng cấu trúc với logistic.run_ablation(): chỉ dùng train, số liệu tính trên OOF, không đụng test.
    X, y, timestamps = load_data()
    X_train, _, y_train, _ = time_split(X, y, timestamps)
    y_train_arr = y_train.to_numpy()

    def oof_proba_for(class_weight, k_features):
        # Không import được src.train (import vòng) nên tự lắp pipeline giống build_pipeline.
        skf = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED)
        oof = np.zeros(len(y_train), dtype=float)
        for fold_train_idx, fold_val_idx in skf.split(X_train, y_train):
            prep = build_preprocessor(k_features)
            model = build_model(class_weight=class_weight)
            pipeline = Pipeline(list(prep.steps) + [(config.STEP_MODEL, model)])
            pipeline.fit(X_train.iloc[fold_train_idx], y_train.iloc[fold_train_idx])
            oof[fold_val_idx] = pipeline.predict_proba(X_train.iloc[fold_val_idx])[:, config.FAIL]
        return oof

    def metrics(variant, oof, threshold):
        y_pred = (oof >= threshold).astype(int)
        return {
            "variant": variant,
            "threshold": float(threshold),
            "precision": float(precision_score(y_train_arr, y_pred, zero_division=0)),
            "recall": float(recall_score(y_train_arr, y_pred, zero_division=0)),
            "f1": float(f1_score(y_train_arr, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_train_arr, oof)),
            "pr_auc": float(average_precision_score(y_train_arr, oof)),
        }

    results = []
    # 1. baseline: class_weight="balanced_subsample", có chọn đặc trưng.
    oof_baseline = oof_proba_for("balanced_subsample", config.K_FEATURES)
    results.append(metrics("baseline", oof_baseline, choose_threshold(y_train_arr, oof_baseline, method="f1")))
    # 2. no_class_weight: bỏ class_weight.
    oof = oof_proba_for(None, config.K_FEATURES)
    results.append(metrics("no_class_weight", oof, choose_threshold(y_train_arr, oof, method="f1")))
    # 3. no_feature_selection: bỏ bước "select".
    oof = oof_proba_for("balanced_subsample", None)
    results.append(metrics("no_feature_selection", oof, choose_threshold(y_train_arr, oof, method="f1")))
    # 4. threshold_0.5: dùng lại OOF của baseline, chỉ đổi ngưỡng thành 0.5.
    results.append(metrics("threshold_0.5", oof_baseline, 0.5))

    path = config.ablation_path("rf")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[rf] ablation -> {path}")
    for row in results:
        print(row)
    return results
