"""Giải thích model (Ch.7): hệ số LR, importance RF, SHAP cho XGBoost.

XGBoost: dùng giá trị SHAP có sẵn của XGBoost, không cần cài thêm thư viện shap:
    booster.predict(xgboost.DMatrix(X_đã_tiền_xử_lý), pred_contribs=True)   # cột cuối là bias
Lấy các bước từ pipeline.named_steps (tên bước: config.PREP_STEPS, config.STEP_MODEL).

Lưu ý khi viết báo cáo: đây là mức độ đóng góp của sensor vào cảnh báo, không phải quan hệ nhân quả.
Chỉ số cột phải được map về cột gốc trong 590 cột, không phải chỉ số sau khi lọc.

Kết quả trả về được đánh nhãn theo cột gốc:
- cột giá trị cảm biến: nhãn là số nguyên = chỉ số cột gốc (0..589);
- cột chỉ báo thiếu (chỉ có khi config.ADD_MISSING_INDICATOR = True): nhãn "missing_<chỉ số cột gốc>".
"""
import numpy as np
import pandas as pd
import xgboost
from sklearn.impute import SimpleImputer

import config


def feature_labels(pipeline) -> list:
    """Nhãn (theo cột gốc) của từng cột đi vào bước model, đúng thứ tự cột của model.

    Đi lần lượt các bước tiền xử lý theo thứ tự trong pipeline:
    - bước có get_support() ("missing", "variance", "select"): chỉ giữ các cột mask = True;
    - "impute" (SimpleImputer): bỏ cột rỗng hoàn toàn (nếu có) và, nếu có add_indicator,
      nối thêm các cột chỉ báo thiếu vào CUỐI (đúng thứ tự output của SimpleImputer);
    - các bước khác ("scale"): không đổi số cột.
    """
    prep_steps = [(name, step) for name, step in pipeline.steps if name != config.STEP_MODEL]
    first_step = prep_steps[0][1]
    labels = np.array(list(range(len(first_step.get_support()))), dtype=object)

    for name, step in prep_steps:
        if isinstance(step, SimpleImputer):
            # Cột có statistics_ = NaN (rỗng hoàn toàn trên train) bị SimpleImputer bỏ đi
            # khi keep_empty_features=False (mặc định).
            value_labels = labels
            if not step.keep_empty_features:
                value_labels = labels[~np.isnan(step.statistics_)]
            indicator_labels = []
            if step.indicator_ is not None:
                indicator_labels = [f"missing_{labels[i]}" for i in step.indicator_.features_]
            labels = np.array(list(value_labels) + indicator_labels, dtype=object)
        elif hasattr(step, "get_support"):
            labels = labels[step.get_support()]

    n_model_features = pipeline.named_steps[config.STEP_MODEL].n_features_in_
    assert len(labels) == n_model_features, (
        f"Map cot sai: {len(labels)} nhan nhung model nhan {n_model_features} cot")
    return list(labels)


def surviving_mask(pipeline) -> np.ndarray:
    """Mảng boolean dài bằng số cột gốc (590): True = cột giá trị cảm biến sống sót tới bước model."""
    n_original = len(pipeline.steps[0][1].get_support())
    mask = np.zeros(n_original, dtype=bool)
    for label in feature_labels(pipeline):
        if not isinstance(label, str):
            mask[label] = True
    return mask


def _transform(pipeline, X):
    """Chạy X qua các bước tiền xử lý (không gồm model)."""
    for name, step in pipeline.steps:
        if name != config.STEP_MODEL:
            X = step.transform(X)
    return X


def lr_coefficients(pipeline):
    """Hệ số LR (coef_[0]) gắn nhãn cột gốc. Trả về pd.Series."""
    coef = pipeline.named_steps[config.STEP_MODEL].coef_[0]
    return pd.Series(coef, index=feature_labels(pipeline), name="coef")


def rf_importance(pipeline):
    """feature_importances_ của Random Forest gắn nhãn cột gốc. Trả về pd.Series."""
    importance = pipeline.named_steps[config.STEP_MODEL].feature_importances_
    return pd.Series(importance, index=feature_labels(pipeline), name="importance")


def xgb_shap(pipeline, X):
    """Giá trị SHAP của XGBoost cho từng mẫu của X (đủ 590 cột gốc).

    Trả về DataFrame (n_mẫu, n_cột_vào_model), cột gắn nhãn cột gốc, đã bỏ cột bias.
    """
    X_prep = _transform(pipeline, X)
    booster = pipeline.named_steps[config.STEP_MODEL].get_booster()
    contribs = booster.predict(xgboost.DMatrix(X_prep), pred_contribs=True)[:, :-1]
    index = X.index if isinstance(X, pd.DataFrame) else None
    return pd.DataFrame(contribs, index=index, columns=feature_labels(pipeline))
