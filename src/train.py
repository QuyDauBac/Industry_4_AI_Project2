"""Runner chung. Các file khác (evaluate, explain, dashboard) phụ thuộc file này -> giao bản thô sớm.

Chạy (từ thư mục gốc của repo):
    python -m src.train --model lr|rf|xgb                      huấn luyện + chọn ngưỡng
    python -m src.train --model lr --threshold-method f1       f1 (mặc định) | min_recall | budget
    python -m src.train --model lr --ablation                  chạy ablation (gọi run_ablation của model)
    python -m src.train --model lr --time-block K              robustness: train/test ở giai đoạn khác

Chạy thường tạo ra 4 file. Đường dẫn LẤY TỪ config, không tự ghép chuỗi:
- config.oof_path(model)        results/oof_proba/<model>.npy    P(Fail) out-of-fold trên train (5-fold stratified)
- config.test_path(model)       results/test_proba/<model>.npy   P(Fail) trên test
- config.threshold_path(model)  results/threshold_<model>.json   ngưỡng chọn trên oof (threshold.save_threshold)
- config.model_path(model)      saved_models/<model>.joblib      pipeline huấn luyện trên toàn bộ train (không commit)
--ablation ghi config.ablation_path(model); --time-block ghi config.robustness_path(model).
Định dạng JSON của ablation / robustness: xem README, mục "File đầu ra dùng chung".

Pipeline đầy đủ = build_preprocessor() + 1 bước model cuối (tên config.STEP_MODEL).
Mọi nơi (train, explain, dashboard) lấy pipeline từ build_pipeline(), không tự lắp ráp.

Ghi chú mở rộng: nếu sau này thêm chế độ stratified CV lặp lại để đối chiếu, ngưỡng và hyperparameter
phải chọn BÊN TRONG từng fold train (nested), nếu không kết quả đối chiếu bị lạc quan.
"""
import argparse
import json

import joblib
import numpy as np
from imblearn.pipeline import Pipeline
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

import config
from src.data_loader import load_data
from src.models import logistic, random_forest, xgboost_model
from src.preprocess import build_preprocessor
from src.split import time_split
from src.threshold import choose_threshold, save_threshold


def build_pipeline(name: str, y_train=None, k_features: int | None = config.K_FEATURES) -> Pipeline:
    """Ghép tiền xử lý + model thành 1 Pipeline chưa fit.

    - Các bước tiền xử lý giữ tên trong config.PREP_STEPS; bước cuối tên config.STEP_MODEL.
    - y_train chỉ cần cho "xgb": scale_pos_weight = n_pass / n_fail (tính trên train).
    - k_features=None bỏ bước chọn đặc trưng (dùng cho ablation).
    """
    prep = build_preprocessor(k_features)
    if name == "lr":
        model = logistic.build_model()
    elif name == "rf":
        model = random_forest.build_model()
    elif name == "xgb":
        spw = None
        if y_train is not None:
            spw = float((y_train == config.PASS).sum() / (y_train == config.FAIL).sum())
        model = xgboost_model.build_model(scale_pos_weight=spw)
    else:
        raise ValueError(f"model khong hop le: {name}. Chon mot trong {config.MODEL_NAMES}")
    return Pipeline(list(prep.steps) + [(config.STEP_MODEL, model)])


def _split_time_block(X, y, time_block: int):
    """Chia dữ liệu (đã sắp theo thời gian) thành config.N_TIME_BLOCKS khối liên tiếp bằng số dòng.
    Train = các khối 0..time_block-1 nối lại, test = khối time_block."""
    if not 1 <= time_block <= config.N_TIME_BLOCKS - 1:
        raise ValueError(f"time_block phai trong khoang 1..{config.N_TIME_BLOCKS - 1}, nhan duoc {time_block}")
    blocks = np.array_split(np.arange(len(X)), config.N_TIME_BLOCKS)
    train_idx = np.concatenate(blocks[:time_block])
    test_idx = blocks[time_block]
    return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]


def train_model(name: str, time_block=None, *, threshold_method: str = "f1",
                min_recall: float | None = None, max_flag_rate: float | None = None) -> dict:
    """Huấn luyện model `name`, tính OOF trên train, chọn ngưỡng trên OOF, dự đoán test.

    time_block=None: chia 80/20 theo thời gian, lưu 4 file chuẩn (oof, test, threshold, model).
    time_block=K   : robustness - train là khối 0..K-1, test là khối K; chỉ ghi config.robustness_path(name).
    """
    # Bước 1: load dữ liệu (đã sắp theo thời gian).
    X, y, timestamps = load_data()

    # Bước 2: chia train/test.
    if time_block is None:
        X_train, X_test, y_train, y_test = time_split(X, y, timestamps)
    else:
        X_train, X_test, y_train, y_test = _split_time_block(X, y, time_block)

    # Bước 3: xác suất out-of-fold trên train. Mỗi fold dựng pipeline MỚI, fit trên phần train của fold,
    # dự đoán phần validation và ghi vào đúng vị trí => mỗi mẫu train được dự đoán đúng 1 lần.
    skf = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED)
    oof_proba = np.zeros(len(y_train), dtype=float)
    for fold_train_idx, fold_val_idx in skf.split(X_train, y_train):
        X_fold_train, y_fold_train = X_train.iloc[fold_train_idx], y_train.iloc[fold_train_idx]
        fold_pipeline = build_pipeline(name, y_fold_train)
        fold_pipeline.fit(X_fold_train, y_fold_train)
        oof_proba[fold_val_idx] = fold_pipeline.predict_proba(X_train.iloc[fold_val_idx])[:, config.FAIL]

    # Bước 4: fit lại trên TOÀN BỘ train để dự đoán test.
    full_pipeline = build_pipeline(name, y_train)
    full_pipeline.fit(X_train, y_train)
    test_proba = full_pipeline.predict_proba(X_test)[:, config.FAIL]

    # Bước 5: chọn ngưỡng trên OOF của train (không bao giờ trên test). Chỉ truyền tham số không None.
    threshold_params = {k: v for k, v in {"min_recall": min_recall, "max_flag_rate": max_flag_rate}.items()
                        if v is not None}
    threshold = choose_threshold(y_train.to_numpy(), oof_proba, method=threshold_method, **threshold_params)

    if time_block is None:
        # Bước 6: lưu 4 file chuẩn. Đường dẫn lấy từ config.
        config.oof_path(name).parent.mkdir(parents=True, exist_ok=True)
        np.save(config.oof_path(name), oof_proba)
        config.test_path(name).parent.mkdir(parents=True, exist_ok=True)
        np.save(config.test_path(name), test_proba)
        save_threshold(name, threshold, threshold_method, **threshold_params)
        config.model_path(name).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(full_pipeline, config.model_path(name))
        print(f"[{name}] threshold={threshold:.2f} ({threshold_method}); da luu "
              f"{config.oof_path(name)}, {config.test_path(name)}, "
              f"{config.threshold_path(name)}, {config.model_path(name)}")
        return {"model": name, "threshold": threshold, "threshold_method": threshold_method, **threshold_params}

    # Bước 7: robustness - áp ngưỡng (chọn trên OOF của train khối này) lên test của khối.
    y_pred = (test_proba >= threshold).astype(int)
    row = {
        "scenario": f"time_block_{time_block}",
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, test_proba)) if y_test.nunique() > 1 else None,
    }
    # Đọc file cũ nếu có, thay dòng cùng scenario, giữ các dòng khác.
    path = config.robustness_path(name)
    rows = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    rows = [r for r in rows if r.get("scenario") != row["scenario"]] + [row]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"[{name}] {row} -> {path}")
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Huan luyen model SECOM")
    parser.add_argument("--model", required=True, choices=config.MODEL_NAMES)
    parser.add_argument("--threshold-method", default="f1", choices=["f1", "min_recall", "budget"])
    parser.add_argument("--min-recall", type=float, default=None)
    parser.add_argument("--max-flag-rate", type=float, default=None)
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--time-block", type=int, default=None)
    args = parser.parse_args()

    if args.threshold_method == "min_recall" and args.min_recall is None:
        parser.error("--threshold-method min_recall can --min-recall")
    if args.threshold_method == "budget" and args.max_flag_rate is None:
        parser.error("--threshold-method budget can --max-flag-rate")

    if args.ablation:
        module = {"lr": logistic, "rf": random_forest, "xgb": xgboost_model}[args.model]
        module.run_ablation()
        return

    train_model(args.model, time_block=args.time_block, threshold_method=args.threshold_method,
                min_recall=args.min_recall, max_flag_rate=args.max_flag_rate)


if __name__ == "__main__":
    main()
