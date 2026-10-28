"""Đánh giá 3 mức. Đọc kết quả đã lưu trong results/ (viết sớm, thử trên LR).
Đường dẫn lấy từ config: oof_path, test_path, threshold_path, ablation_path, robustness_path.

Model level   : Precision / Recall / F1, ROC-AUC, PR-AUC, detection rate (= recall Fail),
                bootstrap khoảng tin cậy. Báo cáo số lượng cụ thể (vd 12/20) cạnh tỉ lệ %.
System level  : false alarm, latency (đo thời gian predict_proba trên 1 mẫu), robustness (đổi giai đoạn
                thời gian, thêm nhiễu, nhiều seed), computational cost.
Industrial    : tỉ lệ lỗi chặn được, lỗi lọt qua, khối lượng cảnh báo trên 1.000 lô;
                so với 3 mốc (không kiểm tra, kiểm tra ngẫu nhiên cùng số lượng, kiểm tra 100%);
                bảng capture@k. Không quy ra tiền. Ghi rõ SECOM không đo được downtime, năng lượng.

Chạy: python -m src.evaluate
"""
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

import config
from src.data_loader import load_data
from src.split import time_split
from src.threshold import load_threshold

CAPTURE_KS = [0.05, 0.1, 0.2]   # các mức k (tỉ lệ mẫu được kiểm tra) cho bảng capture@k
LATENCY_REPEATS = 50            # số lần lặp predict_proba trên 1 mẫu để lấy latency trung bình
BOOTSTRAP_METRICS = ["precision", "recall", "f1", "roc_auc"]


def _load_test_labels() -> pd.Series:
    """y_test của cách chia chuẩn (80/20 theo thời gian), cùng thứ tự với test_proba đã lưu."""
    X, y, timestamps = load_data()
    _, _, _, y_test = time_split(X, y, timestamps)
    return y_test


def model_level(y_true, proba, threshold: float) -> dict:
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    y_pred = (proba >= threshold).astype(int)
    n_fail_total = int((y_true == config.FAIL).sum())
    n_fail_detected = int(((y_true == config.FAIL) & (y_pred == config.FAIL)).sum())
    one_class = np.unique(y_true).size < 2
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": None if one_class else float(roc_auc_score(y_true, proba)),
        "pr_auc": None if one_class else float(average_precision_score(y_true, proba)),
        # detection rate = recall của lớp Fail; kèm số lượng cụ thể (vd 12/20).
        "detection_rate": n_fail_detected / n_fail_total if n_fail_total else None,
        "n_fail_total": n_fail_total,
        "n_fail_detected": n_fail_detected,
    }


def bootstrap_ci(y_true, proba, threshold: float, metric: str, n_boot: int = 1000, seed: int = 42):
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    scorers = {
        "precision": lambda yt, p: precision_score(yt, (p >= threshold).astype(int), zero_division=0),
        "recall": lambda yt, p: recall_score(yt, (p >= threshold).astype(int), zero_division=0),
        "f1": lambda yt, p: f1_score(yt, (p >= threshold).astype(int), zero_division=0),
        "roc_auc": roc_auc_score,
    }
    if metric not in scorers:
        raise ValueError(f"metric khong hop le: {metric}. Chon mot trong {list(scorers)}")

    rng = np.random.default_rng(seed)
    n = len(y_true)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)   # lấy mẫu có hoàn lại
        # ROC-AUC không xác định nếu mẫu bootstrap chỉ có 1 lớp -> bỏ lần lấy mẫu đó.
        if metric == "roc_auc" and np.unique(y_true[idx]).size < 2:
            continue
        values.append(scorers[metric](y_true[idx], proba[idx]))
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def system_level(model_name: str) -> dict:
    model_file = config.model_path(model_name)
    if not model_file.exists():
        return {"model": model_name, "status": "chua co model, bo qua"}
    if not (config.test_path(model_name).exists() and config.threshold_path(model_name).exists()):
        return {"model": model_name, "status": "thieu test_proba hoac threshold, bo qua"}

    pipeline = joblib.load(model_file)
    X, y, timestamps = load_data()
    _, X_test, _, y_test = time_split(X, y, timestamps)

    # Latency: thời gian predict_proba trên 1 mẫu, lặp nhiều lần lấy trung bình (bỏ lần chạy đầu để "làm nóng").
    sample = X_test.iloc[[0]]
    pipeline.predict_proba(sample)
    durations = []
    for _ in range(LATENCY_REPEATS):
        start = time.perf_counter()
        pipeline.predict_proba(sample)
        durations.append(time.perf_counter() - start)

    # False alarm = 1 - precision trên test, áp ngưỡng đã lưu (chọn trên OOF của train).
    threshold = load_threshold(model_name)
    test_proba = np.load(config.test_path(model_name))
    y_pred = (test_proba >= threshold).astype(int)
    n_flagged = int(y_pred.sum())
    precision = precision_score(y_test, y_pred, zero_division=0)

    robustness = []
    if config.robustness_path(model_name).exists():
        robustness = json.loads(config.robustness_path(model_name).read_text(encoding="utf-8"))

    return {
        "model": model_name,
        "status": "ok",
        "threshold": threshold,
        "n_flagged": n_flagged,
        # Không gắn cờ mẫu nào thì precision không xác định -> false_alarm để None.
        "false_alarm": float(1 - precision) if n_flagged else None,
        "latency_ms_mean": float(np.mean(durations) * 1000),
        "latency_ms_std": float(np.std(durations) * 1000),
        # Computational cost: kích thước pipeline đã lưu.
        "model_size_kb": model_file.stat().st_size / 1024,
        "robustness": robustness,
    }


def industrial_level(y_true, proba, threshold: float, per: int = 1000) -> dict:
    """So sánh model với 3 mốc kiểm tra. Mọi số liệu tính trên tập được truyền vào (thường là test).

    Không quy đổi ra tiền (không dùng hàm chi phí - SECOM không có dữ liệu chi phí).
    SECOM không đo được downtime và năng lượng nên KHÔNG đưa 2 chỉ số này vào.
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    n_samples = len(y_true)
    n_fail = int((y_true == config.FAIL).sum())
    flagged = proba >= threshold
    flag_rate = float(flagged.mean())
    n_detected = int((flagged & (y_true == config.FAIL)).sum())

    def row(rate, detected):
        return {
            "flag_rate": rate,
            "ti_le_loi_chan_duoc": detected / n_fail if n_fail else None,
            "n_loi_chan_duoc": detected,
            "n_loi_lot_qua": n_fail - detected,
            "canh_bao_tren_per_lo": rate * per,
        }

    return {
        "n_samples": n_samples,
        "n_fail": n_fail,
        "per": per,
        "model": row(flag_rate, n_detected),
        # Không kiểm tra: không gắn cờ lô nào, không chặn được lỗi nào.
        "khong_kiem_tra": row(0.0, 0),
        # Kiểm tra ngẫu nhiên cùng flag_rate: kỳ vọng chặn được flag_rate * n_fail lỗi.
        "kiem_tra_ngau_nhien": row(flag_rate, flag_rate * n_fail),
        # Kiểm tra 100%: chặn hết lỗi nhưng khối lượng kiểm tra = toàn bộ số lô.
        "kiem_tra_100%": row(1.0, n_fail),
    }


def capture_at_k(y_true, proba, ks) -> "pd.DataFrame":
    """Kiểm tra top k (tỉ lệ) mẫu có proba cao nhất: bắt được bao nhiêu % tổng số Fail."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    order = np.argsort(-proba, kind="stable")
    n_fail_total = int((y_true == config.FAIL).sum())
    rows = []
    for k in ks:
        n_flagged = int(round(k * len(y_true)))
        n_fail_captured = int((y_true[order[:n_flagged]] == config.FAIL).sum())
        rows.append({
            "k": k,
            "n_flagged": n_flagged,
            "n_fail_captured": n_fail_captured,
            "capture_rate": n_fail_captured / n_fail_total if n_fail_total else None,
        })
    return pd.DataFrame(rows, columns=["k", "n_flagged", "n_fail_captured", "capture_rate"])


def available_models() -> list[str]:
    """Các model trong config.MODEL_NAMES đã có đủ oof / test / threshold."""
    ready = []
    for name in config.MODEL_NAMES:
        missing = [p for p in (config.oof_path(name), config.test_path(name), config.threshold_path(name))
                   if not p.exists()]
        if missing:
            print(f"[{name}] bo qua - thieu file: {', '.join(str(p) for p in missing)}")
        else:
            ready.append(name)
    return ready


def main() -> None:
    models = available_models()
    if not models:
        print("Chua co model nao du file ket qua. Chay python -m src.train --model <ten> truoc.")
        return

    y_test = _load_test_labels().to_numpy()
    for name in models:
        proba = np.load(config.test_path(name))
        if len(proba) != len(y_test):
            print(f"[{name}] bo qua - test_proba co {len(proba)} mau, y_test co {len(y_test)} mau")
            continue
        threshold = load_threshold(name)
        print(f"\n===== {name} (threshold={threshold:.2f}) =====")

        print("-- Model level (test) --")
        ml = model_level(y_test, proba, threshold)
        for key, value in ml.items():
            print(f"  {key}: {value}")
        print(f"  detection: {ml['n_fail_detected']}/{ml['n_fail_total']}")
        for metric in BOOTSTRAP_METRICS:
            low, high = bootstrap_ci(y_test, proba, threshold, metric, seed=config.SEED)
            print(f"  {metric} 95% CI: [{low:.3f}, {high:.3f}]")

        print("-- System level --")
        for key, value in system_level(name).items():
            print(f"  {key}: {value}")

        print("-- Industrial level (test) --")
        il = industrial_level(y_test, proba, threshold)
        table = pd.DataFrame({k: il[k] for k in ["model", "khong_kiem_tra", "kiem_tra_ngau_nhien", "kiem_tra_100%"]}).T
        print(f"  n_samples={il['n_samples']}, n_fail={il['n_fail']}, per={il['per']}")
        print(table.to_string())
        print("-- capture@k (test) --")
        print(capture_at_k(y_test, proba, CAPTURE_KS).to_string(index=False))


if __name__ == "__main__":
    main()
