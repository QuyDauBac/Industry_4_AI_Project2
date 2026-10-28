"""Chọn ngưỡng quyết định, dùng chung cho cả 3 model.

Quy tắc: chọn trên xác suất out-of-fold của tập TRAIN, không bao giờ chọn trên test.
Tiêu chí: tối đa F1, hoặc recall tối thiểu cho trước (không dùng hàm chi phí).
Có thể chọn theo ngân sách kiểm tra (số mẫu được đánh dấu tối đa).
"""
import json

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

import config

# Dải ngưỡng quét chung cho cả 3 method (tăng dần: chỉ số nhỏ = ngưỡng thấp).
# Làm tròn 2 chữ số để bỏ sai số dấu phẩy động của linspace (vd 0.15000000000000002 -> 0.15).
THRESHOLD_GRID = np.round(np.linspace(0.01, 0.99, 99), 2)


def choose_threshold(y_true: np.ndarray, oof_proba: np.ndarray, method: str = "f1",
                     min_recall: float | None = None, max_flag_rate: float | None = None) -> float:
    """method: "f1" | "min_recall" | "budget". Trả về ngưỡng."""
    y_true = np.asarray(y_true)
    oof_proba = np.asarray(oof_proba)
    grid = THRESHOLD_GRID

    if method == "f1":
        # F1 tại từng ngưỡng; np.argmax trả về vị trí ĐẦU TIÊN của giá trị lớn nhất,
        # vì grid tăng dần nên khi hòa sẽ lấy ngưỡng thấp nhất (bắt được nhiều Fail hơn).
        scores = np.array([f1_score(y_true, (oof_proba >= t).astype(int), zero_division=0) for t in grid])
        return float(grid[np.argmax(scores)])

    if method == "min_recall":
        if min_recall is None:
            raise ValueError('method="min_recall" can truyen min_recall')
        recalls = np.array([recall_score(y_true, (oof_proba >= t).astype(int), zero_division=0) for t in grid])
        precisions = np.array([precision_score(y_true, (oof_proba >= t).astype(int), zero_division=0) for t in grid])
        ok = recalls >= min_recall
        if ok.any():
            # Trong các ngưỡng đạt recall tối thiểu, lấy precision cao nhất; hòa thì lấy ngưỡng thấp nhất.
            candidates = np.flatnonzero(ok)
            return float(grid[candidates[np.argmax(precisions[candidates])]])
        # Không ngưỡng nào đạt: lấy ngưỡng có recall cao nhất (hòa thì ngưỡng thấp nhất), không raise.
        return float(grid[np.argmax(recalls)])

    if method == "budget":
        if max_flag_rate is None:
            raise ValueError('method="budget" can truyen max_flag_rate')
        flag_rates = np.array([(oof_proba >= t).mean() for t in grid])
        distance = np.abs(flag_rates - max_flag_rate)
        # Hòa thì lấy ngưỡng CAO hơn (gắn cờ ít hơn): tìm vị trí cuối cùng đạt khoảng cách nhỏ nhất.
        best = np.flatnonzero(distance == distance.min())[-1]
        return float(grid[best])

    raise ValueError(f'method khong hop le: {method}. Chon "f1", "min_recall" hoac "budget"')


def save_threshold(model_name: str, threshold: float, method: str, **params) -> None:
    """Lưu ngưỡng vào config.threshold_path(model_name) để evaluate.py và dashboard dùng lại.

    Định dạng: {"model": ..., "method": "f1" | "min_recall" | "budget", "threshold": 0.37, ...tham số}
    """
    data = {"model": model_name, "method": method, "threshold": float(threshold), **params}
    path = config.threshold_path(model_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_threshold(model_name: str) -> float:
    """Đọc ngưỡng đã lưu của model."""
    data = json.loads(config.threshold_path(model_name).read_text(encoding="utf-8"))
    return float(data["threshold"])
