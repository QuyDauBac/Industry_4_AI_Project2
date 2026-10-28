"""Phân tích khi nào model sai (Ch.7): mẫu Fail bị bỏ sót, báo động giả,
và chuyện gì xảy ra khi điều kiện vận hành thay đổi (drift theo thời gian)."""
import numpy as np
import pandas as pd

import config


def _select(y_true, proba, timestamps, mask) -> pd.DataFrame:
    """Bảng các mẫu thỏa mask: sample_index (index gốc nếu y_true là Series), proba, timestamp (nếu có)."""
    index = y_true.index if isinstance(y_true, pd.Series) else pd.RangeIndex(len(y_true))
    table = pd.DataFrame({"sample_index": index, "proba": np.asarray(proba, dtype=float)})
    if timestamps is not None:
        # Truyền timestamps để xem mẫu sai tập trung ở giai đoạn nào (phân tích drift).
        table["timestamp"] = np.asarray(timestamps)
    return table[mask].reset_index(drop=True)


def missed_failures(y_true, proba, threshold: float, timestamps=None):
    """Mẫu Fail bị model bỏ sót: y_true = Fail nhưng proba < threshold."""
    mask = (np.asarray(y_true) == config.FAIL) & (np.asarray(proba) < threshold)
    return _select(y_true, proba, timestamps, mask)


def false_alarms(y_true, proba, threshold: float, timestamps=None):
    """Báo động giả: y_true = Pass nhưng proba >= threshold."""
    mask = (np.asarray(y_true) == config.PASS) & (np.asarray(proba) >= threshold)
    return _select(y_true, proba, timestamps, mask)
