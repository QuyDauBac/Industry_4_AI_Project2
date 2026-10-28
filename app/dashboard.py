"""Dashboard Streamlit (mô phỏng chạy trên máy cá nhân: Data -> AI -> Decision).

Chạy: streamlit run app/dashboard.py

Hiển thị: trạng thái lô/mẫu hiện tại (SECOM không có mã máy), risk score P(Fail), cảnh báo,
khuyến nghị hành động, nút kỹ sư xác nhận (human-in-the-loop).

Đầu vào: pipeline đã huấn luyện của model tốt nhất (config.model_path), ngưỡng của model đó
(threshold.load_threshold), và các mẫu của tập test để "phát lại" như dữ liệu đang đến.
"""
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from src.data_loader import load_data  # noqa: E402
from src.split import time_split  # noqa: E402
from src.threshold import load_threshold  # noqa: E402


def predict_risk(pipeline, sample) -> float:
    """Trả về P(Fail) của một mẫu (DataFrame 1 dòng, đủ 590 cột gốc)."""
    return float(pipeline.predict_proba(sample)[0, config.FAIL])


def recommend_action(risk: float, threshold: float) -> str:
    """Từ risk score và ngưỡng đã chọn trả về khuyến nghị hành động
    (ví dụ "Giữ lô, kiểm tra thêm" / "Cho qua"). Kỹ sư xác nhận hoặc bác bỏ khuyến nghị này."""
    margin = risk - threshold
    if margin >= 0.15:
        return "Giữ lô, kiểm tra kỹ (risk cao hơn hẳn ngưỡng)"
    if margin >= 0:
        return "Giữ lô, kiểm tra thêm"
    if margin >= -0.10:
        return "Cho qua, nhưng ghi nhận để theo dõi (risk gần ngưỡng)"
    return "Cho qua"


def main() -> None:
    """Dựng giao diện Streamlit."""
    st.set_page_config(page_title="SECOM - Cảnh báo lỗi", layout="centered")
    st.title("SECOM - Cảnh báo lô lỗi (Data → AI → Decision)")

    model_name = "lr"
    if not config.model_path(model_name).exists():
        st.error(f"Chưa có model {config.model_path(model_name)}. "
                 f"Hãy chạy `python -m src.train --model {model_name}` trước.")
        return

    pipeline = joblib.load(config.model_path(model_name))
    threshold = load_threshold(model_name)
    X, y, timestamps = load_data()
    _, X_test, _, y_test = time_split(X, y, timestamps)

    if "sample_idx" not in st.session_state:
        st.session_state.sample_idx = 0
    if "log" not in st.session_state:
        st.session_state.log = []

    idx = st.session_state.sample_idx % len(X_test)
    sample = X_test.iloc[[idx]]
    risk = predict_risk(pipeline, sample)
    action = recommend_action(risk, threshold)

    st.subheader(f"Lô {idx + 1}/{len(X_test)} (model: {model_name})")
    col_risk, col_threshold = st.columns(2)
    col_risk.metric("Risk score P(Fail)", f"{risk:.3f}")
    col_threshold.metric("Ngưỡng", f"{threshold:.2f}")

    if risk >= threshold:
        st.error(f"CẢNH BÁO: risk vượt ngưỡng. Khuyến nghị: {action}")
    else:
        st.success(f"Bình thường. Khuyến nghị: {action}")

    actual = "Fail" if int(y_test.iloc[idx]) == config.FAIL else "Pass"
    st.caption(f"Nhãn thực tế: {actual} (chỉ để đối chiếu khi demo, không dùng để quyết định thật)")

    col_next, col_accept, col_reject = st.columns(3)
    if col_next.button("Lô tiếp theo"):
        st.session_state.sample_idx += 1
        st.rerun()
    for column, label, decision in [(col_accept, "Kỹ sư xác nhận khuyến nghị", "xác nhận"),
                                    (col_reject, "Kỹ sư bác bỏ khuyến nghị", "bác bỏ")]:
        if column.button(label):
            st.session_state.log.append({
                "lo": idx + 1,
                "risk": round(risk, 4),
                "nguong": threshold,
                "khuyen_nghi": action,
                "ky_su": decision,
            })
            st.toast(f"Đã ghi nhận: kỹ sư {decision} khuyến nghị cho lô {idx + 1}")

    if st.session_state.log:
        st.subheader("Lịch sử xác nhận của kỹ sư")
        st.dataframe(pd.DataFrame(st.session_state.log))


if __name__ == "__main__":
    main()
