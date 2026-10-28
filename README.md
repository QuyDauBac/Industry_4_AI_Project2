# Industry 4 AI - Dự đoán lỗi trong sản xuất bán dẫn (SECOM)

Đồ án cuối kỳ học phần Ứng dụng AI trong công nghiệp. Bài toán: phân loại nhị phân Pass/Fail trên bộ dữ liệu SECOM.

## Quy ước chung (chốt trong cuộc họp đầu, KHÔNG tự ý đổi)

- Nhãn: **0 = Pass, 1 = Fail** (đổi từ -1/1 một lần duy nhất trong `src/data_loader.py`).
- Chia 80/20 **theo thời gian** (cắt theo số dòng). Chế độ stratified chỉ để đối chiếu.
- Tuning: 5-fold stratified **bên trong train**. Seed = 42.
- Mọi bước tiền xử lý chỉ `fit` trên train (đặt trong Pipeline).
- Ngưỡng quyết định chọn trên xác suất out-of-fold của train, áp lên test đúng một lần.
- Không dùng hàm chi phí (SECOM không có dữ liệu chi phí).
- Kết quả dùng chung lưu ở `results/` (xem mục "File đầu ra dùng chung").
- Tên các bước Pipeline cố định trong `config.py` (`PREP_STEPS`, `STEP_MODEL`); tham số tuning có dạng
  `model__<tên>`. Pipeline đầy đủ luôn lấy từ `src.train.build_pipeline()`, không tự lắp ráp.
- Đường dẫn file kết quả luôn lấy từ hàm trong `config.py` (`oof_path`, `test_path`, `threshold_path`,
  `ablation_path`, `robustness_path`, `model_path`), không tự ghép chuỗi.

## Cài đặt

Cần Python 3.10 trở lên (code dùng cú pháp `int | None`); cả nhóm nên dùng cùng một phiên bản Python.

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Dữ liệu SECOM (công khai từ UCI, ~5,4 MB) đã có sẵn trong `data/secom/`, không cần tải thêm.

## Thứ tự chạy (chạy từ thư mục gốc của repo)

```bash
# 1. Huấn luyện + chọn ngưỡng cho từng model (ra 4 file, xem bảng bên dưới)
python -m src.train --model lr
python -m src.train --model rf
python -m src.train --model xgb

# 2. Ablation từng model (ra ablation_<model>.json)
python -m src.train --model lr --ablation

# 3. Robustness: train/test ở giai đoạn thời gian khác (ra robustness_<model>.json)
python -m src.train --model lr --time-block 1

# 4. Đánh giá 3 mức (đọc hết file trong results/) và demo
python -m src.evaluate
streamlit run app/dashboard.py
```

Đổi cách chọn ngưỡng: thêm `--threshold-method f1|min_recall|budget` (mặc định `f1`).
`--time-block K`: chia dữ liệu thành `N_TIME_BLOCKS` khối liên tiếp theo thời gian; test là khối K, train là các khối 0 đến K-1 (K từ 1 đến 4).

Notebook: thêm thư mục gốc vào `sys.path` (đã có sẵn ở cell đầu tiên) để `import config` và `import src...`.

## File đầu ra dùng chung

Dùng hàm trong `config.py` để lấy đường dẫn. Mỗi model có file riêng nên không đụng nhau khi merge.

| File | Hàm đường dẫn | Nội dung | Commit? |
|---|---|---|---|
| `results/oof_proba/<model>.npy` | `oof_path` | mảng (n_train,) P(Fail) out-of-fold trên train | có |
| `results/test_proba/<model>.npy` | `test_path` | mảng (n_test,) P(Fail) trên test | có |
| `results/threshold_<model>.json` | `threshold_path` | `{"model", "method", "threshold", ...}` (ghi bằng `threshold.save_threshold`, đọc bằng `threshold.load_threshold`) | có |
| `results/ablation_<model>.json` | `ablation_path` | danh sách `{"variant", "threshold", "precision", "recall", "f1", "roc_auc", "pr_auc"}` | có |
| `results/robustness_<model>.json` | `robustness_path` | danh sách `{"scenario", "precision", "recall", "f1", "roc_auc"}`; `scenario` vd `"time_block_1"`, `"noise_0.1"`, `"seed_7"` | có |
| `saved_models/<model>.joblib` | `model_path` | pipeline đã huấn luyện trên toàn bộ train | không (mỗi người tự train) |

## Phân công

| File | Phụ trách |
|---|---|
| `src/data_loader.py`, `src/split.py`, `src/preprocess.py`, `notebooks/01_eda_3B.ipynb` | Người 1 |
| `src/models/logistic.py`, `app/dashboard.py` | Người 1 |
| `src/models/random_forest.py`, `src/models/xgboost_model.py`, `src/train.py`, `src/threshold.py`, `src/explain.py` | Người 2 |
| `src/evaluate.py`, `src/failure_analysis.py`, `notebooks/02_results_comparison.ipynb` | Người 3 |
| `README.md`, `requirements.txt`, hoàn thiện code | Người 3 |

## Cấu trúc

```text
config.py            quy ước chung
data/secom/          dữ liệu thô (không sửa)
src/                 toàn bộ logic (notebook và dashboard chỉ import từ đây)
notebooks/           EDA, so sánh kết quả
app/                 dashboard Streamlit
results/             oof_proba, test_proba, threshold_*.json, ablation_*.json, robustness_*.json
saved_models/        file .joblib (không commit)
figures/             hình cho báo cáo (sơ đồ kiến trúc, biểu đồ)
report/              báo cáo, slide
```
