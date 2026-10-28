"""Quy ước chung của cả nhóm. Mọi file khác import từ đây, không hard-code lại."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Đường dẫn
DATA_DIR = ROOT / "data" / "secom"
FEATURE_FILE = DATA_DIR / "secom.data"
LABEL_FILE = DATA_DIR / "secom_labels.data"
RESULTS_DIR = ROOT / "results"
OOF_DIR = RESULTS_DIR / "oof_proba"
TEST_DIR = RESULTS_DIR / "test_proba"
MODELS_DIR = ROOT / "saved_models"
FIGURES_DIR = ROOT / "figures"

# Nhãn: file gốc dùng -1 (Pass) và 1 (Fail); trong code dùng 0 = Pass, 1 = Fail
LABEL_MAP = {-1: 0, 1: 1}
PASS, FAIL = 0, 1

# Chia dữ liệu và CV
SEED = 42
TEST_SIZE = 0.20          # chia theo thời gian, cắt theo số dòng
N_FOLDS = 5               # stratified, chỉ trong train
N_TIME_BLOCKS = 5  # số khối thời gian cho --time-block

# Tiền xử lý
MISSING_THRESHOLD = 0.70  # bỏ cột thiếu > 70% (quy ước của bài thực hành, không phải chuẩn công nghiệp)
K_FEATURES = 20          # số đặc trưng chọn bằng F-test

MODEL_NAMES = ["lr", "rf", "xgb"]

# Tên các bước trong Pipeline (cố định, để tuning / explain / dashboard dùng chung)
# Pipeline đầy đủ = các bước tiền xử lý + 1 bước model cuối cùng; luôn lấy từ src.train.build_pipeline().
PREP_STEPS = ["missing", "impute", "variance", "outlier", "scale", "select"]  # "outlier" chỉ có nếu nhóm dùng
STEP_MODEL = "model"      # tham số tuning có dạng "model__<tên>", vd "model__C"
ADD_MISSING_INDICATOR = False  # feature engineering: thêm cột đánh dấu giá trị thiếu (False = không tạo đặc trưng mới)


# Đường dẫn file kết quả dùng chung: LUÔN dùng các hàm này, không tự ghép chuỗi.
def oof_path(model: str) -> Path:
    return OOF_DIR / f"{model}.npy"


def test_path(model: str) -> Path:
    return TEST_DIR / f"{model}.npy"


def threshold_path(model: str) -> Path:
    return RESULTS_DIR / f"threshold_{model}.json"


def ablation_path(model: str) -> Path:
    return RESULTS_DIR / f"ablation_{model}.json"


def robustness_path(model: str) -> Path:
    return RESULTS_DIR / f"robustness_{model}.json"


def model_path(model: str) -> Path:
    return MODELS_DIR / f"{model}.joblib"
