"""Pipeline tiền xử lý. Mọi bước chỉ fit trên train.

Các bước dự kiến (tên bước lấy từ config.PREP_STEPS; giải thích lý do từng bước trong báo cáo Ch.4):
1. "missing"  : MissingRateSelector - bỏ cột thiếu > config.MISSING_THRESHOLD
2. "impute"   : SimpleImputer(median, add_indicator=config.ADD_MISSING_INDICATOR)
                Đây cũng là bước FEATURE ENGINEERING: thêm cột đánh dấu "giá trị này bị thiếu"
                (thiếu dữ liệu cảm biến có thể là tín hiệu). Nếu đặt ADD_MISSING_INDICATOR=False thì
                báo cáo phải ghi rõ lý do không tạo đặc trưng mới (590 cột đã ẩn danh).
3. "variance" : VarianceThreshold - bỏ cột hằng số
4. "outlier"  : (tùy chọn) xử lý ngoại lệ; không dùng thì bỏ bước này
5. "scale"    : StandardScaler
6. "select"   : SelectKBest(f_classif, k=config.K_FEATURES)

Imbalance: xử lý bằng class_weight / scale_pos_weight trong từng model. Nếu vẫn dùng resampling
(imblearn) thì thêm thành MỘT BƯỚC RIÊNG ở cuối danh sách bước, ngay trước bước model - không lồng
Pipeline trong Pipeline (src.train.build_pipeline ghép bằng cách nối các bước lại).
Chỉ số cột sau khi lọc/thêm phải truy ngược được về cột gốc: MissingRateSelector cần có get_support().

LƯU Ý: lớp MissingRateSelector được định nghĩa ở DUY NHẤT file này. File .joblib ghi nhớ
đường dẫn module của lớp, nên không đổi tên/chuyển file sau khi đã lưu pipeline.
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config


class MissingRateSelector(BaseEstimator, TransformerMixin):
    """Bỏ các cột có tỉ lệ thiếu vượt ngưỡng (tham khảo secom_preprocessing.py của thầy)."""

    def __init__(self, threshold: float = config.MISSING_THRESHOLD):
        # Chỉ gán tham số, không làm gì khác: đây là quy ước của scikit-learn,
        # nếu thêm logic ở đây thì clone() và GridSearchCV sẽ bị lỗi.
        self.threshold = threshold

    def fit(self, X, y=None):
        """Tính tỉ lệ thiếu của từng cột trên dữ liệu train và lưu mặt nạ các cột được giữ."""
        # Đổi sang mảng numpy kiểu float để DataFrame hay mảng đều xử lý giống nhau.
        X_array = np.asarray(X, dtype=float)

        # Tỉ lệ thiếu của mỗi cột = số ô NaN / số dòng (mean của mảng True/False theo từng cột).
        missing_rate = np.isnan(X_array).mean(axis=0)

        # Cột thiếu quá nhiều thì gần như không mang thông tin, và điền bù sẽ chủ yếu là
        # "bịa" số liệu. Giữ cột có tỉ lệ thiếu <= ngưỡng, bỏ cột có tỉ lệ thiếu LỚN HƠN ngưỡng.
        # Mặt nạ này chỉ tính từ train; test sẽ dùng lại đúng mặt nạ này.
        self.support_ = missing_rate <= self.threshold
        return self

    def transform(self, X):
        """Giữ lại các cột đã chọn lúc fit."""
        # Chưa fit thì chưa có mặt nạ, báo lỗi rõ ràng thay vì để Python báo lỗi khó hiểu.
        if not hasattr(self, "support_"):
            raise ValueError("MissingRateSelector chưa được fit. Hãy gọi fit() trên tập train trước khi transform().")

        # Chỉ lấy các cột có mặt nạ True. Dùng mặt nạ của train, không tính lại trên test.
        X_array = np.asarray(X, dtype=float)
        return X_array[:, self.support_]

    def get_support(self):
        """Mask boolean các cột được giữ (explain.py dùng để truy ngược về cột gốc)."""
        return self.support_


def build_preprocessor(k_features: int | None = config.K_FEATURES) -> Pipeline:
    """Trả về Pipeline tiền xử lý chưa fit (các bước đặt tên theo config.PREP_STEPS).

    Chỉ gồm tiền xử lý, KHÔNG gồm model: model được nối thêm ở src.train.build_pipeline().
    k_features=None để bỏ bước "select" (ablation).
    """
    # Mọi bước đều nằm trong một Pipeline: khi fit chỉ học từ train (ngưỡng, median, độ lệch chuẩn,
    # điểm F...), còn với test chỉ gọi transform. Nếu fit cả trên test thì thông tin của test
    # "rò rỉ" vào quá trình huấn luyện và điểm đánh giá sẽ đẹp hơn thực tế.
    steps = []

    # Bước 1: bỏ các cột thiếu quá ngưỡng config.MISSING_THRESHOLD (quy ước của bài thực hành, không phải chuẩn công nghiệp).
    steps.append(("missing", MissingRateSelector()))

    # Bước 2: điền giá trị thiếu bằng median (trung vị). Dùng median thay vì mean vì dữ liệu cảm biến
    # có nhiều giá trị lệch/cực trị; mean bị kéo lệch theo các giá trị đó còn median thì ổn định hơn.
    # add_indicator=True thêm các cột 0/1 đánh dấu "ô này vốn bị thiếu", vì việc cảm biến không
    # ghi được số liệu có thể chính là dấu hiệu của lô lỗi (đây là bước feature engineering).
    steps.append(("impute", SimpleImputer(strategy="median", add_indicator=config.ADD_MISSING_INDICATOR)))

    # Bước 3: bỏ cột hằng số (phương sai = 0, ngưỡng mặc định 0). Cột không đổi thì không giúp
    # phân biệt Pass/Fail và còn gây chia cho 0 khi chuẩn hóa.
    steps.append(("variance", VarianceThreshold()))

    # Không thêm bước "outlier": giá trị cực trị của cảm biến có thể chính là tín hiệu lỗi
    # nên không tự động loại bỏ, và các phương pháp dựa trên cây ít nhạy với ngoại lệ.

    # Bước 4: chuẩn hóa về trung bình 0, độ lệch chuẩn 1. Các cảm biến có thang đo rất khác nhau;
    # Logistic Regression nhạy với thang đo, và việc chọn đặc trưng nên so sánh các cột trên cùng một thang.
    steps.append(("scale", StandardScaler()))

    # Bước 5: chọn K đặc trưng có điểm F-test (f_classif) cao nhất, tức là cột có giá trị trung bình
    # khác nhau nhiều nhất giữa lớp Pass và Fail. F-test nhanh, dễ giải thích, và giảm số cột
    # (590 cột nhưng chỉ có 1253 mẫu train, ít Fail) để hạn chế overfit.
    # k_features=None thì bỏ bước này (dùng cho ablation).
    if k_features is not None:
        steps.append(("select", SelectKBest(score_func=f_classif, k=k_features)))

    return Pipeline(steps)
