"""Load dữ liệu SECOM.

Cần làm:
- Đọc secom.data (sep=r"\\s+", header=None): kỳ vọng (1567, 590).
- Đọc secom_labels.data: cột 0 = nhãn (-1/1), cột 1 = timestamp (dd/mm/YYYY HH:MM:SS).
- Đổi nhãn sang 0 = Pass, 1 = Fail (config.LABEL_MAP).
- Giữ nguyên thứ tự theo thời gian (sắp xếp theo timestamp nếu cần).
"""
from collections import Counter

import pandas as pd

import config


def load_data() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Trả về (X, y, timestamps).

    X: DataFrame (n_mẫu, 590), cột đánh số 0..589
    y: Series 0/1 (1 = Fail)
    timestamps: Series datetime cùng index với X, đã sắp xếp tăng dần
    """
    # Bước 1: đọc đặc trưng. File ngăn cách bằng khoảng trắng nên dùng sep=r"\s+";
    # file không có dòng tiêu đề nên phải đặt header=None, nếu không pandas sẽ
    # lấy dòng dữ liệu đầu tiên làm tên cột và mất một mẫu.
    X = pd.read_csv(config.FEATURE_FILE, sep=r"\s+", header=None)

    # Bước 2: đọc file nhãn. Timestamp nằm trong dấu ngoặc kép nên pandas coi nó
    # là một trường duy nhất => chỉ có 2 cột: cột 0 = nhãn, cột 1 = chuỗi thời gian.
    labels_raw = pd.read_csv(config.LABEL_FILE, sep=r"\s+", header=None)

    # Bước 3: đổi chuỗi thời gian sang kiểu datetime. Ghi rõ format ngày/tháng/năm
    # vì nếu để pandas tự đoán, "07/08/2008" có thể bị hiểu nhầm thành tháng/ngày/năm.
    timestamps = pd.to_datetime(labels_raw[1], format="%d/%m/%Y %H:%M:%S")

    # Bước 4: đổi nhãn gốc (-1 = Pass, 1 = Fail) sang quy ước của nhóm (0 = Pass, 1 = Fail).
    # Việc đổi nhãn chỉ làm ở đây, các file khác dùng thẳng 0/1.
    y = labels_raw[0].map(config.LABEL_MAP)

    # Bước 5: sắp xếp theo thời gian. argsort cho biết thứ tự dòng khi xếp tăng dần;
    # phải áp CÙNG MỘT thứ tự cho X, y, timestamps để dòng thứ i của ba biến
    # vẫn là cùng một mẫu (nếu mỗi biến sắp một kiểu thì đặc trưng sẽ lệch nhãn).
    order = timestamps.argsort(kind="stable")
    X = X.iloc[order]
    y = y.iloc[order]
    timestamps = timestamps.iloc[order]

    # Sau khi sắp xếp, index bị xáo trộn nên đánh lại 0..n-1 để ba biến có index giống nhau.
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    timestamps = timestamps.reset_index(drop=True)

    # Bước 6: kiểm tra dữ liệu đúng như kỳ vọng, sai thì dừng ngay để khỏi chạy tiếp với dữ liệu hỏng.
    assert X.shape == (1567, 590), f"Kích thước X sai: {X.shape}, kỳ vọng (1567, 590)"
    assert len(X) == len(y) == len(timestamps), "X, y, timestamps không cùng số dòng"
    assert y.isin([0, 1]).all(), "y chứa giá trị khác 0 và 1 (có thể LABEL_MAP thiếu khóa)"
    assert timestamps.is_monotonic_increasing, "timestamps chưa tăng dần sau khi sắp xếp"

    # Bước 7: trả kết quả.
    return X, y, timestamps


def validate_structure() -> None:
    """Kiểm tra số trường mỗi dòng của secom.data (kỳ vọng đều = 590) và in kết quả."""
    # Đếm xem có bao nhiêu dòng có 590 trường, bao nhiêu dòng có số trường khác.
    # Dùng open() thay vì pandas để thấy được cả những dòng bị lỗi cấu trúc.
    field_counts = Counter()
    with open(config.FEATURE_FILE, "r") as file:
        for line in file:
            number_of_fields = len(line.split())
            field_counts[number_of_fields] += 1

    # In kết quả, kỳ vọng: Counter({590: 1567}).
    print(field_counts)

    # Nếu có dòng khác 590 trường thì cảnh báo rõ ràng.
    for number_of_fields in field_counts:
        if number_of_fields != 590:
            print(f"CẢNH BÁO: có {field_counts[number_of_fields]} dòng có {number_of_fields} trường (kỳ vọng 590)")
