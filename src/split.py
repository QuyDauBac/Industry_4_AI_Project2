"""Chia dữ liệu.

Quy ước: 80/20 theo thời gian (cắt theo số dòng), in số mẫu / số lỗi / tỉ lệ lỗi từng phần.
Có thêm chế độ stratified để đối chiếu. Không cân bằng lại tập test.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

import config


def time_split(X: pd.DataFrame, y: pd.Series, timestamps: pd.Series, test_size: float = config.TEST_SIZE):
    """Sắp xếp theo timestamp, cắt theo số dòng. Trả về X_train, X_test, y_train, y_test."""
    # Bước 1: lấy thứ tự sắp xếp theo thời gian rồi áp CÙNG MỘT thứ tự cho X, y, timestamps.
    # Dữ liệu đã sắp sẵn nhưng vẫn làm lại để an toàn; dùng chung một thứ tự để mẫu không bị lệch nhãn.
    order = timestamps.argsort(kind="stable")
    X = X.iloc[order]
    y = y.iloc[order]
    timestamps = timestamps.iloc[order]

    # Bước 2: tính điểm cắt theo số dòng. Dùng một điểm cắt duy nhất cho cả X và y
    # để dòng nào của X nằm ở train thì nhãn của dòng đó cũng nằm ở train.
    cut = int(len(X) * (1 - test_size))

    # Bước 3: cắt bằng iloc, không xáo trộn. Train là các lô cũ, test là các lô mới hơn,
    # giống nhà máy thật: học từ quá khứ để dự đoán các lô sau, không được nhìn thấy tương lai.
    # Không reset_index để vẫn truy ngược được mẫu nằm ở dòng nào của dữ liệu gốc.
    X_train = X.iloc[:cut]
    X_test = X.iloc[cut:]
    y_train = y.iloc[:cut]
    y_test = y.iloc[cut:]
    timestamps_train = timestamps.iloc[:cut]
    timestamps_test = timestamps.iloc[cut:]

    # Bước 4: kiểm tra việc chia không làm mất hay lệch mẫu, và train luôn nằm trước test về thời gian.
    assert len(X_train) + len(X_test) == len(X), "Số mẫu train + test không bằng tổng số mẫu"
    assert len(X_train) == len(y_train), "X_train và y_train không cùng số dòng"
    assert len(X_test) == len(y_test), "X_test và y_test không cùng số dòng"
    assert timestamps_train.max() <= timestamps_test.min(), "Thời gian train và test bị chồng lấn"

    # Bước 5: trả kết quả. Không cân bằng lại tập test vì test phải phản ánh tỉ lệ lỗi thực tế.
    return X_train, X_test, y_train, y_test


def stratified_split(X: pd.DataFrame, y: pd.Series, test_size: float = config.TEST_SIZE, seed: int = config.SEED):
    """Chia ngẫu nhiên có stratify - chỉ dùng để đối chiếu với chia theo thời gian.

    Đây KHÔNG phải cách chia chính của nhóm. Trả về cùng thứ tự với time_split:
    X_train, X_test, y_train, y_test.
    """
    # Chia ngẫu nhiên nhưng giữ tỉ lệ Fail ở train và test gần bằng nhau (stratify=y).
    # random_state=seed để mỗi lần chạy cho cùng một kết quả.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=seed,
    )
    return X_train, X_test, y_train, y_test


def describe_split(y_train: pd.Series, y_test: pd.Series) -> pd.DataFrame:
    """Bảng số mẫu / số Fail / tỉ lệ Fail của train và test (in ra và đưa vào báo cáo)."""
    # Đếm số mẫu của từng phần.
    n_train = len(y_train)
    n_test = len(y_test)

    # Fail được mã hóa là 1 nên tổng của y chính là số lô Fail.
    n_fail_train = int(y_train.sum())
    n_fail_test = int(y_test.sum())

    # Tỉ lệ Fail = số Fail / số mẫu, làm tròn 4 chữ số.
    fail_rate_train = round(n_fail_train / n_train, 4)
    fail_rate_test = round(n_fail_test / n_test, 4)

    # Gom thành bảng 2 hàng (train, test) x 3 cột rồi in ra.
    table = pd.DataFrame(
        {
            "n_samples": [n_train, n_test],
            "n_fail": [n_fail_train, n_fail_test],
            "fail_rate": [fail_rate_train, fail_rate_test],
        },
        index=["train", "test"],
    )
    print(table)
    return table
