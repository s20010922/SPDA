"""階段4：時間切分與序列切窗。
按時間先後切 train(前60%)/val(中20%)/test(後20%)，每根杆體各自切滑動窗口。
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

CLEANED_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cleaned"
)
MODEL_DATASET_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model_dataset"
)
os.makedirs(MODEL_DATASET_DIR, exist_ok=True)

FEATURE_COLS = ["w_total_detrended_z", "circuit_count", "light_circuit_w"]
# 原本用60(3小時)，實測發現SP-07/SP-08的20天持續性數值異常事件在window=120(6小時)下
# 偵測率明顯提升(60%左右->75%左右，5個種子都驗證過)。原因是這類異常本身變化緩慢，
# 窗口太短時異常期內部看起來平穩，容易被模型誤學成正常模式的一種；窗口拉長後能讓
# 模型看到更完整的偏移幅度和持續性。見對話紀錄window20/60/120三方對照實驗。
WINDOW_SIZE = 120
STRIDE = 1


def time_split(df: pd.DataFrame, train_ratio: float = 0.6, val_ratio: float = 0.2):
    """按reporttime先後切分成train/val/test，回傳切分時間點跟三個子集。"""
    times = df["reporttime"].sort_values().unique()
    n = len(times)
    train_end = times[int(n * train_ratio)]
    val_end = times[int(n * (train_ratio + val_ratio))]
    train = df[df["reporttime"] <= train_end]
    val = df[(df["reporttime"] > train_end) & (df["reporttime"] <= val_end)]
    test = df[df["reporttime"] > val_end]
    return train, val, test, train_end, val_end


def make_windows(df: pd.DataFrame, feature_cols: list, window_size: int = WINDOW_SIZE,
                  stride: int = STRIDE):
    """依polename分組各自切滑動窗口，回傳 (X, is_normal, polename, start_time)。
    is_normal 是整個窗口內所有時間點都normal才算True(嚴格條件，避免窗口裡混入異常時刻
    卻被當正常樣本訓練)。"""
    windows, is_normal_list, polename_list, start_time_list = [], [], [], []
    for pole, group in df.groupby("polename"):
        group = group.sort_values("reporttime")
        values = group[feature_cols].to_numpy()
        normal_flags = group["is_normal"].to_numpy()
        times = group["reporttime"].to_numpy()
        n = len(group)
        for start in range(0, n - window_size + 1, stride):
            windows.append(values[start:start + window_size])
            is_normal_list.append(bool(normal_flags[start:start + window_size].all()))
            polename_list.append(pole)
            start_time_list.append(times[start])
    if not windows:
        X = np.empty((0, window_size, len(feature_cols)))
    else:
        X = np.stack(windows)
    return (
        X,
        np.array(is_normal_list),
        np.array(polename_list),
        np.array(start_time_list),
    )


if __name__ == "__main__":
    df = pd.read_parquet(os.path.join(CLEANED_DIR, "merged_normalized.parquet"))
    df = df.dropna(subset=FEATURE_COLS)
    print(f"去除特徵缺值後剩 {len(df)} 筆")
    print(f"時間範圍：{df['reporttime'].min()} ~ {df['reporttime'].max()}")

    train, val, test, train_end, val_end = time_split(df)
    print(f"\ntrain: {len(train)} 筆（截至 {train_end}）")
    print(f"val:   {len(val)} 筆（截至 {val_end}）")
    print(f"test:  {len(test)} 筆")
    print("\n各集合 is_normal 比例：")
    for name, part in [("train", train), ("val", val), ("test", test)]:
        print(f"  {name}: {part['is_normal'].mean()*100:.1f}%")

    for name, part in [("train", train), ("val", val), ("test", test)]:
        X, is_normal, polename, start_time = make_windows(part, FEATURE_COLS)
        print(f"\n{name} windows shape: {X.shape}, is_normal比例: {is_normal.mean()*100:.1f}%")
        np.savez(
            os.path.join(MODEL_DATASET_DIR, f"windows_{name}.npz"),
            X=X, is_normal=is_normal, polename=polename, start_time=start_time,
        )
        print(f"已存至 {MODEL_DATASET_DIR}\\windows_{name}.npz")
