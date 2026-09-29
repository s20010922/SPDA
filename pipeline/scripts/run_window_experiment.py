"""驗證實驗：window大小是否影響「持續性數值異常」的偵測率。
沿用現有正式版的清洗結果(merged_normalized.parquet，12迴路排除+去趨勢)，
只改window_size重新切窗，輸出到獨立目錄，不動現有的正式版windows。
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from step4_windowing import time_split, make_windows, FEATURE_COLS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEANED_DIR = os.path.join(ROOT, "cleaned")

if __name__ == "__main__":
    window_size = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    out_dir = os.path.join(ROOT, "experiments", f"window{window_size}_seed1")
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_parquet(os.path.join(CLEANED_DIR, "merged_normalized.parquet"))
    df = df.dropna(subset=FEATURE_COLS)
    print(f"[window={window_size}] 去除特徵缺值後剩 {len(df)} 筆")

    train, val, test, train_end, val_end = time_split(df)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        X, is_normal, polename, start_time = make_windows(part, FEATURE_COLS, window_size=window_size)
        print(f"[window={window_size}] {name} windows shape: {X.shape}, is_normal比例: {is_normal.mean()*100:.1f}%")
        np.savez(
            os.path.join(out_dir, f"windows_{name}.npz"),
            X=X, is_normal=is_normal, polename=polename, start_time=start_time,
        )
    print(f"[window={window_size}] 已存至 {out_dir}")
