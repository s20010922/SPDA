"""驗證實驗：跳過7天移動平均去趨勢，直接用原始w_total做正規化/切窗，
輸出到獨立目錄 pipeline/experiments/notrend_seed1，跟現有12迴路+有去趨勢版本比較。
目的：確認去趨勢處理在新pipeline上是否真的有幫助，而不是照搬舊專案的結論。
排除迴路清單維持跟目前正式版一致(12個)，只把「是否去趨勢」這個變因單獨切開。
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from step2_feature_engineering import (
    aggregate_pole_features, compute_delta, merge_with_light, KNOWN_BAD_CIRCUITS,
)
from step3_normalize import flag_outliers_per_pole, normalize_per_pole
from step4_windowing import time_split, make_windows

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEANED_DIR = os.path.join(ROOT, "cleaned")
OUT_DIR = os.path.join(ROOT, "experiments", "notrend_seed1")
os.makedirs(OUT_DIR, exist_ok=True)

# 沒有去趨勢，直接對 w_total 做正規化，特徵欄位對應改成 w_total_z
FEATURE_COLS = ["w_total_z", "circuit_count", "light_circuit_w"]

if __name__ == "__main__":
    meter_raw = pd.read_parquet(os.path.join(CLEANED_DIR, "meter_raw.parquet"))
    light_wide = pd.read_parquet(os.path.join(CLEANED_DIR, "light_wide.parquet"))

    features = aggregate_pole_features(meter_raw, exclude_circuits=KNOWN_BAD_CIRCUITS)
    features = compute_delta(features)
    # 刻意跳過 compute_detrended，直接用原始 w_total
    merged = merge_with_light(features, light_wide)
    print(f"[notrend] 合併表共 {len(merged)} 筆")

    merged = flag_outliers_per_pole(merged, value_col="w_total")
    merged = normalize_per_pole(merged, value_col="w_total")
    merged["circuit_count"] = merged["circuit_count"].fillna(0)
    merged["light_circuit_w"] = merged["light_circuit_w"].fillna(0)

    df = merged.dropna(subset=FEATURE_COLS)
    print(f"[notrend] 去除特徵缺值後剩 {len(df)} 筆")

    train, val, test, train_end, val_end = time_split(df)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        X, is_normal, polename, start_time = make_windows(part, FEATURE_COLS)
        print(f"[notrend] {name} windows shape: {X.shape}, is_normal比例: {is_normal.mean()*100:.1f}%")
        np.savez(
            os.path.join(OUT_DIR, f"windows_{name}.npz"),
            X=X, is_normal=is_normal, polename=polename, start_time=start_time,
        )
    print(f"[notrend] 已存至 {OUT_DIR}")
