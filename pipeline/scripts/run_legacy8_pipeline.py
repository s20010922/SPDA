"""驗證實驗：用舊的8迴路排除清單（不含SP-07網路設備），重跑step2~4，
輸出到獨立目錄 pipeline/experiments/legacy8_seed1，不動現有12迴路版本的產出。
目的：把「多排除4個迴路」跟「訓練跑滿80輪」這兩個效果拆開來看。
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

from step2_feature_engineering import (
    aggregate_pole_features, compute_delta, compute_detrended, merge_with_light,
)
from step3_normalize import flag_outliers_per_pole, normalize_per_pole
from step4_windowing import time_split, make_windows, FEATURE_COLS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEANED_DIR = os.path.join(ROOT, "cleaned")
OUT_DIR = os.path.join(ROOT, "experiments", "legacy8_seed1")
os.makedirs(OUT_DIR, exist_ok=True)

# 舊版排除清單：只有原始8個迴路，不含SP-07的網路設備(11324)
LEGACY_8_CIRCUITS = {
    11336, 11337, 11339,  # SCCP-SP-02: MP、數位看板、攝影機
    11321, 11322,          # SCCP-SP-07: MP、數位看板（不含網路設備11324）
    11331, 11332, 11334,  # SCCP-SP-08: MP、數位看板、攝影機
    11326, 11327, 11329,  # SCCP-SP-09: MP、數位看板、攝影機
}

if __name__ == "__main__":
    meter_raw = pd.read_parquet(os.path.join(CLEANED_DIR, "meter_raw.parquet"))
    light_wide = pd.read_parquet(os.path.join(CLEANED_DIR, "light_wide.parquet"))

    n_excluded = meter_raw["circuitid"].isin(LEGACY_8_CIRCUITS).sum()
    print(f"[legacy8] 排除迴路 {sorted(LEGACY_8_CIRCUITS)}，共 {n_excluded} 筆讀數不計入")

    features = aggregate_pole_features(meter_raw, exclude_circuits=LEGACY_8_CIRCUITS)
    features = compute_delta(features)
    features = compute_detrended(features)
    merged = merge_with_light(features, light_wide)
    print(f"[legacy8] 合併表共 {len(merged)} 筆")

    merged = flag_outliers_per_pole(merged, value_col="w_total_detrended")
    merged = normalize_per_pole(merged, value_col="w_total_detrended")
    merged["circuit_count"] = merged["circuit_count"].fillna(0)
    merged["light_circuit_w"] = merged["light_circuit_w"].fillna(0)

    df = merged.dropna(subset=FEATURE_COLS)
    print(f"[legacy8] 去除特徵缺值後剩 {len(df)} 筆")

    train, val, test, train_end, val_end = time_split(df)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        import numpy as np
        X, is_normal, polename, start_time = make_windows(part, FEATURE_COLS)
        print(f"[legacy8] {name} windows shape: {X.shape}, is_normal比例: {is_normal.mean()*100:.1f}%")
        np.savez(
            os.path.join(OUT_DIR, f"windows_{name}.npz"),
            X=X, is_normal=is_normal, polename=polename, start_time=start_time,
        )
    print(f"[legacy8] 已存至 {OUT_DIR}")
