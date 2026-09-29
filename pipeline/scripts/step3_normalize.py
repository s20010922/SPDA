"""階段3：正規化與離群標記。
每根杆體各自對 w_total_detrended 算 2.5%/97.5% 百分位數，標記統計離群值；
每根杆體各自用「非離群值子集」估計 mean/std，做z-score正規化。
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd

CLEANED_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cleaned"
)


def flag_outliers_per_pole(df: pd.DataFrame, value_col: str,
                            lower_q: float = 0.025, upper_q: float = 0.975) -> pd.DataFrame:
    """回傳加了 is_outlier 欄位的新DataFrame。每根杆體各自算百分位數信賴區間，
    用百分位數法而非常態假設(mean±1.96*std)，因為實際用電資料常有偏態分布。"""
    out = df.copy()
    out["is_outlier"] = False
    for pole, group in out.groupby("polename"):
        valid = group[value_col].dropna()
        if len(valid) < 10:
            print(f"[警告] {pole} 的 {value_col} 有效樣本數僅 {len(valid)}，信賴區間可能不可靠")
            continue
        low, high = valid.quantile(lower_q), valid.quantile(upper_q)
        mask = out["polename"] == pole
        out.loc[mask, "is_outlier"] = (out.loc[mask, value_col] < low) | (out.loc[mask, value_col] > high)
    out["is_normal"] = ~out["is_outlier"]
    return out


def normalize_per_pole(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """回傳加了 <value_col>_z 欄位的新DataFrame。mean/std只從每根杆體的
    is_normal=True子集估計，避免離群值汙染統計量，套用時對全部資料轉換
    （含離群值本身，讓模型看得到偏離多少）。"""
    out = df.copy()
    z_col = f"{value_col}_z"
    out[z_col] = float("nan")
    for pole, group in out.groupby("polename"):
        normal_subset = group[group["is_normal"]][value_col].dropna()
        if len(normal_subset) < 10:
            print(f"[警告] {pole} 的正常子集樣本數僅 {len(normal_subset)}，正規化跳過")
            continue
        mean, std = normal_subset.mean(), normal_subset.std(ddof=0)
        if std == 0:
            std = 1.0
        mask = out["polename"] == pole
        out.loc[mask, z_col] = (out.loc[mask, value_col] - mean) / std
    return out


if __name__ == "__main__":
    merged = pd.read_parquet(os.path.join(CLEANED_DIR, "merged.parquet"))
    merged = flag_outliers_per_pole(merged, value_col="w_total_detrended")

    print("=== 每根杆體離群值比例 ===")
    summary = merged.groupby("polename")["is_outlier"].mean()
    print(summary)
    print(f"\n預期每根杆體離群比例接近 5%（因為用 2.5%~97.5% 信賴區間），"
          f"若某杆體明顯偏離這個數字，代表該杆體資料分布不穩定，需要留意。")

    merged = normalize_per_pole(merged, value_col="w_total_detrended")
    print("\n=== 正規化後 w_total_detrended_z 的每杆體統計（正常子集應接近 mean=0, std=1）===")
    print(merged[merged["is_normal"]].groupby("polename")["w_total_detrended_z"].agg(["mean", "std"]))

    merged["circuit_count"] = merged["circuit_count"].fillna(0)
    merged["light_circuit_w"] = merged["light_circuit_w"].fillna(0)

    merged.to_parquet(os.path.join(CLEANED_DIR, "merged_normalized.parquet"), index=False)
    print(f"\n已存檔至 {CLEANED_DIR}\\merged_normalized.parquet")
