"""階段7：規則式斷訊偵測。
LSTM-Autoencoder只能判斷「有資料回傳、但數值異常」，對「資料完全消失」這種異常
結構性看不到(缺口太大切不出滑動窗口)，這是已知事件驗證發現的死角(SP-10斷訊9天、
SP-06提前離線都被LSTM判成0%異常)。這裡用簡單規則補上：正常回報間隔約3分鐘一筆，
超過門檻沒有任何回報就標記為斷訊異常，不需要訓練、不會跟LSTM互相干擾，兩者的結果
分開輸出，合併使用。
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd

CLEANED_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cleaned"
)
MODEL_DATASET_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model_dataset"
)

# 正常回報間隔中位數/p99都是3分鐘，10倍(30分鐘)明顯超出正常波動範圍才算斷訊，
# 避免把電表本身偶發的幾分鐘延遲誤判成斷訊。
OUTAGE_THRESHOLD_MINUTES = 30


def detect_outages(merged: pd.DataFrame, threshold_minutes: int = OUTAGE_THRESHOLD_MINUTES) -> pd.DataFrame:
    """逐杆體掃描reporttime間隔，超過門檻的區間標記為斷訊事件。
    回傳每筆斷訊事件的起訖時間、長度(分鐘)。跟資料末端到全域最後時間點的缺口
    也一併檢查(涵蓋"提前離線"這種只有結尾缺資料、沒有中間缺口的情況)。"""
    global_end = merged["reporttime"].max()
    global_start = merged["reporttime"].min()
    outages = []
    for pole, g in merged.groupby("polename"):
        g = g.sort_values("reporttime")
        times = g["reporttime"]
        gaps = times.diff()
        big_gaps = gaps[gaps > pd.Timedelta(minutes=threshold_minutes)]
        for idx in big_gaps.index:
            loc = g.index.get_loc(idx)
            start = g.iloc[loc - 1]["reporttime"]
            end = g.loc[idx, "reporttime"]
            outages.append({
                "polename": pole, "outage_start": start, "outage_end": end,
                "duration_minutes": (end - start).total_seconds() / 60,
            })
        last_time = times.max()
        if global_end - last_time > pd.Timedelta(minutes=threshold_minutes):
            outages.append({
                "polename": pole, "outage_start": last_time, "outage_end": global_end,
                "duration_minutes": (global_end - last_time).total_seconds() / 60,
            })
        first_time = times.min()
        if first_time - global_start > pd.Timedelta(minutes=threshold_minutes):
            outages.append({
                "polename": pole, "outage_start": global_start, "outage_end": first_time,
                "duration_minutes": (first_time - global_start).total_seconds() / 60,
            })
    return pd.DataFrame(outages).sort_values("duration_minutes", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    merged = pd.read_parquet(os.path.join(CLEANED_DIR, "merged.parquet"))
    outages = detect_outages(merged)

    print(f"=== 規則式斷訊偵測(門檻={OUTAGE_THRESHOLD_MINUTES}分鐘無回報) ===")
    print(outages.to_string(index=False))
    print(f"\n共偵測到 {len(outages)} 起斷訊事件，涉及 {outages['polename'].nunique()} 根杆體")

    total_outage_time = outages.groupby("polename")["duration_minutes"].sum().sort_values(ascending=False)
    print("\n各杆體累計斷訊時間(分鐘)：")
    print(total_outage_time.to_string())

    outages.to_csv(os.path.join(MODEL_DATASET_DIR, "outage_events.csv"), index=False, encoding="utf-8-sig")
    print(f"\n已存至 {MODEL_DATASET_DIR}\\outage_events.csv")
