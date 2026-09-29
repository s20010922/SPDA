"""從原始資料直接觀察，系統性找出跟模型判斷完全無關的「真實異常事件」候選，
作為驗證模型偵測能力的外部錨點。事件分兩類（event_type欄位）：
  - outage: 斷訊/資料缺失。判斷標準：時間間隔遠超過正常回報頻率(電表通常3分鐘
    一筆)，或資料提前結束(比其他柱子早很多結束)。這類事件目前LSTM窗口機制
    結構性看不到(缺口太大切不出窗口)，需要另外做規則式的斷訊偵測，不在本腳本
    範圍內。
  - value_anomaly: 資料持續存在，但數值本身有真實的巨幅異常。這類事件無法用
    自動掃描找到（跟斷訊不同，沒有客觀的時間間隔門檻可以套），目前是靠人工
    深入追查MSE異常時段的原始迴路讀數才發現的(例如SP-07/SP-08的路燈迴路在
    2026/1/2~1/22同步跳升100倍，1/23恢復)。發現後手動寫入 cleaned/known_events.csv，
    本腳本只自動重新產生 outage 類事件，value_anomaly 類需人工維護。

本腳本只重新掃描並更新 outage 類事件；若 known_events.csv 已存在人工新增的
value_anomaly 事件，會保留不覆蓋。
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd

CLEANED_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cleaned"
)
EVENTS_PATH = os.path.join(CLEANED_DIR, "known_events.csv")

if __name__ == "__main__":
    merged = pd.read_parquet(os.path.join(CLEANED_DIR, "merged.parquet"))
    global_end = merged["reporttime"].max()

    print("=== 逐杆體斷訊掃描（間隔>30分鐘，正常回報約3分鐘一筆）===")
    events = []
    for pole, g in merged.groupby("polename"):
        g = g.sort_values("reporttime")
        gaps = g["reporttime"].diff()
        big_gaps = gaps[gaps > pd.Timedelta("30min")]
        for idx in big_gaps.index:
            loc = g.index.get_loc(idx)
            start = g.iloc[loc - 1]["reporttime"]
            end = g.loc[idx, "reporttime"]
            events.append({
                "polename": pole, "event_type": "outage",
                "event_start": start, "event_end": end, "duration": end - start,
                "description": f"斷訊{end - start}",
            })
        last_time = g["reporttime"].max()
        if global_end - last_time > pd.Timedelta("12h"):
            events.append({
                "polename": pole, "event_type": "outage",
                "event_start": last_time, "event_end": global_end,
                "duration": global_end - last_time,
                "description": f"資料提前結束(比其他柱體早{global_end - last_time}無資料)",
            })

    outage_df = pd.DataFrame(events).sort_values("duration", ascending=False)
    outage_df = outage_df[outage_df["duration"] > pd.Timedelta("1h")].reset_index(drop=True)
    print(outage_df.to_string(index=False))

    # 保留既有的 value_anomaly 事件（人工維護，本腳本不產生也不覆蓋）
    if os.path.exists(EVENTS_PATH):
        existing = pd.read_csv(EVENTS_PATH)
        value_anomaly = existing[existing["event_type"] == "value_anomaly"]
    else:
        value_anomaly = pd.DataFrame(columns=outage_df.columns)

    combined = pd.concat([outage_df, value_anomaly], ignore_index=True)
    combined.to_csv(EVENTS_PATH, index=False)
    print(f"\n已更新 {EVENTS_PATH}（outage類自動重新掃描共{len(outage_df)}筆，"
          f"value_anomaly類保留人工維護的{len(value_anomaly)}筆）")
