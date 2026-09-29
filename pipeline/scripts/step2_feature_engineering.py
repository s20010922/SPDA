"""階段2：特徵工程。
把逐迴路讀數聚合成杆體層級特徵：w_total(全部迴路加總)、light_circuit_w(路燈迴路獨立用電)、
delta_whplus_total(累積電量一階差分)、w_total_detrended(7天移動平均去趨勢後的偏離量)。
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd

CLEANED_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cleaned"
)

# 已驗證的異常迴路。判斷方式：不是用絕對門檻(如"w>5000")，而是拿同類設備
# (同category)跨柱體互相對照——例如正常的MP(資料收集器)迴路耗電約100~290瓦，
# SP-02/07/08/09卻是5357~7465瓦(20~74倍)；正常攝影機約10~14瓦，這幾根柱子卻是
# 400~433瓦(30~40倍)；正常網路設備約8.5~9.4瓦，SP-07卻386.7瓦(41倍)；正常數位
# 看板約166~224瓦，這幾根柱子卻3997~5628瓦(18~34倍)。物理上這些設備不可能耗電
# 差這麼多倍，判斷為感測器/計量錯誤，非真實用電。
# 修正記錄：最初只用單一絕對門檻(w>5000)篩查MP+數位看板，漏掉SP-07的網路設備
# (386.7瓦未達5000)；改用「同category跨柱體互相對照」的方法後才抓到。
# SP-03刻意排除在外：它的MP(859瓦)、攝影機(524瓦)雖然也比其他柱體同類設備高
# 3~50倍，但倍數遠低於SP-02/07/08/09那種20~70倍的離譜程度；且SP-03其餘3個迴路
# (數位看板/智慧指標/路燈)本來就全部是0(可能未安裝)，若排除MP+攝影機，這根柱子
# 會完全沒有非零訊號，代價過大，故保留原始讀數，不視為需排除的異常。
# 路燈迴路(category=10)刻意不在此清單：SP-02/07/08/09的路燈讀數同樣異常
# (519~526瓦 vs 正常15~31瓦)，但路燈用電是模型的關鍵特徵(light_circuit_w)，
# 直接排除會讓這幾根柱子完全沒有路燈特徵，需要另外決定修正或排除策略，暫不處理。
KNOWN_BAD_CIRCUITS = {
    11336, 11337, 11339,  # SCCP-SP-02: MP、數位看板、攝影機
    11321, 11324, 11322,  # SCCP-SP-07: MP、網路設備、數位看板
    11331, 11332, 11334,  # SCCP-SP-08: MP、數位看板、攝影機
    11326, 11327, 11329,  # SCCP-SP-09: MP、數位看板、攝影機
}


def aggregate_pole_features(meter_raw: pd.DataFrame, exclude_circuits: set = None) -> pd.DataFrame:
    """依 polename+reporttime 聚合：w_total=全部迴路w加總，whplus_total=全部迴路whplus加總，
    light_circuit_w=該時間戳路燈迴路(is_light=True)的w值(若同時間戳有多筆路燈讀數取平均)，
    circuit_count=該時間戳有幾個迴路回報了資料。
    exclude_circuits 指定的迴路會從 w_total/whplus_total 加總跟 circuit_count 中排除，
    但不影響 light_circuit_w（路燈迴路本身不在已知異常清單內）。"""
    exclude_circuits = exclude_circuits or set()
    clean = meter_raw[~meter_raw["circuitid"].isin(exclude_circuits)]

    total = clean.groupby(["polename", "reporttime"]).agg(
        w_total=("w", "sum"),
        whplus_total=("whplus", "sum"),
        circuit_count=("circuitid", "nunique"),
    ).reset_index()

    light_only = meter_raw[meter_raw["is_light"]]
    light_agg = light_only.groupby(["polename", "reporttime"]).agg(
        light_circuit_w=("w", "mean")
    ).reset_index()

    merged = total.merge(light_agg, on=["polename", "reporttime"], how="left")
    return merged


def compute_delta(df: pd.DataFrame) -> pd.DataFrame:
    """whplus_total是累積值，每根杆體各自依時間排序後做一階差分。"""
    out = df.sort_values(["polename", "reporttime"]).copy()
    out["delta_whplus_total"] = out.groupby("polename")["whplus_total"].diff()
    return out


def compute_detrended(df: pd.DataFrame, window: str = "7D") -> pd.DataFrame:
    """對 w_total 算7天移動平均(只看過去，不看未來)，w_total_detrended = w_total - 移動平均。
    去趨勢是為了不讓緩慢的季節性/趨勢性用電上升被誤判成統計離群值
    （既有專案已用實際資料驗證過這個問題，見對話記錄）。"""
    out = df.sort_values(["polename", "reporttime"]).copy()

    def _rolling_per_pole(group: pd.DataFrame) -> pd.Series:
        s = group.set_index("reporttime")["w_total"].rolling(window, min_periods=1).mean()
        s.index = group.index
        return s

    parts = [_rolling_per_pole(g) for _, g in out.groupby("polename")]
    out["w_total_rolling_mean"] = pd.concat(parts)
    out["w_total_detrended"] = out["w_total"] - out["w_total_rolling_mean"]
    return out


def merge_with_light(meter_features: pd.DataFrame, light_wide: pd.DataFrame,
                      tolerance: str = "5min") -> pd.DataFrame:
    """電表跟路燈取樣頻率不同，用逐杆體 merge_asof 做最近鄰時間對齊，
    超出容忍度找不到就留NaN，不能硬拼接。"""
    parts = []
    for pole in sorted(meter_features["polename"].unique()):
        m = meter_features[meter_features["polename"] == pole].sort_values("reporttime")
        l = light_wide[light_wide["polename"] == pole].sort_values("reporttime")
        if len(l) == 0:
            merged = m.copy()
            merged["status"] = float("nan")
            merged["level"] = float("nan")
        else:
            merged = pd.merge_asof(
                m, l.drop(columns=["polename"]), on="reporttime",
                direction="nearest", tolerance=pd.Timedelta(tolerance),
            )
        parts.append(merged)
    return pd.concat(parts, ignore_index=True)


if __name__ == "__main__":
    meter_raw = pd.read_parquet(os.path.join(CLEANED_DIR, "meter_raw.parquet"))
    light_wide = pd.read_parquet(os.path.join(CLEANED_DIR, "light_wide.parquet"))

    n_excluded = meter_raw["circuitid"].isin(KNOWN_BAD_CIRCUITS).sum()
    print(f"排除已知異常迴路 {sorted(KNOWN_BAD_CIRCUITS)}，共 {n_excluded} 筆讀數不計入w_total加總")

    features = aggregate_pole_features(meter_raw, exclude_circuits=KNOWN_BAD_CIRCUITS)
    print(f"聚合後共 {len(features)} 個 (polename, reporttime) 組合")
    print(features.head(10).to_string(index=False))
    print("\n每根杆體 light_circuit_w 缺失比例：")
    print(features.groupby("polename")["light_circuit_w"].apply(lambda s: s.isna().mean()))

    features = compute_delta(features)
    features = compute_detrended(features)
    merged = merge_with_light(features, light_wide)

    print(f"\n最終合併表共 {len(merged)} 筆，欄位：{merged.columns.tolist()}")
    print(merged.head(10).to_string(index=False))
    print("\n每根杆體 status/level 缺失比例（電表-路燈時間對齊後）：")
    print(merged.groupby("polename")[["status", "level"]].apply(lambda d: d.isna().mean()))

    merged.to_parquet(os.path.join(CLEANED_DIR, "merged.parquet"), index=False)
    print(f"\n已存檔至 {CLEANED_DIR}\\merged.parquet")
