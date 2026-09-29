"""階段1：資料清洗。
把電表4張表(deviceconfig/metercircuit/meterprofile/meterinfo2)串接成
「每筆讀數屬於哪根杆體、哪個迴路類型」的乾淨長表，並把路燈資料轉成寬表。

杆體歸屬用經緯度比對，不用 deviceconfig.devicename（該欄位原始位元組已損毀，
任何編碼都解不回來，只能靠獨立資料源的經緯度反推，見對話中前置探索確認）。
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd

DATA_ROOT = r"C:\Users\s9663\Desktop\智慧桿_v2\智慧杆資料應用整合工作坊_數據資料包_0930&1007場次"
METER_DIR = os.path.join(DATA_ROOT, "智慧電表", "115年1月-2月_各設備耗能_台北")
LIGHT_DIR = os.path.join(DATA_ROOT, "路燈照明")
OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cleaned"
)
os.makedirs(OUT_DIR, exist_ok=True)


def read_csv_relaxed(path: str, **kwargs) -> pd.DataFrame:
    """有些檔案(如deviceconfig)的devicename欄位原始位元組已損毀，非合法UTF-8。
    我們不需要用到該欄位內容，用errors="replace"容錯讀取，不影響其他欄位正確性。"""
    with open(path, "rb") as f:
        raw = f.read()
    text = raw.decode("utf-8", errors="replace")
    from io import StringIO
    return pd.read_csv(StringIO(text), **kwargs)


def build_dcid_to_polename() -> pd.DataFrame:
    """用經緯度比對 deviceconfig 跟官方杆體主檔，回傳 (dcid, polename) 對照表。"""
    dc = read_csv_relaxed(
        os.path.join(METER_DIR, "deviceconfig_202604011636.csv")
    )
    pole_master = pd.read_csv(
        os.path.join(DATA_ROOT, "杆體資訊_smartpole_list_2026.csv"), encoding="utf-8-sig"
    )
    pole_master = pole_master[pole_master["polename"].str.startswith("SCCP")].copy()

    dc["lat_r"] = dc["latitude"].round(5)
    dc["lon_r"] = dc["longitude"].round(5)
    pole_master["lat_r"] = pole_master["latitude"].round(5)
    pole_master["lon_r"] = pole_master["longitude"].round(5)

    merged = dc.merge(
        pole_master[["polename", "lat_r", "lon_r"]], on=["lat_r", "lon_r"], how="left"
    )
    unmatched = merged[merged["polename"].isna()]
    if len(unmatched) > 0:
        raise ValueError(
            f"有 {len(unmatched)} 筆 deviceconfig 資料找不到對應杆體（經緯度對不上），"
            f"dcid清單：{unmatched['dcid'].tolist()}。需要先人工確認再繼續，不能靜默丟棄。"
        )
    return merged[["dcid", "polename"]].drop_duplicates()


def build_circuit_category_map() -> pd.DataFrame:
    """回傳 (circuitid, category, is_light) 對照表。category=10 代表路燈迴路
    （已用 metercircuit.loadname 如「松菸智慧杆 #04-路燈」逐筆核對確認，
    loadname本身是乾淨UTF-8不需修復，只有deviceconfig.devicename才損毀）。"""
    mc = pd.read_csv(
        os.path.join(METER_DIR, "metercircuit_202604011646.csv"), encoding="utf-8", dtype=str
    )
    mc = mc.dropna(subset=["category"]).copy()
    mc["category"] = mc["category"].astype(float).astype(int)
    mc["circuitid"] = mc["circuitid"].astype(int)
    mc["is_light"] = mc["category"] == 10
    return mc[["circuitid", "category", "is_light", "loadname"]].drop_duplicates("circuitid")


def build_meter_raw(dcid_map: pd.DataFrame, circuit_map: pd.DataFrame) -> pd.DataFrame:
    """串接 meterinfo2(歷史逐筆) + meterprofile(meterid->dcid/circuitid靜態對照)，
    回傳含 polename, is_light, reporttime, w, whplus 的長表。"""
    profile = pd.read_csv(
        os.path.join(METER_DIR, "meterprofile_202604011700.csv"), encoding="utf-8"
    )
    profile_map = profile[["meterid", "dcid", "circuitid"]].drop_duplicates("meterid")

    info = pd.read_csv(
        os.path.join(METER_DIR, "meterinfo2_202604011708.csv"), encoding="utf-8"
    )
    info["reporttime"] = pd.to_datetime(info["reporttime"])

    merged = info.merge(profile_map, on="meterid", how="left")
    orphan = merged[merged["dcid"].isna()]
    orphan_ratio = len(orphan) / len(merged)
    print(f"meterinfo2 -> meterprofile join：孤兒列 {len(orphan)} 筆（{orphan_ratio*100:.2f}%）")
    if orphan_ratio > 0.01:
        raise ValueError(
            f"孤兒列比例 {orphan_ratio*100:.2f}% 超過1%，需要先查清楚原因再繼續，"
            "不能靜默丟棄（可能是meterid對照表不完整或有新設備未登記）。"
        )
    merged = merged.dropna(subset=["dcid"]).copy()
    merged["dcid"] = merged["dcid"].astype(int)

    merged = merged.merge(dcid_map, on="dcid", how="left")
    merged = merged.merge(circuit_map[["circuitid", "is_light"]], on="circuitid", how="left")
    merged["is_light"] = merged["is_light"].fillna(False)

    return merged[["polename", "reporttime", "meterid", "circuitid", "is_light", "w", "whplus"]]


def build_light_wide() -> pd.DataFrame:
    """讀路燈長表(deviceid, attrid, reporttime, value)，轉成寬表(polename, reporttime, status, level)。
    deviceid->polename 用 lightinfo_代號對照.xlsx 下半部的對照表轉換。
    attrid: 103800=控制器狀態(1連線/3異常/9其他), 100800=亮度百分比。"""
    xlsx_path = os.path.join(LIGHT_DIR, "lightinfo_代號對照.xlsx")
    raw = pd.read_excel(xlsx_path, sheet_name=0, header=None)
    header_row = raw[raw.iloc[:, 0] == "group"].index[0]
    device_map = pd.read_excel(xlsx_path, sheet_name=0, skiprows=header_row + 1,
                                names=["group", "deviceid", "polename"])
    device_map = device_map.dropna(subset=["deviceid"])[["deviceid", "polename"]].drop_duplicates()
    # xlsx裡的deviceid含多餘空白(如"INL1 --- 1765521500346")，實際csv裡沒有空白
    # (如"INL1---1765521500346")，須先去除所有空白字元再比對(既有專案也記錄過這個問題)。
    device_map["deviceid"] = device_map["deviceid"].str.replace(" ", "", regex=False)

    light = pd.read_csv(
        os.path.join(LIGHT_DIR, "lightnumberinfo_114.12-115.04.csv"), encoding="utf-8"
    )
    light["reporttime"] = pd.to_datetime(light["reporttime"])
    light["deviceid"] = light["deviceid"].str.replace(" ", "", regex=False)
    light = light.merge(device_map, on="deviceid", how="left")

    orphan = light[light["polename"].isna()]
    orphan_ratio = len(orphan) / len(light)
    print(f"路燈 deviceid -> polename join：孤兒列 {len(orphan)} 筆（{orphan_ratio*100:.2f}%）")
    if len(orphan) > 0:
        orphan_devices = sorted(orphan["deviceid"].unique())
        print(f"[已知資料缺陷] 對照表(lightinfo_代號對照.xlsx)缺漏以下deviceid，"
              f"且對照表本身把SCCP-SP-09重複列了兩次（可能是原始維護錯誤，"
              f"其中一列本該對應這個缺漏的deviceid，但無法確認，不用猜測填補）：")
        print(f"  {orphan_devices}")
        print(f"  這些deviceid的資料筆數共 {len(orphan)} 筆，將被排除，"
              f"對應杆體在路燈特徵上會有缺失，需在後續評估報告中明確標註。")
    if orphan_ratio > 0.15:
        raise ValueError(
            f"路燈孤兒列比例 {orphan_ratio*100:.2f}% 過高，超出可接受範圍，需要先確認對照表。"
        )
    light = light.dropna(subset=["polename"])

    wide = light.pivot_table(
        index=["polename", "reporttime"], columns="attrid", values="value", aggfunc="first"
    ).reset_index()
    wide = wide.rename(columns={103800: "status", 100800: "level"})
    keep_cols = [c for c in ["polename", "reporttime", "status", "level"] if c in wide.columns]
    return wide[keep_cols]


if __name__ == "__main__":
    dcid_map = build_dcid_to_polename()
    print("=== dcid -> polename 對照表 ===")
    print(dcid_map.to_string(index=False))
    print(f"\n共 {len(dcid_map)} 筆，應為 10（台北SCCP-SP-01~10各一個dcid）")

    circuit_map = build_circuit_category_map()
    print(f"\n=== 迴路類型對照表，共 {len(circuit_map)} 筆 ===")
    print(f"其中路燈迴路(category=10)：{circuit_map['is_light'].sum()} 筆")
    light_circuits = circuit_map[circuit_map["is_light"]]
    print(light_circuits[["circuitid", "loadname"]].to_string(index=False))

    meter_raw = build_meter_raw(dcid_map, circuit_map)
    print(f"\n=== meter_raw 串接完成，共 {len(meter_raw)} 筆 ===")
    print(meter_raw.head(10).to_string(index=False))
    print("\n每根杆體筆數：")
    print(meter_raw.groupby("polename").size())
    print("\n每根杆體是否有路燈迴路資料：")
    print(meter_raw.groupby("polename")["is_light"].sum())

    light_wide = build_light_wide()
    print(f"\n=== light_wide 轉換完成，共 {len(light_wide)} 筆 ===")
    print(light_wide.head(10).to_string(index=False))
    print("\n每根杆體路燈資料筆數：")
    print(light_wide.groupby("polename").size())

    meter_raw.to_parquet(os.path.join(OUT_DIR, "meter_raw.parquet"), index=False)
    light_wide.to_parquet(os.path.join(OUT_DIR, "light_wide.parquet"), index=False)
    print(f"\n已存檔至 {OUT_DIR}\\meter_raw.parquet 和 light_wide.parquet")
