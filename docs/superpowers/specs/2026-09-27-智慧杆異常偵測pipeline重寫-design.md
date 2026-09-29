# 智慧杆異常用電偵測 Pipeline 重寫 — 設計文件

> 日期：2026-09-27
> 目的：從零重寫「智慧杆在每個時間點為正常/異常」二元分類 pipeline，取代既有專案（`智慧杆資料應用整合工作坊_數據資料包_0930&1007場次-20260924T105722Z-1-001/`）中閾值穩健性不足的版本。全程前台逐步執行並解說細節，供使用者學習每一步的原理與程式碼。

## 背景與決策依據

既有專案（團隊「釣端抖」）已完整跑過一輪：資料清洗 → per-pole 正規化 → LSTM-Autoencoder 訓練 → 全域共用閾值評估。經實測驗證：
- train/val/test 正常子集的重建誤差量級相近（0.32/0.30/0.27），**不是傳統 train≫test 的 overfitting**
- 但 **閾值穩健性不足**：val 校準閾值（p99）套到 test 正常子集，超標比例達 1.93%，是校準目標（1%）的近2倍，代表資料分布隨時間有漂移
- 既有專案已用「移動平均去趨勢」修正過一次趨勢污染問題（見其 git commit `5f0ef17`），但殘留漂移仍在

本次重寫**沿用既有專案已驗證有效的架構決策**（見下方逐項依據），**只在閾值機制上做核心改進**：從「全部杆體共用一個閾值」改為「每根杆體各自校準閾值」。

| 決策 | 選擇 | 依據 |
|---|---|---|
| 杆體範圍 | 只做台北 10 根（SCCP-SP-01~10） | 高雄場域除電表外，路燈/人流/環境感測資料幾乎全空，既有專案已查證 |
| 電表資料版本 | 1-2月「各設備耗能」迴路級（非5-6月總耗能） | 迴路級才能拆出「路燈迴路」獨立用電，精準比對路燈亮滅狀態 vs 實際耗電，直接對應提案書的核心異常型態定義 |
| 正規化 | per-pole z-score（非全域） | 10根杆體有結構性的 A群(~5迴路)/B群(~6迴路)差異，全域正規化會讓「杆體規模」變成混淆因子 |
| 去趨勢 | 7天移動平均去趨勢 | 既有專案已用實際資料驗證：固定期間統計量會把季節性/趨勢性緩慢上升誤判為離群值 |
| 時間切分 | 按時間先後切 train/val/test（非隨機） | 避免滑動窗口重疊造成資料洩漏 |
| **閾值機制（本次核心改動）** | **每根杆體各自校準閾值**（非全域共用） | 既有專案「全域共用閾值」被證實穩健性不足；不同杆體物理規模/雜訊水準不同，用該杆體自己的 val 正常子集校準閾值，邏輯上與 per-pole 正規化一致 |

## 架構：6 個階段

```
階段1 資料清洗 → 階段2 特徵工程 → 階段3 正規化與離群標記
    → 階段4 時間切分與切窗 → 階段5 LSTM-Autoencoder訓練 → 階段6 每杆體閾值校準與評估
```

### 階段1：資料清洗

**輸入**：
- `智慧電表/115年1月-2月_各設備耗能_台北/deviceconfig_202604011636.csv`
- `.../metercircuit_202604011646.csv`
- `.../meterprofile_202604011700.csv`
- `.../meterinfo2_202604011708.csv`
- `路燈照明/lightnumberinfo_114.12-115.04.csv`
- `路燈照明/lightinfo_代號對照.xlsx`
- `杆體資訊_smartpole_list_2026.csv`

**處理**：
1. `deviceconfig` 用 `latitude`/`longitude` 比對 `杆體資訊_smartpole_list_2026.csv` 的經緯度，建立 `dcid → polename` 對照表（devicename 欄位是亂碼，不可用；緯經度是唯一可靠關聯鍵）
2. 串接鏈：`meterinfo2(meterid)` ⟵join⟶ `meterprofile(meterid, dcid, circuitid)` ⟵join⟶ `metercircuit(circuitid, dcid, loadid, loadname, category)`，再用 `dcid` 對到 `polename`
3. 路燈：用 `lightinfo_代號對照.xlsx` 下半部 `deviceid↔polename` 表轉換，長表 `(deviceid, attrid, reporttime, value)` pivot 成寬表 `(polename, reporttime, status, level)`
4. 輸出中繼檔：`meter_raw.parquet`（含 polename, reporttime, meterid, circuitid, loadname, v, a, w, whplus）、`light_wide.parquet`（polename, reporttime, status, level）

**驗證**：每個 polename 至少要對到一筆 deviceconfig 記錄；join 後檢查是否有孤兒列（dcid/circuitid 對不到的），若有需回報而非靜默丟棄。

### 階段2：特徵工程

1. 依 `polename + reporttime` 把同時間戳的所有迴路 `w` 加總 → `w_total`
2. 從 `metercircuit.loadname`/`category` 找出路燈迴路，取其 `w` 當 `light_circuit_w`（若某杆體對不到路燈迴路，記錄警告並以 0 或 NaN 填補，需明確標記非靜默處理）
3. `whplus_total`（w同樣邏輯加總）做一階差分 → `delta_whplus_total`（每根杆體第一筆為 NaN）
4. 對 `w_total` 算 7 天移動平均 `w_total_rolling_mean`（只看過去，`min_periods=1`），再算 `w_total_detrended = w_total - w_total_rolling_mean`
5. 路燈寬表用 `merge_asof` 最近鄰對齊電表時間戳（tolerance 5分鐘，超出留 NaN）

**輸出**：合併後的主表 `merged.parquet`，欄位含 `polename, reporttime, w_total, w_total_detrended, delta_whplus_total, light_circuit_w, status, level, circuit_count`

### 階段3：正規化與離群標記

1. 每根杆體各自對 `w_total_detrended` 算 2.5%/97.5% 百分位數 → 標記 `is_outlier`
2. 每根杆體各自用「非離群值子集」估計 mean/std，做 z-score 正規化 → `w_total_detrended_z`
3. `is_normal = ~is_outlier`，作為後續訓練篩選欄位

### 階段4：時間切分與切窗

1. 按 `reporttime` 先後切分：前 60% 時間 train，中間 20% val，後 20% test（依實際資料涵蓋期間換算日期界線，不用隨機切分）
2. 每根杆體各自切滑動窗口（window_size=60, stride=1），不跨杆體邊界
3. 特徵欄位：`[w_total_detrended_z, circuit_count, light_circuit_w]`（沿用既有專案驗證過的3特徵組合）
4. 輸出 `windows_train.npz` / `windows_val.npz` / `windows_test.npz`，各含 `X, is_normal, polename, start_time`

### 階段5：LSTM-Autoencoder 訓練

- Encoder: LSTM(3→32)，取最後 hidden state 當 latent
- Decoder: LSTM(32→32) + Linear(32→3)，RepeatVector 方式重建整個窗口
- 只用 train 集 `is_normal=True` 的窗口訓練，MSE loss
- val 集 loss 做 early stopping（patience=5）
- 輸出 `lstm_autoencoder.pt`、`feature_scaler.npz`（訓練時用的標準化 mean/std）

### 階段6：每杆體閾值校準與評估（核心改進）

1. 對每根杆體 p，用該杆體在 val 集 `is_normal=True` 子集上的重建誤差，算 p99 當該杆體的閾值 `threshold[p]`
2. 若某杆體 val 正常樣本數過少（如 <30 筆），記錄警告，退回用全域閾值當備援，不能無聲產生不可靠的極端閾值
3. 在 test 集驗證：每根杆體「正常子集」的實際超標比例是否貼近 1%（校準目標），比較「全域共用閾值」vs「每杆體別閾值」何者更接近目標，作為本次改進是否成功的量化證據
4. 用 SCCP-SP-06 已知「提前離線」真實異常案例，檢查該時段窗口是否被新閾值機制標記為異常

**輸出**：`thresholds_per_pole.json`、`evaluation_report.md`（含每杆體超標比例對照表、SCCP-SP-06驗證結果）

## 錯誤處理原則

- join 失敗（孤兒列）：記錄筆數與比例，不靜默丟棄，若比例異常需先回報使用者而非自動略過
- 路燈對不到迴路的杆體：明確標記，不用預設值掩蓋資料缺失
- val 正常樣本數過少導致閾值不可靠：記錄警告並退回全域閾值，附上判斷依據

## 範圍界線

**包含**：資料清洗、特徵工程、正規化、切窗、模型訓練、閾值校準、評估報告。
**不包含**：人工模擬異常事件注入、視覺化/儀表板、Transformer 監督式版本 — 這些是既有專案待辦清單，本次重寫聚焦在解決閾值穩健性問題，其餘沿用既有專案規劃即可，時程允許再議。

## 待確認事項

無（本次設計範圍內的決策均已在對話中與使用者確認）。
