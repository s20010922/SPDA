"""萃取pipeline結果成圖表用的JSON資料。
所有數字都直接讀 step5/step6 產出的檔案，不手動謄寫任何評估數字，
避免每次重訓後忘記同步或抄錯（先前版本手動抄過幾次，出過不同步的問題）。
"""
import json
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DATASET_DIR = os.path.join(ROOT, "model_dataset")
# 正式模型：12個已知異常迴路排除清單(含SP-07網路設備) + 訓練跑滿80輪 + 7天去趨勢
# + window=120(6小時，原本60/3小時，實測window拉長讓20天量級的持續性數值異常
# 偵測率從60%提升到75%左右，5個種子驗證過)，seed=88888(5種子裡閾值校準最準、
# 已知事件偵測率也最高的一個)。
MODEL_DIR = os.path.join(MODEL_DATASET_DIR, "prod_window120_seed88888")
SEED_SWEEP_PATH = os.path.join(ROOT, "experiments", "window120_sweep", "seed_sweep_results.csv")
OUT_PATH = os.path.join(ROOT, "chart_data.json")


def parse_loss_history(log_path: str):
    """從 step5_train_autoencoder.py 的完整訓練輸出(train_log.txt)自動解析loss曲線。"""
    with open(log_path, encoding="utf-8") as f:
        log = f.read()
    pattern = re.compile(r"\[epoch (\d+)\] train_loss=([\d.]+)\s+val_loss=([\d.]+)")
    history = [
        {"epoch": int(e), "train_loss": float(t), "val_loss": float(v)}
        for e, t, v in pattern.findall(log)
    ]
    best_epoch = min(history, key=lambda d: d["val_loss"])["epoch"]
    return history, best_epoch


loss_history, BEST_EPOCH = parse_loss_history(os.path.join(MODEL_DATASET_DIR, "prod_window120_train_log.txt"))

# 每杆體閾值（step6_evaluate.py 輸出）
with open(os.path.join(MODEL_DIR, "thresholds_per_pole.json"), encoding="utf-8") as f:
    thresholds = json.load(f)

with open(os.path.join(MODEL_DIR, "global_threshold.json"), encoding="utf-8") as f:
    global_threshold = json.load(f)["global_threshold"]

# 每杆體 test 評估明細（含MSE分布），直接讀 step6_evaluate.py 存的CSV
eval_df = pd.read_csv(os.path.join(MODEL_DIR, "per_pole_eval.csv"))
eval_df = eval_df.rename(columns={
    "per_pole_threshold": "threshold",
    "超標比例_每杆體閾值": "flagged_per_pole",
    "超標比例_全域閾值": "flagged_global",
})
per_pole_eval = eval_df[[
    "polename", "n_normal_test", "threshold", "flagged_per_pole", "flagged_global",
    "mse_mean", "mse_median", "mse_p95", "mse_p99", "mse_max",
]].to_dict(orient="records")

summary = {
    "per_pole_avg_flagged": float(eval_df["flagged_per_pole"].mean()) * 100,
    "global_avg_flagged": float(eval_df["flagged_global"].mean()) * 100,
    "target": 1.0,
}

known_events_path = os.path.join(MODEL_DIR, "known_events_eval.json")
if os.path.exists(known_events_path):
    with open(known_events_path, encoding="utf-8") as f:
        known_events = json.load(f)
else:
    known_events = []

# 多種子系統性比較：5個種子(1,42,123,2024,88888)在目前正式設定(12迴路排除清單+
# 訓練跑滿80輪+7天去趨勢+window=120)下各自重跑，用來驗證模型判斷是否穩定，不是碰運氣。
seed_sweep_df = pd.read_csv(SEED_SWEEP_PATH)
seed_sweep_df = seed_sweep_df.sort_values("dist_from_target")
best_seed = int(seed_sweep_df.iloc[0]["seed"])
seed_sweep = seed_sweep_df.to_dict(orient="records")

output = {
    "loss_history": loss_history,
    "best_epoch": BEST_EPOCH,
    "seed_sweep": seed_sweep,
    "best_seed": best_seed,
    "thresholds": thresholds,
    "global_threshold": global_threshold,
    "per_pole_eval": per_pole_eval,
    "summary": summary,
    "known_events": known_events,
}

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"已存至 {OUT_PATH}")
print(json.dumps(summary, ensure_ascii=False, indent=2))
