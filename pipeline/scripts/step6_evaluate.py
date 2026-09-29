"""階段6：每杆體閾值校準與評估。
核心改進：改用每根杆體各自的val正常子集重建誤差p99當閾值，取代既有專案
「全部杆體共用一個閾值」的做法（該做法實測test超標比例是校準目標的近2倍）。

evaluate(model_dir) 可被 run_seed_sweep.py 重複呼叫，各自對不同種子訓練出的
模型評估，不寫死路徑，才能做多種子系統性比較。
"""
import argparse
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

DEFAULT_MODEL_DATASET_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model_dataset"
)
HIDDEN_SIZE = 32
MIN_VAL_SAMPLES = 30
DEVICE = torch.device("cpu")


class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features: int, hidden_size: int):
        super().__init__()
        self.encoder = nn.LSTM(n_features, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, n_features)

    def forward(self, x):
        window_size = x.shape[1]
        _, (h_n, _) = self.encoder(x)
        latent = h_n[-1]
        decoder_input = latent.unsqueeze(1).repeat(1, window_size, 1)
        decoded, _ = self.decoder(decoder_input)
        return self.output_layer(decoded)


def load_split(name: str, windows_dir: str):
    data = np.load(os.path.join(windows_dir, f"windows_{name}.npz"), allow_pickle=True)
    return data["X"].astype(np.float32), data["is_normal"], data["polename"], data["start_time"]


def compute_recon_error_per_window(model, X, mean, std, batch_size=512):
    Xs = (X - mean) / std
    errors = []
    with torch.no_grad():
        for i in range(0, len(Xs), batch_size):
            batch = torch.from_numpy(Xs[i:i + batch_size]).to(DEVICE)
            recon = model(batch)
            mse = ((recon - batch) ** 2).mean(dim=(1, 2))
            errors.append(mse.cpu().numpy())
    return np.concatenate(errors)


def check_known_events(pole_all, time_all, errors_all, is_normal_all, thresholds, events_path):
    """讀取 cleaned/known_events.csv 裡人工整理、跟模型判斷完全無關的真實事件
    （靠原始資料斷訊掃描或人工深入追查MSE異常時段找到的，見 build_known_events.py），
    逐一計算「事件期偵測率」對照「事件前一段乾淨基準期偵測率」，避免只看事件期
    偵測率被基準期本身誤判率虛高的假象誤導（例如事件前一週如果本身也是異常期，
    會錯把真異常誤判成模型雜訊，這個問題實際發生過一次，見對話紀錄SP-07/SP-08）。
    outage類事件（斷訊）通常查不到窗口，因為缺口太大切不出窗口，這是已知的
    模型結構性限制，不是bug，仍會如實記錄「事件期窗口數=0」。
    pole_all/time_all/errors_all/is_normal_all 應涵蓋train+val+test全部切分，因為
    事件可能落在任何一個切分裡（例如SP-07/SP-08事件落在train集）。
    基準期只用is_normal=True的窗口計算誤判率（假警報率的定義本來就是「模型在
    本該正常的資料上誤報的比例」，若混入本來就標記異常的窗口會虛高估計；曾經
    因為漏掉這個過濾，把SP-06的假警報率誤算成13.8%，實際上只有0.17%）。
    """
    if not os.path.exists(events_path):
        return [], "找不到 known_events.csv，略過已知事件驗證。"

    events = pd.read_csv(events_path, parse_dates=["event_start", "event_end"])
    results = []
    lines = []
    for _, ev in events.iterrows():
        pole = ev["polename"]
        thresh = thresholds.get(pole)
        if thresh is None:
            continue
        mask_pole = pole_all == pole
        times = time_all[mask_pole]
        errs = errors_all[mask_pole]
        normal = is_normal_all[mask_pole]

        event_start = np.datetime64(ev["event_start"])
        event_end = np.datetime64(ev["event_end"])
        event_mask = (times >= event_start - np.timedelta64(3, "h")) & (times <= event_end)
        baseline_start = event_start - np.timedelta64(14, "D")
        baseline_end = event_start - np.timedelta64(3, "h")
        baseline_mask = (times >= baseline_start) & (times < baseline_end) & normal

        n_event = int(event_mask.sum())
        n_baseline = int(baseline_mask.sum())
        event_flagged = float((errs[event_mask] > thresh).mean() * 100) if n_event > 0 else None
        baseline_flagged = float((errs[baseline_mask] > thresh).mean() * 100) if n_baseline > 0 else None

        results.append({
            "polename": pole, "event_type": ev["event_type"],
            "event_start": str(ev["event_start"]), "event_end": str(ev["event_end"]),
            "description": ev["description"],
            "n_event_windows": n_event, "event_flagged_pct": event_flagged,
            "n_baseline_windows": n_baseline, "baseline_flagged_pct": baseline_flagged,
        })

        line = f"{pole} [{ev['event_type']}] {ev['description']}"
        if n_event == 0:
            line += "\n  事件期沒有窗口資料可驗證（通常是outage類事件，缺口太大切不出窗口——模型結構性看不到這類異常，非bug）"
        else:
            line += f"\n  事件期窗口數={n_event}, 標記異常比例={event_flagged:.1f}%"
            if n_baseline > 0:
                line += f"；事件前14天乾淨基準期窗口數={n_baseline}, 同期誤判率={baseline_flagged:.1f}%（差距 {event_flagged - baseline_flagged:+.1f} 個百分點）"
                if baseline_flagged is not None and baseline_flagged > 10:
                    line += "\n  [警語] 基準期誤判率異常偏高，可能代表基準期本身也落在其他已知異常事件範圍內，此對照不可靠，需人工核對時間區間"
        lines.append(line)

    return results, "\n".join(lines)


def evaluate(model_dir: str, windows_dir: str = None, verbose: bool = True):
    """對 model_dir 底下的 lstm_autoencoder.pt + feature_scaler.npz 做完整評估，
    windows_dir 預設跟 model_dir 同一份（切窗資料是共用的，只有模型權重依種子各自獨立）。
    回傳 (comparison_df, summary_dict, known_events) 供 run_seed_sweep.py 彙整比較。"""
    windows_dir = windows_dir or DEFAULT_MODEL_DATASET_DIR

    scaler = np.load(os.path.join(model_dir, "feature_scaler.npz"))
    mean, std = scaler["mean"], scaler["std"]

    X_val, is_normal_val, polename_val, _ = load_split("val", windows_dir)
    n_features = X_val.shape[2]
    state_dict = torch.load(os.path.join(model_dir, "lstm_autoencoder.pt"), map_location=DEVICE)
    # hidden_size直接從checkpoint的LSTM權重形狀推斷，不寫死，支援不同容量的模型(超參數實驗用)
    hidden_size = state_dict["encoder.weight_hh_l0"].shape[1]
    model = LSTMAutoencoder(n_features=n_features, hidden_size=hidden_size)
    model.load_state_dict(state_dict)
    model.eval()

    val_errors = compute_recon_error_per_window(model, X_val, mean, std)
    global_threshold = float(np.percentile(val_errors[is_normal_val], 99))
    if verbose:
        print(f"全域閾值（備援用）：{global_threshold:.5f}")

    thresholds = {}
    for pole in sorted(set(polename_val)):
        mask = (polename_val == pole) & is_normal_val
        n_samples = mask.sum()
        if n_samples < MIN_VAL_SAMPLES:
            if verbose:
                print(f"[警告] {pole} 的val正常樣本數僅 {n_samples}（<{MIN_VAL_SAMPLES}），"
                      f"退回使用全域閾值 {global_threshold:.5f}")
            thresholds[pole] = global_threshold
        else:
            pole_threshold = float(np.percentile(val_errors[mask], 99))
            thresholds[pole] = pole_threshold
            if verbose:
                print(f"{pole}: val正常樣本數={n_samples}, 專屬閾值={pole_threshold:.5f}")

    with open(os.path.join(model_dir, "thresholds_per_pole.json"), "w", encoding="utf-8") as f:
        json.dump(thresholds, f, ensure_ascii=False, indent=2)
    with open(os.path.join(model_dir, "global_threshold.json"), "w", encoding="utf-8") as f:
        json.dump({"global_threshold": global_threshold}, f)

    X_test, is_normal_test, polename_test, start_time_test = load_split("test", windows_dir)
    test_errors = compute_recon_error_per_window(model, X_test, mean, std)

    rows = []
    for pole in sorted(set(polename_test)):
        mask_all = polename_test == pole
        mask_normal = mask_all & is_normal_test
        if mask_normal.sum() == 0:
            continue
        pole_thresh = thresholds[pole]
        pole_mse = test_errors[mask_normal]

        rows.append({
            "polename": pole,
            "n_normal_test": int(mask_normal.sum()),
            "per_pole_threshold": pole_thresh,
            "超標比例_每杆體閾值": (pole_mse > pole_thresh).mean(),
            "超標比例_全域閾值": (pole_mse > global_threshold).mean(),
            "mse_mean": float(pole_mse.mean()),
            "mse_median": float(np.median(pole_mse)),
            "mse_p95": float(np.percentile(pole_mse, 95)),
            "mse_p99": float(np.percentile(pole_mse, 99)),
            "mse_max": float(pole_mse.max()),
        })
    comparison = pd.DataFrame(rows)

    if verbose:
        print("\n=== test集正常子集 MSE(重建誤差)分布，逐杆體 ===")
        print(comparison[["polename", "n_normal_test", "mse_mean", "mse_median",
                           "mse_p95", "mse_p99", "mse_max", "per_pole_threshold"]].to_string(index=False))
        print("\n=== test集正常子集超標比例對照（目標應接近1%，因為閾值是p99校準）===")
        print(comparison[["polename", "n_normal_test", "per_pole_threshold",
                           "超標比例_每杆體閾值", "超標比例_全域閾值"]].to_string(index=False))

    per_pole_avg = float(comparison["超標比例_每杆體閾值"].mean()) * 100
    global_avg = float(comparison["超標比例_全域閾值"].mean()) * 100
    if verbose:
        print(f"\n每杆體閾值法平均超標比例：{per_pole_avg:.2f}%")
        print(f"全域閾值法平均超標比例：{global_avg:.2f}%")

    # 已知事件可能落在train/val/test任何一個切分裡（例如SP-07/SP-08事件落在train集），
    # 全部集合都要合併查詢，不能只用test集。
    X_train, is_normal_train, polename_train, start_time_train = load_split("train", windows_dir)
    train_errors = compute_recon_error_per_window(model, X_train, mean, std)
    pole_all = np.concatenate([polename_train, polename_val, polename_test])
    time_all = np.concatenate([start_time_train, load_split("val", windows_dir)[3], start_time_test])
    errors_all = np.concatenate([train_errors, val_errors, test_errors])
    is_normal_all = np.concatenate([is_normal_train, is_normal_val, is_normal_test])

    events_path = os.path.join(os.path.dirname(windows_dir), "cleaned", "known_events.csv")
    if not os.path.exists(events_path):
        events_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                    "cleaned", "known_events.csv")
    known_events, anomaly_msg = check_known_events(pole_all, time_all, errors_all, is_normal_all, thresholds, events_path)
    if verbose:
        print(f"\n=== 已知異常案例驗證 ===\n{anomaly_msg}")

    comparison.to_csv(os.path.join(model_dir, "per_pole_eval.csv"), index=False, encoding="utf-8-sig")
    with open(os.path.join(model_dir, "known_events_eval.json"), "w", encoding="utf-8") as f:
        json.dump(known_events, f, ensure_ascii=False, indent=2)

    report_lines = [
        "# 智慧杆異常偵測模型評估報告",
        "",
        "## test集正常子集 MSE(重建誤差)分布，逐杆體",
        "",
        "```",
        comparison[["polename", "n_normal_test", "mse_mean", "mse_median",
                    "mse_p95", "mse_p99", "mse_max", "per_pole_threshold"]].to_string(index=False),
        "```",
        "",
        "## 每杆體閾值校準結果",
        "",
        "```",
        comparison[["polename", "n_normal_test", "per_pole_threshold",
                    "超標比例_每杆體閾值", "超標比例_全域閾值"]].to_string(index=False),
        "```",
        "",
        f"每杆體閾值法平均超標比例：{per_pole_avg:.2f}%（目標1%）",
        f"全域閾值法平均超標比例：{global_avg:.2f}%（目標1%）",
        "",
        "## 已知異常案例驗證（跟模型判斷完全無關，靠原始資料觀察找到的真實事件）",
        "",
        anomaly_msg,
    ]
    with open(os.path.join(model_dir, "evaluation_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    if verbose:
        print(f"\n評估明細已存至 {model_dir}")

    summary = {
        "per_pole_avg_flagged": per_pole_avg,
        "global_avg_flagged": global_avg,
        "known_events": known_events,
    }
    return comparison, summary, known_events


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default=DEFAULT_MODEL_DATASET_DIR)
    parser.add_argument("--windows-dir", type=str, default=None)
    args = parser.parse_args()
    evaluate(args.model_dir, args.windows_dir)
