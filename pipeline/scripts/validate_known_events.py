"""用 build_known_events.py 產出的4個「跟模型判斷完全無關」的真實事件
（純粹從原始時間戳斷訊觀察得到），驗證模型在這些事件附近的異常偵測表現，
同時對照事件發生前一段「正常時期」的誤判率，避免只看事件期偵測率被
整體誤判率虛高的假象誤導。
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEANED_DIR = os.path.join(ROOT, "cleaned")
MODEL_DIR = os.path.join(ROOT, "model_dataset", "excl_v2_seed1")
WINDOWS_DIR = os.path.join(ROOT, "model_dataset")
HIDDEN_SIZE = 32
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


def load_split(name):
    data = np.load(os.path.join(WINDOWS_DIR, f"windows_{name}.npz"), allow_pickle=True)
    return data["X"].astype(np.float32), data["is_normal"], data["polename"], data["start_time"]


def compute_recon_error(model, X, mean, std, batch_size=512):
    Xs = (X - mean) / std
    errors = []
    with torch.no_grad():
        for i in range(0, len(Xs), batch_size):
            batch = torch.from_numpy(Xs[i:i + batch_size]).to(DEVICE)
            recon = model(batch)
            mse = ((recon - batch) ** 2).mean(dim=(1, 2))
            errors.append(mse.cpu().numpy())
    return np.concatenate(errors)


if __name__ == "__main__":
    events = pd.read_csv(os.path.join(CLEANED_DIR, "known_events.csv"), parse_dates=["gap_start", "gap_end"])
    print("=== 驗證用的4個真實事件（跟模型判斷完全無關，純靠原始資料斷訊觀察）===")
    print(events.to_string(index=False))

    scaler = np.load(os.path.join(MODEL_DIR, "feature_scaler.npz"))
    mean, std = scaler["mean"], scaler["std"]
    import json
    with open(os.path.join(MODEL_DIR, "thresholds_per_pole.json"), encoding="utf-8") as f:
        thresholds = json.load(f)

    # 事件可能落在train/val/test任何一個切分裡，全部集合都要查
    all_X, all_is_normal, all_pole, all_time = [], [], [], []
    for split in ["train", "val", "test"]:
        X, is_normal, pole, t = load_split(split)
        all_X.append(X); all_is_normal.append(is_normal); all_pole.append(pole); all_time.append(t)
    X_all = np.concatenate(all_X)
    pole_all = np.concatenate(all_pole)
    time_all = np.concatenate(all_time)

    n_features = X_all.shape[2]
    model = LSTMAutoencoder(n_features=n_features, hidden_size=HIDDEN_SIZE)
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "lstm_autoencoder.pt"), map_location=DEVICE))
    model.eval()

    errors_all = compute_recon_error(model, X_all, mean, std)

    print("\n=== 逐事件驗證：事件期偵測率 vs 事件前一週正常期誤判率（基準線）===")
    for _, ev in events.iterrows():
        pole = ev["polename"]
        gap_start = np.datetime64(ev["gap_start"])
        gap_end = np.datetime64(ev["gap_end"])
        thresh = thresholds.get(pole)
        if thresh is None:
            print(f"{pole}: 找不到閾值，跳過")
            continue

        mask_pole = pole_all == pole
        times = time_all[mask_pole]
        errs = errors_all[mask_pole]

        # 事件期：窗口起點落在 [gap_start - 60*3min緩衝, gap_end] 之間都算受事件影響
        # （窗口涵蓋60步，斷訊前的窗口尾端可能已經開始出現異常訊號）
        event_mask = (times >= gap_start - np.timedelta64(3, "h")) & (times <= gap_end)
        baseline_start = gap_start - np.timedelta64(7, "D") - np.timedelta64(3, "h")
        baseline_mask = (times >= baseline_start) & (times < gap_start - np.timedelta64(3, "h"))

        n_event = event_mask.sum()
        n_baseline = baseline_mask.sum()
        if n_event == 0:
            print(f"{pole} ({ev['gap_duration']}): 事件期沒有窗口資料可驗證（可能因缺口太大切不出窗口）")
            continue
        event_flagged = (errs[event_mask] > thresh).mean() * 100
        baseline_flagged = (errs[baseline_mask] > thresh).mean() * 100 if n_baseline > 0 else float("nan")

        print(f"\n{pole}  斷訊 {ev['gap_start']} ~ {ev['gap_end']}（{ev['gap_duration']}）")
        print(f"  事件期窗口數={n_event}, 被標記異常比例={event_flagged:.1f}%")
        print(f"  事件前一週正常期窗口數={n_baseline}, 同期誤判率={baseline_flagged:.1f}%")
        if n_baseline > 0:
            lift = event_flagged - baseline_flagged
            print(f"  差距（事件期 - 正常期基準）：{lift:+.1f} 個百分點")
