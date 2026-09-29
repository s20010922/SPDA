"""階段5：LSTM-Autoencoder 訓練。
只用train集is_normal=True的窗口訓練，MSE loss，val集loss做early stopping。

支援 --seed 參數固定隨機性(PyTorch權重初始化、DataLoader洗牌)，讓結果可重現，
用來做多種子系統性比較(見 scripts/run_seed_sweep.py)，而不是每次重跑都碰運氣。
"""
import argparse
import os
import random
import sys

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

MODEL_DATASET_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model_dataset"
)
HIDDEN_SIZE = 32
BATCH_SIZE = 256
EPOCHS = 80  # 30輪時train/val loss都還在下降，拉高上限讓訓練跑到真正收斂
LR = 1e-3
PATIENCE = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 曾嘗試過LR=5e-4+ReduceLROnPlateau學習率衰減，訓練全程更平順但最終val_loss=0.170，
# 比固定LR=1e-3更差，故維持原設定。同一組LR=1e-3設定重跑6次val_loss落在0.123~0.176，
# 顯示訓練結果本身隨機性很大，且val_loss越低不代表閾值校準/異常偵測能力越好，
# 因此改用固定種子做系統性多種子比較，而非繼續盲目重訓。


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


def load_split(name: str, windows_dir: str = MODEL_DATASET_DIR):
    data = np.load(os.path.join(windows_dir, f"windows_{name}.npz"), allow_pickle=True)
    return data["X"].astype(np.float32), data["is_normal"], data["polename"], data["start_time"]


def standardize(X, mean, std):
    return (X - mean) / std


def compute_recon_error(model, X, device, batch_size=512):
    model.eval()
    errors = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.from_numpy(X[i:i + batch_size]).to(device)
            recon = model(batch)
            mse_per_window = ((recon - batch) ** 2).mean(dim=(1, 2))
            errors.append(mse_per_window.cpu().numpy())
    return np.concatenate(errors)


def set_seed(seed: int):
    """固定所有隨機性來源：Python內建random、numpy、PyTorch(CPU+CUDA)。
    DataLoader的shuffle=True也吃torch的全域RNG，固定後洗牌順序才可重現。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(seed: int, model_dir: str, windows_dir: str = MODEL_DATASET_DIR, hidden_size: int = HIDDEN_SIZE):
    set_seed(seed)
    os.makedirs(model_dir, exist_ok=True)

    print(f"使用裝置：{DEVICE}，random seed={seed}")
    X_train_all, is_normal_train, _, _ = load_split("train", windows_dir)
    X_val_all, is_normal_val, _, _ = load_split("val", windows_dir)
    print(f"train全部窗口數：{len(X_train_all)}，train正常窗口數：{is_normal_train.sum()}")
    print(f"val全部窗口數：{len(X_val_all)}，val正常窗口數：{is_normal_val.sum()}")

    X_train = X_train_all[is_normal_train]
    X_val_normal = X_val_all[is_normal_val]

    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True)
    std[std == 0] = 1.0
    np.savez(os.path.join(model_dir, "feature_scaler.npz"), mean=mean, std=std)

    X_train_s = standardize(X_train, mean, std)
    X_val_normal_s = standardize(X_val_normal, mean, std)

    # generator固定，確保DataLoader的shuffle順序在同一seed下每次重跑都一樣
    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train_s)),
                               batch_size=BATCH_SIZE, shuffle=True, generator=g)
    val_tensor = torch.from_numpy(X_val_normal_s).to(DEVICE)

    model = LSTMAutoencoder(n_features=X_train.shape[2], hidden_size=hidden_size).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_state = None
    best_epoch = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []
        for (batch,) in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(val_tensor), val_tensor).item()

        train_loss = float(np.mean(train_losses))
        print(f"[epoch {epoch:02d}] train_loss={train_loss:.5f}  val_loss={val_loss:.5f}")

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"val_loss連續{PATIENCE}輪沒改善，提前停止")
                break

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), os.path.join(model_dir, "lstm_autoencoder.pt"))
    print(f"最佳 val_loss={best_val_loss:.5f}（第{best_epoch}輪），模型已存至 {model_dir}/lstm_autoencoder.pt")
    return best_val_loss, best_epoch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="固定隨機種子，確保結果可重現")
    parser.add_argument("--model-dir", type=str, default=MODEL_DATASET_DIR,
                         help="模型與log輸出目錄，多種子比較時各自指定獨立目錄避免互相覆蓋")
    parser.add_argument("--windows-dir", type=str, default=MODEL_DATASET_DIR,
                         help="windows_*.npz來源目錄，用來跑不同資料清洗版本的對照實驗")
    parser.add_argument("--hidden-size", type=int, default=HIDDEN_SIZE,
                         help="LSTM隱藏層大小，用來跑不同模型容量的超參數對照實驗")
    args = parser.parse_args()
    main(args.seed, args.model_dir, args.windows_dir, args.hidden_size)
