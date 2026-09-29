"""多種子系統性比較：固定種子跑N次訓練+評估，彙整成一份表格，
取代先前「重跑碰運氣、挑好看數字」的做法。

每個種子的模型/評估檔案存在獨立子目錄 model_dataset/seed_<N>/，
互不覆蓋，方便事後追溯查證。
"""
import argparse
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DATASET_DIR = os.path.join(ROOT, "model_dataset")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step5_train_autoencoder as train_mod
import step6_evaluate as eval_mod

SEEDS = [1, 42, 123, 2024, 88888]


def main(hidden_size: int = 32, out_dir: str = MODEL_DATASET_DIR, windows_dir: str = None):
    windows_dir = windows_dir or MODEL_DATASET_DIR
    rows = []
    for seed in SEEDS:
        seed_dir = os.path.join(out_dir, f"seed_{seed}")
        print(f"\n{'='*60}\n訓練 seed={seed}，hidden_size={hidden_size}，windows_dir={windows_dir}，輸出至 {seed_dir}\n{'='*60}")

        best_val_loss, best_epoch = train_mod.main(
            seed=seed, model_dir=seed_dir, windows_dir=windows_dir, hidden_size=hidden_size
        )

        print(f"\n---評估 seed={seed}---")
        comparison, summary, known_events = eval_mod.evaluate(
            model_dir=seed_dir, windows_dir=windows_dir, verbose=False
        )

        # 彙整各已知事件的event_flagged_pct成一個字典欄位，事件期沒有窗口的(outage類)記為None
        event_ratios = {
            f"{ev['polename']}_{ev['event_type']}_flagged_pct": ev["event_flagged_pct"]
            for ev in known_events
        }
        rows.append({
            "seed": seed,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "per_pole_avg_flagged_pct": summary["per_pole_avg_flagged"],
            "global_avg_flagged_pct": summary["global_avg_flagged"],
            **event_ratios,
        })
        print(f"seed={seed} 完成：val_loss={best_val_loss:.5f}, "
              f"每杆體閾值法超標={summary['per_pole_avg_flagged']:.2f}%, "
              f"已知事件偵測={event_ratios}")

    result_df = pd.DataFrame(rows)
    result_df["dist_from_target"] = (result_df["per_pole_avg_flagged_pct"] - 1.0).abs()
    result_df = result_df.sort_values("dist_from_target")

    print(f"\n{'='*60}\n=== 多種子比較彙總（按每杆體閾值法離1%目標的距離排序）===\n{'='*60}")
    print(result_df.to_string(index=False))

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "seed_sweep_results.csv")
    result_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n已存至 {out_path}")

    best_seed = int(result_df.iloc[0]["seed"])
    print(f"\n閾值校準最準的種子：seed={best_seed}"
          f"（每杆體閾值法超標比例={result_df.iloc[0]['per_pole_avg_flagged_pct']:.2f}%，"
          f"離1%目標僅{result_df.iloc[0]['dist_from_target']:.2f}個百分點）")
    return result_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--out-dir", type=str, default=MODEL_DATASET_DIR)
    parser.add_argument("--windows-dir", type=str, default=None)
    args = parser.parse_args()
    main(hidden_size=args.hidden_size, out_dir=args.out_dir, windows_dir=args.windows_dir)
