# %%
import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import torch


def plot_lr(df, fig_dir):
    plt.figure()
    plt.plot(df["train_lr"])
    plt.xlabel("epoch")
    plt.ylabel("learning rate")
    plt.savefig(os.path.join(fig_dir, "train_lr.png"))
    plt.close()


def plot_metrics(df, fig_dir, log_key):
    plt.figure()
    plt.plot(df[f"train_{log_key}"], label="train")
    plt.plot(df[f"test_{log_key}"], label="val")
    plt.xlabel("epoch")
    plt.ylabel(log_key)
    plt.legend()
    plt.savefig(os.path.join(fig_dir, f"{log_key}.png"))
    plt.close()


def main(report_dir):
    log_keys = [
        "class_error",
        "loss",
        "loss_bbox",
        "loss_ce",
        "loss_giou",
        "cardinality_error_unscaled",
    ]
    log_file = os.path.join(report_dir, "log.txt")
    with open(log_file) as f:
        logs = f.readlines()
    logs = [eval(log) for log in logs]
    df = pd.DataFrame(logs)
    fig_dir = os.path.join(report_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    plot_lr(df, fig_dir)
    log_key = log_keys[1]
    for log_key in log_keys:
        plot_metrics(df, fig_dir, log_key)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Plot training logs")
    parser.add_argument("--report-dir", default=str)
    args = parser.parse_args()
    main(args.report_dir)
# %%
