from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

RUN_DIR = Path("runs/generated_entity")
OUT_DIR = RUN_DIR / "presentation_plots_clean"
OUT_DIR.mkdir(exist_ok=True)

epoch_df = pd.read_csv(RUN_DIR / "epoch_loss.csv")
step_df = pd.read_csv(RUN_DIR / "loss_log.csv")


def save_plot(x, y, title, xlabel, ylabel, filename, marker=None):
    plt.figure(figsize=(9, 5))
    plt.plot(x, y, marker=marker)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{filename}.png", dpi=300)
    plt.savefig(OUT_DIR / f"{filename}.svg")
    plt.close()


# 1. Prediction loss by epoch
save_plot(
    epoch_df["epoch"],
    epoch_df["pred_loss"],
    "Prediction Loss by Epoch",
    "Epoch",
    "Prediction loss",
    "pred_loss_by_epoch_clean",
    marker="o",
)

# 2. SigReg loss by epoch
save_plot(
    epoch_df["epoch"],
    epoch_df["sigreg_loss"],
    "SigReg Loss by Epoch",
    "Epoch",
    "SigReg loss",
    "sigreg_loss_by_epoch_clean",
    marker="o",
)

# 3. Decoder loss by epoch
save_plot(
    epoch_df["epoch"],
    epoch_df["decoded_loss"],
    "Decoder Loss by Epoch",
    "Epoch",
    "Decoder loss",
    "decoder_loss_by_epoch_clean",
    marker="o",
)

# 4. Smoothed prediction loss by step
window = 50
step_df["pred_loss_smooth"] = step_df["pred_loss"].rolling(window=window, min_periods=1).mean()

save_plot(
    step_df["step"],
    step_df["pred_loss_smooth"],
    f"Smoothed Prediction Loss by Step, Rolling {window}",
    "Training step",
    "Prediction loss",
    "smoothed_pred_loss_by_step_clean",
)