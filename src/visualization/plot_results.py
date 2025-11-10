"""
Plotting utilities for model evaluation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_actual_vs_predicted(
    y_true: Iterable[float],
    y_pred: Iterable[float],
    save_path: Optional[Path] = None,
    show: bool = True,
    title: str = "Actual vs Predicted",
):
    """
    Scatter plot comparing actual targets to predicted values.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=y_true_arr, y=y_pred_arr, ax=ax, color="dodgerblue", s=60, alpha=0.7)
    line_min = min(y_true_arr.min(), y_pred_arr.min())
    line_max = max(y_true_arr.max(), y_pred_arr.max())
    ax.plot([line_min, line_max], [line_min, line_max], linestyle="--", color="firebrick", label="Ideal")
    ax.set_title(title)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_residuals(
    y_true: Iterable[float],
    y_pred: Iterable[float],
    save_path: Optional[Path] = None,
    show: bool = True,
    title: str = "Residual Distribution",
):
    """
    Plot histogram and KDE of residuals (y_true - y_pred).
    """
    residuals = np.asarray(y_true) - np.asarray(y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(residuals, bins=25, kde=True, ax=ax, color="slateblue", alpha=0.7)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Residual")
    ax.set_ylabel("Frequency")
    fig.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig
