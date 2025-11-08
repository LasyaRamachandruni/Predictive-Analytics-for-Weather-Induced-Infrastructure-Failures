"""
plot_results.py
---------------
Generates simple visualizations for model evaluation.
"""

import matplotlib.pyplot as plt
import seaborn as sns


def plot_actual_vs_predicted(y_test, y_pred):
    """Plot Actual vs Predicted values."""
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_pred, color="dodgerblue", s=70, alpha=0.7)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)],
             color="red", linestyle="--", label="Perfect Prediction")
    plt.title("Actual vs Predicted Infrastructure Failures")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show(block=False)
plt.pause(3)
plt.close()

