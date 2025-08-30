import matplotlib.pyplot as plt
import seaborn as sns

def plot_actual_vs_pred(y_val, pred, save_path=None):
    plt.figure(figsize=(6,6))
    plt.scatter(y_val, pred, alpha=0.6, color="blue")
    plt.plot([y_val.min(), y_val.max()], [pred.min(), pred.max()], 'r--')
    plt.xlabel("Actural Values")
    plt.ylabel("Predicted Values")
    plt.title("Predicted vs Actual")
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_residuals(y_val, pred, save_path=None):
    residuals = y_val - pred
    plt.figure(figsize=(8,5))
    sns.scatterplot(x=pred, y=residuals, alpha=0.7)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    if save_path:
        plt.savefig(save_path)
    plt.show()
