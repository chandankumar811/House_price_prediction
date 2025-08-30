from src.data_preprocessing import load_data, preprocess_data
from src.train_model import train_models
from src.evaluate import evaluate_models
from src.visualization import plot_residuals, plot_actual_vs_pred

def main():
    train, test = load_data("data/train.csv", "data/test.csv")
    x_train, x_val, y_train, y_val, test_processed = preprocess_data(train, test)

    models = train_models(x_train, y_train)
    results = evaluate_models(models, x_val, y_val)

    print("✅ Training Complete! Model preformance:", results)

    best_model_name = min(results, key=results.get)
    best_model = models[best_model_name]
    print(f"🏆 Best Model: {best_model_name}")





    pred = best_model.predict(x_val)
    plot_actual_vs_pred(y_val, pred, save_path="outputs/actual_vs_pred.png")
    plot_residuals(y_val, pred, save_path="outputs/residulas.png")

if __name__ == "__main__":
    main()