import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_models(models, x_val, y_val):
    results = {}
    for name, model in models.items():
        preds = model.predict(x_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        mae = mean_absolute_error(y_val, preds)
        r2 = r2_score(y_val, preds)
        results[name] = rmse
        print(f"{name}")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAE: {mae:.2f}")
        print(f"R2: {r2:.2f}\n\n")
        

    # Save results
    with open("outputs/model_results.txt", "w") as f:
        for name, rmse in results.items():
            f.write(f"{name}\n")
            f.write(f"RMSE: {rmse:.2f}\n")           
            f.write(f"MAE: {mae:.2f}\n")           
            f.write(f"R2: {r2:.2f}\n\n")           

    return results