import pandas as pd
import math

def compute_accuracy_ci(csv_path, confidence=0.95):
    df = pd.read_csv(csv_path)

    correct_predictions = (df['true_label'] == df['predicted_label']).sum()
    total_predictions = len(df)
    accuracy = correct_predictions / total_predictions

    se = math.sqrt((accuracy * (1 - accuracy)) / total_predictions)

    # Z-score for desired confidence level
    z = 1.96 if confidence == 0.95 else {
        0.90: 1.645,
        0.99: 2.576
    }.get(confidence, 1.96)  # default to 1.96 if unknown confidence

    # Confidence interval
    lower = accuracy - z * se
    upper = accuracy + z * se

    # Clamp to [0, 1]
    lower = max(0.0, lower)
    upper = min(1.0, upper)

    return {
        "accuracy": round(accuracy, 4),
        f"{int(confidence * 100)}%_ci": (round(lower, 4), round(upper, 4))
    }


score_ci = compute_accuracy_ci('result_file_linear224.csv')
score_ci