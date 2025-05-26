import json
import os

def load_metrics(filepath="model/metrics.json"):
    if not os.path.exists(filepath):
        print(f"No metrics file found at {filepath}")
        return None
    with open(filepath, "r") as f:
        metrics = json.load(f)
    return metrics

metrics = load_metrics()

if metrics:
    print("Saved training metrics:")
    print(f"Epochs trained: {metrics['epochs']}")
    print("Loss per epoch:")
    for i, loss in enumerate(metrics["epoch_losses"], 1):
        print(f"  Epoch {i}: {loss:.4f}")
    print("Evaluation metrics:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1 Score: {metrics['f1_score']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
else:
    print("No saved metrics found.")
