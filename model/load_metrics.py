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
    print("=" * 60)
    print("  Saved Training Metrics")
    print("=" * 60)

    print(f"\n  Classes: {metrics.get('num_classes', 'N/A')}")
    print(f"  Sequence length: {metrics.get('seq_len', 1)}")
    if 'class_display_names' in metrics:
        for name in metrics['class_display_names']:
            print(f"    - {name}")

    trained = metrics.get('epochs_trained', metrics.get('epochs', '?'))
    max_ep = metrics.get('max_epochs', trained)
    stopped = metrics.get('early_stopped', False)
    print(f"\n  Epochs trained: {trained}/{max_ep}" + (" (early stopped)" if stopped else ""))
    print(f"  Best val F1: {metrics.get('best_val_f1', 'N/A')}")

    if 'train_losses' in metrics:
        print("\n  Training Loss per epoch:")
        for i, loss in enumerate(metrics["train_losses"], 1):
            val_loss = metrics.get('val_losses', [None] * i)[i - 1]
            val_f1 = metrics.get('val_f1_scores', [None] * i)[i - 1]
            line = f"    Epoch {i:2d}: train={loss:.4f}"
            if val_loss is not None:
                line += f", val={val_loss:.4f}"
            if val_f1 is not None:
                line += f", val_f1={val_f1:.4f}"
            print(line)
    elif 'epoch_losses' in metrics:
        print("\n  Loss per epoch:")
        for i, loss in enumerate(metrics["epoch_losses"], 1):
            print(f"    Epoch {i}: {loss:.4f}")

    print("\n  Test Set Metrics:")
    print(f"    Accuracy:          {metrics.get('test_accuracy', metrics.get('accuracy', 0)):.4f}")
    print(f"    F1 (Macro):        {metrics.get('test_f1_macro', metrics.get('f1_macro', 0)):.4f}")
    print(f"    F1 (Weighted):     {metrics.get('test_f1_weighted', metrics.get('f1_weighted', 0)):.4f}")
    print(f"    Recall (Macro):    {metrics.get('test_recall_macro', metrics.get('recall_macro', 0)):.4f}")
    print(f"    Precision (Macro): {metrics.get('test_precision_macro', metrics.get('precision_macro', 0)):.4f}")
    print("=" * 60)
else:
    print("No saved metrics found.")
