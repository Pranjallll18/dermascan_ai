import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (classification_report, accuracy_score, f1_score,
                             recall_score, precision_score, confusion_matrix,
                             ConfusionMatrixDisplay)
from focal_loss import FocalLoss
from cnn_ctrnn_model import CNN_CTRNN
import numpy as np
import copy

# =====================================================================
#  CONFIG
# =====================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LR = 1e-4            # Initial learning rate
BATCH_SIZE = 32      # Smaller batch to fit multi-view sequences in memory

if device.type == 'cuda':
    SEQ_LEN = 4          # Multi-view sequence length (1 base + 3 augmented)
    EPOCHS = 30          # Max epochs (early stopping will likely cut this short)
    PATIENCE = 5         # Early stopping patience (epochs without val improvement)
    MIN_CLASS_SAMPLES = 800  # Moderate oversampling target per class
    MAX_CLASS_SAMPLES = 100000 # No limit on GPU
else:
    # CPU: Scale down to ensure training finishes in a reasonable time
    SEQ_LEN = 2          # Multi-view sequence length (1 base + 1 augmented)
    EPOCHS = 8           # Max epochs (early stopping will likely cut this short)
    PATIENCE = 2         # Early stopping patience (epochs without val improvement)
    MIN_CLASS_SAMPLES = 400  # Moderate oversampling target per class
    MAX_CLASS_SAMPLES = 400  # Downsample majority classes to perfectly balance training set

# =====================================================================
#  CLASS METADATA
# =====================================================================
CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
CLASS_LABELS = {name: idx for idx, name in enumerate(CLASS_NAMES)}
CLASS_DISPLAY = {
    'akiec': 'Actinic Keratosis',
    'bcc':   'Basal Cell Carcinoma',
    'bkl':   'Benign Keratosis',
    'df':    'Dermatofibroma',
    'mel':   'Melanoma',
    'nv':    'Melanocytic Nevus',
    'vasc':  'Vascular Lesion',
}
NUM_CLASSES = len(CLASS_NAMES)

# =====================================================================
#  TRANSFORMS
# =====================================================================
# Base transform — deterministic, used as first view in every sequence
base_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Augmentation transform — random, used for views 2..SEQ_LEN in training
aug_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.08),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Validation/test TTA transforms — deterministic augmentations for multi-view inference
# Each gives the CTRNN a different but reproducible perspective
tta_transforms = [
    # View 1: base (already handled separately)
    # View 2: horizontal flip
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    # View 3: vertical flip
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomVerticalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    # View 4: both flips
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.RandomVerticalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
]

# =====================================================================
#  METRICS UTILS
# =====================================================================
def save_metrics(metrics, filepath="model/metrics.json"):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {filepath}")

def load_metrics(filepath="model/metrics.json"):
    if not os.path.exists(filepath):
        return None
    with open(filepath, "r") as f:
        return json.load(f)

# =====================================================================
#  DATASET — Multi-view sequences for CTRNN
# =====================================================================
image_cache = {}

class HAM10000Dataset(Dataset):
    """
    Returns a sequence of SEQ_LEN views per image so the CTRNN actually
    aggregates across multiple perspectives of the same lesion.

    Training:  view_0 = base_transform, views_1..N = random augmentations
    Val/Test:  view_0 = base_transform, views_1..N = deterministic TTA
    """
    def __init__(self, dataframe, img_dir, seq_len=4, is_training=True):
        self.data = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.seq_len = seq_len
        self.is_training = is_training
        
        # Preload and resize unique images to memory to avoid disk I/O bottlenecks during training
        unique_ids = self.data['image_id'].unique()
        print(f"Preloading and resizing {len(unique_ids)} unique images for {'training' if is_training else 'validation/test'}...")
        for img_id in unique_ids:
            if img_id not in image_cache:
                img_path = os.path.join(self.img_dir, img_id + '.jpg')
                image_cache[img_id] = Image.open(img_path).convert("RGB").resize((224, 224))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_id = row['image_id']
        image = image_cache[img_id]
        label = CLASS_LABELS[row['dx']]

        views = [base_transform(image)]  # View 0 is always the clean base

        if self.is_training:
            # Random augmented views
            for _ in range(self.seq_len - 1):
                views.append(aug_transform(image))
        else:
            # Deterministic TTA views for val/test
            for tta in tta_transforms[:self.seq_len - 1]:
                views.append(tta(image))

        sequence = torch.stack(views)  # [seq_len, C, H, W]
        return sequence, label

# =====================================================================
#  LOAD DATA
# =====================================================================
csv_path = "../dataset/HAM10000_metadata.csv"
img_dir = "../dataset/HAM10000_images"
df = pd.read_csv(csv_path)

print(f"Dataset: {len(df)} images, {len(df['dx'].unique())} classes")
print(f"Original distribution:\n{df['dx'].value_counts().to_string()}\n")

# =====================================================================
#  STRATIFIED TRAIN / VAL / TEST SPLIT  (64% / 16% / 20%)
# =====================================================================
# First split: 80% train+val, 20% test
splitter1 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
trainval_idx, test_idx = next(splitter1.split(df, df['dx']))
trainval_df = df.iloc[trainval_idx].reset_index(drop=True)
test_df = df.iloc[test_idx].reset_index(drop=True)

# Second split: 80% train, 20% val (from trainval)
splitter2 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(splitter2.split(trainval_df, trainval_df['dx']))
train_df = trainval_df.iloc[train_idx].reset_index(drop=True)
val_df = trainval_df.iloc[val_idx].reset_index(drop=True)

print(f"Split sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

# =====================================================================
#  CLASS BALANCING (training set only)
#  Oversamples minority classes up to MIN_CLASS_SAMPLES.
#  Undersamples majority classes down to MAX_CLASS_SAMPLES (for fast CPU training).
# =====================================================================
original_train_counts = train_df['dx'].value_counts()
oversampled_dfs = []

print(f"\nDataset balancing (Target: {MIN_CLASS_SAMPLES} to {MAX_CLASS_SAMPLES} samples per class)")
for cls_name in CLASS_NAMES:
    cls_df = train_df[train_df['dx'] == cls_name]
    current = len(cls_df)

    if current < MIN_CLASS_SAMPLES:
        repeat = int(np.ceil(MIN_CLASS_SAMPLES / current))
        cls_up = pd.concat([cls_df] * repeat, ignore_index=True).head(MIN_CLASS_SAMPLES)
        oversampled_dfs.append(cls_up)
        print(f"  {cls_name:6s}: {current:4d} -> {MIN_CLASS_SAMPLES} (x{repeat})")
    elif current > MAX_CLASS_SAMPLES:
        cls_down = cls_df.sample(MAX_CLASS_SAMPLES, random_state=42)
        oversampled_dfs.append(cls_down)
        print(f"  {cls_name:6s}: {current:4d} -> {MAX_CLASS_SAMPLES} (downsampled)")
    else:
        oversampled_dfs.append(cls_df)
        print(f"  {cls_name:6s}: {current:4d}  (unchanged)")

train_df = pd.concat(oversampled_dfs, ignore_index=True)
train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
print(f"\nFinal training set: {len(train_df)} samples")

# =====================================================================
#  CLASS WEIGHTS — computed from ORIGINAL distribution (before oversampling)
# =====================================================================
orig_counts = df['dx'].value_counts()
total = orig_counts.sum()
class_weights = [total / (NUM_CLASSES * orig_counts[c]) for c in CLASS_NAMES]
# Clamp extreme weights to prevent instability
class_weights = [min(w, 10.0) for w in class_weights]
weights_tensor = torch.tensor(class_weights, dtype=torch.float)
print(f"\nClass weights (clamped): {dict(zip(CLASS_NAMES, [f'{w:.2f}' for w in class_weights]))}")

# =====================================================================
#  DATALOADERS
# =====================================================================
train_dataset = HAM10000Dataset(train_df, img_dir, seq_len=SEQ_LEN, is_training=True)
val_dataset   = HAM10000Dataset(val_df,   img_dir, seq_len=SEQ_LEN, is_training=False)
test_dataset  = HAM10000Dataset(test_df,  img_dir, seq_len=SEQ_LEN, is_training=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

# =====================================================================
#  MODEL, LOSS, OPTIMIZER, SCHEDULER
# =====================================================================
print(f"\nDevice: {device}")

model = CNN_CTRNN(num_classes=NUM_CLASSES).to(device)

if device.type == 'cpu':
    print("CPU detected. Freezing ResNet18 backbone to speed up training...")
    for param in model.cnn.parameters():
        param.requires_grad = False
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
else:
    optimizer = optim.Adam(model.parameters(), lr=LR)

criterion = FocalLoss(alpha=weights_tensor.to(device), gamma=2.0)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=2
)

# =====================================================================
#  TRAINING WITH VALIDATION & EARLY STOPPING
# =====================================================================
best_val_f1 = 0.0
best_model_state = None
epochs_no_improve = 0

train_losses = []
val_losses = []
val_f1_scores = []

print(f"\n{'='*60}")
print(f"  Training: {EPOCHS} max epochs, patience={PATIENCE}, seq_len={SEQ_LEN}")
print(f"{'='*60}\n")

for epoch in range(1, EPOCHS + 1):
    # ---------- TRAIN ----------
    model.train()
    running_loss = 0.0
    train_correct = 0
    train_total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        train_correct += (preds == labels).sum().item()
        train_total += labels.size(0)

    train_loss = running_loss / len(train_loader)
    train_acc = train_correct / train_total
    train_losses.append(train_loss)

    # ---------- VALIDATE ----------
    model.eval()
    val_running_loss = 0.0
    val_preds_all, val_labels_all = [], []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            val_preds_all.extend(preds.cpu().numpy())
            val_labels_all.extend(labels.cpu().numpy())

    val_loss = val_running_loss / len(val_loader)
    val_acc = accuracy_score(val_labels_all, val_preds_all)
    val_f1 = f1_score(val_labels_all, val_preds_all, average='macro', zero_division=0)
    val_losses.append(val_loss)
    val_f1_scores.append(val_f1)

    # Step the LR scheduler based on val F1
    scheduler.step(val_f1)
    current_lr = optimizer.param_groups[0]['lr']

    print(f"Epoch {epoch:2d}/{EPOCHS} | "
          f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.3f} | "
          f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.3f}, F1: {val_f1:.4f} | "
          f"LR: {current_lr:.1e}")

    # ---------- EARLY STOPPING & CHECKPOINTING ----------
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_model_state = copy.deepcopy(model.state_dict())
        epochs_no_improve = 0
        print(f"  [OK] New best model (val F1: {val_f1:.4f})")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print(f"\n  [NO IMPROVEMENT] Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
            break

actual_epochs = len(train_losses)

# =====================================================================
#  SAVE BEST MODEL
# =====================================================================
os.makedirs("model", exist_ok=True)
if best_model_state is not None:
    torch.save(best_model_state, "model/skin_cancer_model.pth")
    model.load_state_dict(best_model_state)  # Load best for evaluation
    print(f"\nBest model saved (val F1: {best_val_f1:.4f})")
else:
    torch.save(model.state_dict(), "model/skin_cancer_model.pth")
    print("\nModel saved (no improvement was recorded)")

# =====================================================================
#  EVALUATE ON TEST SET
# =====================================================================
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

display_names = [CLASS_DISPLAY[c] for c in CLASS_NAMES]

print(f"\n{'='*60}")
print("  TEST SET RESULTS")
print(f"{'='*60}")
print(classification_report(all_labels, all_preds, target_names=display_names, zero_division=0))

acc = accuracy_score(all_labels, all_preds)
f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
recall_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0)
precision_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0)

metrics = {
    "num_classes": NUM_CLASSES,
    "seq_len": SEQ_LEN,
    "class_names": CLASS_NAMES,
    "class_display_names": display_names,
    "epochs_trained": actual_epochs,
    "max_epochs": EPOCHS,
    "early_stopped": actual_epochs < EPOCHS,
    "best_val_f1": best_val_f1,
    "train_losses": train_losses,
    "val_losses": val_losses,
    "val_f1_scores": val_f1_scores,
    "test_accuracy": acc,
    "test_f1_macro": f1_macro,
    "test_f1_weighted": f1_weighted,
    "test_recall_macro": recall_macro,
    "test_precision_macro": precision_macro,
}
save_metrics(metrics)

# =====================================================================
#  PLOTS
# =====================================================================
os.makedirs("static", exist_ok=True)

fig, axes = plt.subplots(1, 3, figsize=(20, 5))

# 1. Training & Validation Loss
axes[0].plot(range(1, actual_epochs + 1), train_losses, 'o-', color='#4361ee', linewidth=2, label='Train Loss')
axes[0].plot(range(1, actual_epochs + 1), val_losses, 's-', color='#e74c3c', linewidth=2, label='Val Loss')
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_title("Training & Validation Loss")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 2. Validation F1 over epochs
axes[1].plot(range(1, actual_epochs + 1), val_f1_scores, 'D-', color='#27ae60', linewidth=2)
best_epoch = np.argmax(val_f1_scores) + 1
axes[1].axvline(x=best_epoch, color='#e74c3c', linestyle='--', alpha=0.7, label=f'Best (epoch {best_epoch})')
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("F1 Score (Macro)")
axes[1].set_title("Validation F1 Score")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# 3. Test performance bars
metric_names = ["Accuracy", "F1 (Macro)", "F1 (Weighted)", "Recall", "Precision"]
metric_values = [acc, f1_macro, f1_weighted, recall_macro, precision_macro]
colors = ["#4361ee", "#3f37c9", "#4895ef", "#4cc9f0", "#560bad"]
bars = axes[2].bar(metric_names, metric_values, color=colors)
axes[2].set_ylim(0, 1)
axes[2].set_title("Test Set Performance")
for bar, val in zip(bars, metric_values):
    axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig("static/performance_graph.png", dpi=150)
# plt.show()

# --------- Confusion Matrix ---------
cm = confusion_matrix(all_labels, all_preds)
fig, ax = plt.subplots(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_names)
disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
plt.title(f"Confusion Matrix - 7-Class (Best Model, Epoch {best_epoch})")
plt.tight_layout()
plt.savefig("static/confusion_matrix.png", dpi=150)
# plt.show()
plt.close(fig)

# --------- Summary ---------
unique, counts = np.unique(all_preds, return_counts=True)
pred_dist = {CLASS_DISPLAY[CLASS_NAMES[u]]: c for u, c in zip(unique, counts)}
print(f"\nPrediction distribution: {pred_dist}")
print(f"\nTraining complete. Best model from epoch {best_epoch} with val F1={best_val_f1:.4f}")
