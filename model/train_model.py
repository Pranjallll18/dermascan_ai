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
from sklearn.metrics import classification_report, accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, ConfusionMatrixDisplay
from focal_loss import FocalLoss
from cnn_ctrnn_model import CNN_CTRNN  # Your CNN-CTRNN model
import  numpy as np

# --------- Save/load metrics ---------
def save_metrics(metrics, filepath="model/metrics.json"):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(metrics, f)
    print(f"Metrics saved to {filepath}")

def load_metrics(filepath="model/metrics.json"):
    if not os.path.exists(filepath):
        return None
    with open(filepath, "r") as f:
        metrics = json.load(f)
    return metrics

# --------- Dataset ---------
class HAM10000Dataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None, augment_malignant=False):
        self.data = dataframe
        self.img_dir = img_dir
        self.transform = transform
        self.augment_malignant = augment_malignant
        self.label_map = {"benign": 0, "malignant": 1}
        self.augment_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(40),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_dir, row['image_id'] + '.jpg')
        image = Image.open(img_path).convert("RGB")
        label_str = row['label']
        label = self.label_map[label_str]

        if self.augment_malignant and label_str == "malignant":
            image = self.augment_transform(image)
        else:
            image = self.transform(image)

        image = image.unsqueeze(0)  # Add sequence dimension
        return image, label

# --------- Transforms ---------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --------- Load CSV ---------
csv_path = "../dataset/HAM10000_metadata.csv"
img_dir = "../dataset/HAM10000_images"
df = pd.read_csv(csv_path)
df['label'] = df['dx'].apply(lambda x: 'malignant' if x in ['mel', 'bcc'] else 'benign')

# --------- Stratified Split ---------
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(splitter.split(df, df['label']))
train_df = df.iloc[train_idx].reset_index(drop=True)
test_df = df.iloc[test_idx].reset_index(drop=True)

# --------- Manual Oversampling ---------
malignant_df = train_df[train_df['label'] == 'malignant']
benign_df = train_df[train_df['label'] == 'benign']

# Increase malignant samples by duplicating them 3 times
malignant_oversampled = pd.concat([malignant_df] * 10, ignore_index=True)

# Combine and shuffle
train_df = pd.concat([benign_df, malignant_oversampled], ignore_index=True)
train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

# --------- Class Weights ---------
class_counts = train_df['label'].value_counts()
total = class_counts.sum()
weights = torch.tensor([total / class_counts['benign'], total / class_counts['malignant']], dtype=torch.float)

# --------- Datasets & Loaders ---------
train_dataset = HAM10000Dataset(train_df, img_dir, transform, augment_malignant=True)
test_dataset = HAM10000Dataset(test_df, img_dir, transform, augment_malignant=False)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# --------- Model, Loss, Optimizer ---------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN_CTRNN().to(device)
weights =  torch.tensor([1.0, 4.0])
criterion = FocalLoss(alpha=weights.to(device), gamma=2.0)
optimizer = optim.Adam(model.parameters(), lr=1e-5)

# --------- Load Previous Metrics (Optional) ---------
saved_metrics = load_metrics()
if saved_metrics:
    print("Previous Training:")
    print(saved_metrics)

# --------- Training ---------
EPOCHS = 10
epoch_losses = []

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    epoch_losses.append(avg_loss)
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")

# --------- Save Model ---------
os.makedirs("model", exist_ok=True)
torch.save(model.state_dict(), "model/skin_cancer_model.pth")

# --------- Evaluation ---------
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# --------- Metrics ---------
print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=["Benign", "Malignant"]))

acc = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds)

metrics = {
    "epochs": EPOCHS,
    "epoch_losses": epoch_losses,
    "accuracy": acc,
    "f1_score": f1,
    "recall": recall,
    "precision": precision
}
save_metrics(metrics)

# --------- Plot Metrics ---------
os.makedirs("static", exist_ok=True)
plt.figure(figsize=(8, 5))
plt.bar(["Accuracy", "F1 Score", "Recall", "Precision"], [acc, f1, recall, precision], color="skyblue")
plt.ylim(0, 1)
plt.title("Model Performance")
plt.savefig("static/performance_graph.png")
plt.show()

# --------- Confusion Matrix ---------
cm = confusion_matrix(all_labels, all_preds)
fig, ax = plt.subplots(figsize=(6, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Benign", "Malignant"])
disp.plot(ax=ax, cmap='Blues')
plt.title("Confusion Matrix")
plt.savefig("static/confusion_matrix.png")
plt.show()
plt.close(fig)

unique, counts = np.unique(all_preds, return_counts=True)
print(dict(zip(unique, counts)))
