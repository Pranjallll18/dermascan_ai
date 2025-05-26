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
from sklearn.metrics import classification_report, accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, ConfusionMatrixDisplay
from cnn_ctrnn_model import CNN_CTRNN  # Import your model here

# Helper functions to save/load metrics
def save_metrics(metrics, filepath="model/metrics.json"):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(metrics, f)
    print(f"Metrics saved to {filepath}")

def load_metrics(filepath="model/metrics.json"):
    if not os.path.exists(filepath):
        print(f"No metrics file found at {filepath}")
        return None
    with open(filepath, "r") as f:
        metrics = json.load(f)
    print(f"Metrics loaded from {filepath}")
    return metrics


class HAM10000Dataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.label_map = {"benign": 0, "malignant": 1}
        self.data['label'] = self.data['dx'].apply(lambda x: 'malignant' if x in ['mel', 'bcc'] else 'benign')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data.iloc[idx, 1] + ".jpg")
        image = Image.open(img_name).convert("RGB")
        label = self.label_map[self.data.loc[idx, "label"]]
        if self.transform:
            image = self.transform(image)
        image = image.unsqueeze(0)  # Add sequence dimension for CTRNN
        return image, label


# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load Dataset
dataset = HAM10000Dataset(
    csv_file="../dataset/HAM10000_metadata.csv",
    img_dir="../dataset/HAM10000_images",
    transform=transform
)

# Calculate class weights
labels = [dataset[i][1] for i in range(len(dataset))]
benign_count = labels.count(0)
malignant_count = labels.count(1)
total = benign_count + malignant_count
weights = torch.tensor([total / benign_count, total / malignant_count], dtype=torch.float)

# Train/Test Split
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# Data Loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = weights.to(device)

# Load previous metrics if exist
saved_metrics = load_metrics()
if saved_metrics:
    print("Previous training info:")
    print(f"Epochs trained: {saved_metrics['epochs']}")
    print("Loss per epoch:")
    for i, loss in enumerate(saved_metrics["epoch_losses"], 1):
        print(f"  Epoch {i}: {loss:.4f}")
    print("Evaluation metrics:")
    print(f"  Accuracy: {saved_metrics['accuracy']:.4f}")
    print(f"  F1 Score: {saved_metrics['f1_score']:.4f}")
    print(f"  Recall: {saved_metrics['recall']:.4f}")
    print(f"  Precision: {saved_metrics['precision']:.4f}")
    print("------------------------------------------------")

# Model, loss, optimizer
model = CNN_CTRNN()
model.to(device)
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
EPOCHS = 20
epoch_losses = []

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    epoch_losses.append(avg_loss)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

# Save model
os.makedirs("model", exist_ok=True)
torch.save(model.state_dict(), "model/skin_cancer_model.pth")

# Evaluation
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Metrics
print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=["Benign", "Malignant"]))

acc = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds)

print(f"Accuracy: {acc:.4f}, F1 Score: {f1:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}")

# Save all info
metrics = {
    "epochs": EPOCHS,
    "epoch_losses": epoch_losses,
    "accuracy": acc,
    "f1_score": f1,
    "recall": recall,
    "precision": precision
}

save_metrics(metrics)

# Plot performance graph
os.makedirs("static", exist_ok=True)
plt.figure(figsize=(8, 5))
metrics_vals = [acc, f1, recall, precision]
labels = ["Accuracy", "F1 Score", "Recall", "Precision"]
plt.bar(labels, metrics_vals, color="skyblue")
plt.ylim(0, 1)
plt.title("Model Performance Metrics")
plt.savefig("static/performance_graph.png")
plt.show()

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)

fig, ax = plt.subplots(figsize=(6, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Benign", "Malignant"])
disp.plot(ax=ax, cmap='Blues')
plt.title("Confusion Matrix")
plt.savefig("static/confusion_matrix.png")
plt.show()
plt.close(fig)  # Close figure explicitly to free memory

