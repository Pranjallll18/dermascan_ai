import torch
import torchvision.models as models
import time

device = torch.device("cpu")
model = models.resnet18().to(device)
model.eval()

# Batch size 32, seq_len 4 means 128 images of size 224x224x3
x = torch.randn(128, 3, 224, 224).to(device)

print("Starting benchmark...")
t0 = time.time()
with torch.no_grad():
    for _ in range(5):
        _ = model(x)
t1 = time.time()

avg_time = (t1 - t0) / 5
print(f"Average time for 1 batch of 128 images: {avg_time:.3f} seconds")
