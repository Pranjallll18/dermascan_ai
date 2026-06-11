import os
import pandas as pd
from PIL import Image
import time

csv_path = "dataset/HAM10000_metadata.csv"
img_dir = "dataset/HAM10000_images"

df = pd.read_csv(csv_path).head(200)  # Test on 200 images

print("Starting benchmark of loading and resizing 200 images...")
t0 = time.time()
cache = {}
for img_id in df['image_id']:
    img_path = os.path.join(img_dir, img_id + '.jpg')
    img = Image.open(img_path).convert("RGB").resize((224, 224))
    cache[img_id] = img
t1 = time.time()

elapsed = t1 - t0
print(f"Loaded and resized 200 images in {elapsed:.3f} seconds.")
print(f"Average time per image: {elapsed/200*1000:.2f} ms")
print(f"Estimated time for all 10,015 images: {elapsed/200*10015:.2f} seconds")
