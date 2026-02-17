#%%
from ultralytics import YOLO
import torch

print("CUDA:", torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

model = YOLO("yolov8s-cls.pt")

model.train(
    data="/mnt/d/MAJOR PROJECT/new_processed/dataset_split/",  
    epochs=30,
    imgsz=224,
    batch=32,
    device=0,     # GPU
    workers=4
)


# %%
from ultralytics import YOLO

model = YOLO("/mnt/d/MAJOR PROJECT/old files/runs/classify/train3/weights/best.pt")

result = model("/mnt/d/MAJOR PROJECT/WIN_20250921_02_01_02_Pro.jpg")
print(result[0].probs)   # print class probabilities

# %%
import os

root = "/mnt/d/MAJOR PROJECT/processed_alphabets/dataset_split"

for path, dirs, files in os.walk(root):
    print(path)
    for d in dirs:
        print("  DIR :", d)
    for f in files:
        print("  FILE:", f)
    print("------")

# %%
import os
import shutil

root = "/mnt/d/MAJOR PROJECT/processed_alphabets/dataset_split"

for split in ["train", "val"]:
    folder = os.path.join(root, split)

    for item in os.listdir(folder):
        path = os.path.join(folder, item)

        # If it is a file, delete it
        if os.path.isfile(path):
            print("Removing stray file:", path)
            os.remove(path)

# %%
import torch
print(torch.cuda.is_available())

# %%
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

# %%
from ultralytics import YOLO

model = YOLO("yolov11s-cls.pt")   # should download automatically

# %%
