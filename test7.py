
#%%
import cv2
import mediapipe as mp
import os
from tqdm import tqdm
from PIL import Image
# %%

# --- Paths ---
INPUT_DIR = "/mnt/d/MAJOR PROJECT/new dataset/alphabets"   # your captured images
OUTPUT_DIR = "/mnt/d/MAJOR PROJECT/processed_alphabets/dataset"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# %%
# --- Mediapipe Setup ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

# %%
def preprocess_image(img_path, save_path):
    img = cv2.imread(img_path)
    if img is None:
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    h, w, _ = img.shape

    if not results.multi_hand_landmarks:
        # No hand detected – keep original or skip
        cv2.imwrite(save_path, img)
        return

    # bounding box around both hands
    x_min, y_min, x_max, y_max = w, h, 0, 0
    for hand_landmarks in results.multi_hand_landmarks:
        for lm in hand_landmarks.landmark:
            x, y = int(lm.x * w), int(lm.y * h)
            x_min, y_min = min(x_min, x), min(y_min, y)
            x_max, y_max = max(x_max, x), max(y_max, y)

    # --- improved padding ---
    pad_x, pad_y = int(0.15 * w), int(0.15 * h)
    x_min, y_min = max(0, x_min - pad_x), max(0, y_min - pad_y)
    x_max, y_max = min(w, x_max + pad_x), min(h, y_max + pad_y)

    # --- sanity check: expand if too small ---
    box_w, box_h = x_max - x_min, y_max - y_min
    if box_w * box_h < 0.1 * w * h:
        x_min = max(0, x_min - int(0.2 * w))
        y_min = max(0, y_min - int(0.2 * h))
        x_max = min(w, x_max + int(0.2 * w))
        y_max = min(h, y_max + int(0.2 * h))

    cropped = img[y_min:y_max, x_min:x_max]
    cropped = cv2.resize(cropped, (224, 224))

    cv2.imwrite(save_path, cropped)

# %%

# --- Process the whole dataset ---
for root, _, files in os.walk(INPUT_DIR):
    rel_path = os.path.relpath(root, INPUT_DIR)
    out_dir = os.path.join(OUTPUT_DIR, rel_path)
    os.makedirs(out_dir, exist_ok=True)

    for file in tqdm(files, desc=f"Processing {rel_path}"):
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            in_path = os.path.join(root, file)
            out_path = os.path.join(out_dir, file)
            preprocess_image(in_path, out_path)

print("✅ All images preprocessed and saved to:", OUTPUT_DIR)
# %%
import os, shutil
from sklearn.model_selection import train_test_split

# Input paths
base_dirs = [
    "/mnt/d/MAJOR PROJECT/processed_alphabets/dataset"
]

# Output path
output_dir = "/mnt/d/MAJOR PROJECT/processed_alphabets/dataset_split"
splits = ["train", "val", "test"]

# Function to split each class
def split_and_copy(class_dir, class_name):
    images = os.listdir(class_dir)
    images = [img for img in images if img.lower().endswith(('.jpg','.png','.jpeg'))]

    # 70/15/15 split
    train_imgs, temp_imgs = train_test_split(images, test_size=0.3, random_state=42)
    val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=42)

    split_data = {"train": train_imgs, "val": val_imgs, "test": test_imgs}

    for split, img_list in split_data.items():
        split_dir = os.path.join(output_dir, split, class_name)
        os.makedirs(split_dir, exist_ok=True)
        for img in img_list:
            src = os.path.join(class_dir, img)
            dst = os.path.join(split_dir, img)
            shutil.copy(src, dst)

# Process both alphabets and numbers
for base_dir in base_dirs:
    for class_name in os.listdir(base_dir):
        class_dir = os.path.join(base_dir, class_name)
        if os.path.isdir(class_dir):
            print(f"Splitting class: {class_name}")
            split_and_copy(class_dir, class_name)

print("✅ Dataset split into train/val/test at:", output_dir)

# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
# %%
DATA_DIR = "/mnt/d/MAJOR PROJECT/processed_alphabets/dataset_split"  # root folder that has train/val/test
BATCH_SIZE = 32      # safe for 4GB GPU, increase if you have more VRAM
NUM_CLASSES = 26       # 26 alphabets + 10 numbers
EPOCHS = 70
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# %%
# DATA TRANSFORMS
# ===============================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.GaussianBlur(3),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
# %%
# LOAD DATASETS
# ===============================
train_dataset = datasets.ImageFolder(root=f"{DATA_DIR}/train", transform=train_transform)
val_dataset   = datasets.ImageFolder(root=f"{DATA_DIR}/val", transform=eval_transform)
test_dataset  = datasets.ImageFolder(root=f"{DATA_DIR}/test", transform=eval_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# %%
# MODEL
# ===============================
model = models.resnet50(pretrained=True)
for param in model.parameters():
    param.requires_grad = False  # freeze backbone

# Replace final layer
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, NUM_CLASSES)

model = model.to(DEVICE)

for name, param in model.named_parameters():
    if "layer4" in name or "fc" in name:
        param.requires_grad = True
# %%
# LOSS + OPTIMIZER
# ===============================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam([
    {'params': model.layer3.parameters(), 'lr': 1e-5},
    {'params': model.layer4.parameters(), 'lr': 1e-5},
    {'params': model.fc.parameters(), 'lr': 1e-4}
])
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)


# For mixed precision
scaler = torch.cuda.amp.GradScaler()
# %%
# TRAINING LOOP
# ===============================
for epoch in range(EPOCHS):
    model.train()
    train_loss, correct, total = 0, 0, 0
    
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    train_acc = 100. * correct / total
# Save trained model weights
torch.save(model.state_dict(), "alphabet_resnet50_signlang2.pth")
print("✅ Training complete. Model saved as alphabet_resnet50_signlang2.pth")

# %%
    model.eval()
    val_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    val_acc = 100. * correct / total

    print(f"Epoch [{epoch+1}/{EPOCHS}] "
          f"Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}% | "
          f"Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%")


# %%
# FINAL TEST EVALUATION
# ===============================
model.eval()
test_loss, correct, total = 0, 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    print(f"Epoch [{epoch+1}/{EPOCHS}] | "
          f"Train Loss: {train_loss/len(train_loader):.4f} | "
          f"Train Acc: {train_acc:.2f}% | "
          f"Val Loss: {val_loss/len(val_loader):.4f} | "
          f"Val Acc: {val_acc:.2f}%")
        
# %%
test_acc = 100. * correct / total
print(f"\n✅ Final Test Accuracy: {test_acc:.2f}% | Test Loss: {test_loss/len(test_loader):.4f}")

torch.save(model.state_dict(), "alphabet_resnet50_signlang2.pth")
print("✅ Training complete. Model saved as alphabet_resnet50_signlang2.pth")
#%%
import torch
import json
import numpy as np
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------------------
# Load class mapping correctly
# -------------------------------
with open("alphabet_class_to_idx.json", "r") as f:
    alphabet_class_to_idx = json.load(f)

idx_to_class = {v: k for k, v in alphabet_class_to_idx.items()}
num_classes = len(idx_to_class)

# -------------------------------
# Build target_names in correct order
# -------------------------------
target_names = [idx_to_class[i] for i in range(num_classes)]
print(target_names)   # Just to verify order

# -------------------------------
# 2. Load model
# -------------------------------
model = models.resnet50(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 26)
model.load_state_dict(torch.load("alphabet_resnet50_signlang2.pth", map_location="cpu"))
model.eval()

# -------------------------------
# 3. Test loader
# -------------------------------
DATA_DIR = "/mnt/d/MAJOR PROJECT/processed_alphabets/dataset_split"   #  <<–– CHANGE THIS
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

test_dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# -------------------------------
# 4. Evaluate model
# -------------------------------
all_labels = []
all_preds = []

with torch.no_grad():
    for imgs, labels in test_loader:
        outputs = model(imgs)
        _, preds = torch.max(outputs, 1)
        
        all_labels.extend(labels.numpy())
        all_preds.extend(preds.numpy())

# -------------------------------
# 5. Classification Report
# -------------------------------

# Sort classes by their assigned index value
target_names = [k for k, v in sorted(alphabet_class_to_idx.items(), key=lambda x: x[1])]

report = classification_report(all_labels, all_preds, target_names=target_names)
print("\n===== CLASSIFICATION REPORT =====\n")
print(report)

# -------------------------------
# 6. Confusion Matrix
# -------------------------------
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=target_names, yticklabels=target_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

print("\nReport saved as classification_report.txt")
print("Confusion matrix saved as confusion_matrix.png")

%%
import pandas as pd

report_dict = classification_report(all_labels, all_preds, target_names=target_names, output_dict=True)
df = pd.DataFrame(report_dict).transpose()

weak_classes = df[df['f1-score'] < 0.85]   # threshold you choose
print("\n===== WEAK CLASSES (<0.85 F1-score) =====\n")
print(weak_classes)

#%%
# Fine-tune existing weights
model = models.resnet50(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 26)
model.load_state_dict(torch.load("resnet101_signlang8.pth"))
model.to("cuda")

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)
criterion = torch.nn.CrossEntropyLoss()

# train_loader = your dataloader for /train
for epoch in range(3):  # fine-tune for 2–3 epochs
    model.train()
    for imgs, labels in train_loader:
        imgs, labels = imgs.cuda(), labels.cuda()
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
    print("epoch", epoch, "done")

#%%
torch.save(model.state_dict(), "resnet101_signlang3_ft.pth")


# %%
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
import mediapipe as mp
import numpy as np




# %%
import cv2
import torch
import mediapipe as mp
from PIL import Image
from torchvision import models, transforms
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# 1. Rebuild the model architecture
# -------------------------------
num_classes = 36  # 26 alphabets + 10 digits
model = models.resnet101(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

# Load weights
MODEL_PATH = "resnet101_signlang.pth"
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()

# -------------------------------
# 2. Define class names
# -------------------------------
class_names = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 
    'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 
    'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
]

# -------------------------------
# 3. Define preprocessing (clean)
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -------------------------------
# 4. Load the image
# -------------------------------
image_path = "/mnt/d/MAJOR PROJECT/WIN_20250921_02_01_02_Pro.jpg"
image_bgr = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# -------------------------------
# 5. Hand detection with MediaPipe
# -------------------------------
mp_hands = mp.solutions.hands

with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5) as hands:
    results = hands.process(image_rgb)
    
    if not results.multi_hand_landmarks:
        print("❌ No hands detected in the image.")
        exit()

    h, w, _ = image_rgb.shape

    # Initialize min/max for both hands combined
    x_min, y_min, x_max, y_max = w, h, 0, 0

    for hand_landmarks in results.multi_hand_landmarks:
        x_coords = [lm.x for lm in hand_landmarks.landmark]
        y_coords = [lm.y for lm in hand_landmarks.landmark]

        x_min = min(x_min, int(min(x_coords) * w))
        x_max = max(x_max, int(max(x_coords) * w))
        y_min = min(y_min, int(min(y_coords) * h))
        y_max = max(y_max, int(max(y_coords) * h))

    # Add padding to ensure full hands are visible
    pad = 50
    x_min = max(0, x_min - pad)
    y_min = max(0, y_min - pad)
    x_max = min(w, x_max + pad)
    y_max = min(h, y_max + pad)

    # Crop the region containing both hands
    hand_crop = image_rgb[y_min:y_max, x_min:x_max]

    # -------------------------------
    # 6. Optional: sharpen slightly
    # -------------------------------
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    hand_crop = cv2.filter2D(hand_crop, -1, kernel)
    # After hand detection:
    hand_crop = image_rgb[y_min:y_max, x_min:x_max]

    # Add white background padding
    hand_crop = cv2.copyMakeBorder(hand_crop, 50, 50, 50, 50, 
                               cv2.BORDER_CONSTANT, value=[255,255,255])

    # Resize to model input
    hand_crop = cv2.resize(hand_crop, (224, 224))

# -------------------------------
# 7. Convert crop to tensor
# -------------------------------
hand_image = Image.fromarray(hand_crop)
input_tensor = transform(hand_image).unsqueeze(0)

# -------------------------------
# 8. Predict class
# -------------------------------
with torch.no_grad():
    outputs = model(input_tensor)
    probs = torch.nn.functional.softmax(outputs, dim=1)[0]
    top5 = torch.topk(probs, 5)

# -------------------------------
# 9. Print results
# -------------------------------
print("\n🔍 Top 5 Predictions:")
for idx, (cls_idx, prob) in enumerate(zip(top5.indices, top5.values)):
    print(f"Top {idx+1}: {class_names[cls_idx]}  —  {prob.item():.4f}")

predicted_class = class_names[top5.indices[0]]
print(f"\n✅ Predicted Class: {predicted_class}")

# -------------------------------
# 10. Visualize
# -------------------------------
cv2.rectangle(image_bgr, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
cv2.putText(image_bgr, predicted_class, (x_min, y_min - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
cv2.imwrite("predicted_output.jpg", image_bgr)
print("✅ Saved output with bounding box as predicted_output.jpg")

# -------------------------------
# 11. Show input to model
# -------------------------------
plt.imshow(hand_crop)
plt.title("Final Input to Model")
plt.axis("off")
plt.show()

# %%
import torch, json
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms, models

# load class mapping
with open("class_to_idx.json","r") as f:
    class_to_idx = json.load(f)
idx_to_class = {v:k for k,v in class_to_idx.items()}

# model build & load (state_dict)
num_classes = len(class_to_idx)
model = models.resnet101(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load("resnet101_signlang.pth", map_location='cpu'))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

img = Image.open("/mnt/d/MAJOR PROJECT/WIN_20250921_02_01_02_Pro.jpg").convert("RGB")
# If using MediaPipe crop, crop first and then do transform
inp = transform(img).unsqueeze(0)

with torch.no_grad():
    out = model(inp)
    probs = torch.softmax(out, dim=1).squeeze(0)
    topk = torch.topk(probs, k=5)
    for i, (idx, p) in enumerate(zip(topk.indices.tolist(), topk.values.tolist())):
        print(f"Top {i+1}: {idx_to_class[idx]}  —  {p:.4f}")


# 11. Show input to model
# -------------------------------
plt.imshow(img)
plt.title("Final Input to Model")
plt.axis("off")
plt.show()

# %%
import cv2
import torch
import mediapipe as mp
from PIL import Image
from torchvision import models, transforms
import numpy as np
import json

# -------------------------------
# 1. Load class mapping
# -------------------------------
with open("class_to_idx.json", "r") as f:
    class_to_idx = json.load(f)
idx_to_class = {v: k for k, v in class_to_idx.items()}

# -------------------------------
# 2. Load the trained model
# -------------------------------
num_classes = len(class_to_idx)
model = models.resnet101(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

model.load_state_dict(torch.load("resnet101_signlang_new3.pth", map_location='cpu'))
model.eval()

# -------------------------------
# 3. Define preprocessing (same as training)
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -------------------------------
# 4. Setup MediaPipe hands
# -------------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

# -------------------------------
# 5. Function to preprocess using MediaPipe
# -------------------------------
def mediapipe_preprocess(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    h, w, _ = img_bgr.shape

    if not results.multi_hand_landmarks:
        print("❌ No hand detected, using full image.")
        cropped = img_bgr
    else:
        # Bounding box around both hands
        x_min, y_min, x_max, y_max = w, h, 0, 0
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min, y_min = min(x_min, x), min(y_min, y)
                x_max, y_max = max(x_max, x), max(y_max, y)

        # Improved padding
        pad_x, pad_y = int(0.15 * w), int(0.15 * h)
        x_min, y_min = max(0, x_min - pad_x), max(0, y_min - pad_y)
        x_max, y_max = min(w, x_max + pad_x), min(h, y_max + pad_y)

        # Sanity check
        box_w, box_h = x_max - x_min, y_max - y_min
        if box_w * box_h < 0.1 * w * h:
            x_min = max(0, x_min - int(0.2 * w))
            y_min = max(0, y_min - int(0.2 * h))
            x_max = min(w, x_max + int(0.2 * w))
            y_max = min(h, y_max + int(0.2 * h))

        cropped = img_bgr[y_min:y_max, x_min:x_max]

    # Resize same as training
    cropped = cv2.resize(cropped, (224, 224))
    return cropped

# -------------------------------
# 6. Load and preprocess test image
# -------------------------------
image_path = "/mnt/d/MAJOR PROJECT/WIN_20250921_02_01_02_Pro.jpg"
img_bgr = cv2.imread(image_path)
if img_bgr is None:
    raise FileNotFoundError(f"Image not found at {image_path}")

processed_img = mediapipe_preprocess(img_bgr)

# Convert to tensor
img_pil = Image.fromarray(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))
input_tensor = transform(img_pil).unsqueeze(0)

# -------------------------------
# 7. Predict
# -------------------------------
with torch.no_grad():
    output = model(input_tensor)
    probs = torch.softmax(output, dim=1).squeeze(0)
    topk = torch.topk(probs, k=5)

# -------------------------------
# 8. Print results
# -------------------------------
print("\n🔍 Top 5 Predictions:")
for i, (idx, p) in enumerate(zip(topk.indices.tolist(), topk.values.tolist())):
    print(f"Top {i+1}: {idx_to_class[idx]}  —  {p:.4f}")

predicted_class = idx_to_class[topk.indices[0].item()]
print(f"\n✅ Predicted Class: {predicted_class}")




# %%
import cv2
import os
import random
import numpy as np
from glob import glob
from tqdm import tqdm
from PIL import Image
import albumentations as A
#%%
import cv2
import os
import random
import numpy as np
from glob import glob
from tqdm import tqdm
import mediapipe as mp
import albumentations as A

# -------------------------
# PATHS
# -------------------------
INPUT_DIR = "/mnt/d/MAJOR PROJECT/Indian/dataset_split/test"
OUTPUT_DIR = "/mnt/d/MAJOR PROJECT/kaggle_dataset"
BACKGROUND_DIR = "/mnt/d/MAJOR PROJECT/background"

os.makedirs(OUTPUT_DIR, exist_ok=True)

backgrounds = glob(os.path.join(BACKGROUND_DIR, "*"))

# -------------------------
# Mediapipe Hand Segmentation
# -------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

def extract_hand_mask(image):
    """Returns mask of the hand using MediaPipe"""
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    if not result.multi_hand_landmarks:
        return mask  # no hand detected

    h, w = image.shape[:2]

    for hand in result.multi_hand_landmarks:
        pts = []
        for lm in hand.landmark:
            x, y = int(lm.x * w), int(lm.y * h)
            pts.append([x, y])

        pts = np.array(pts)
        hull = cv2.convexHull(pts)
        cv2.fillConvexPoly(mask, hull, 255)

    return mask

# -------------------------
# Realistic Augmentation Pipeline
# -------------------------
augment = A.Compose([
    A.RandomBrightnessContrast(p=0.4),
    A.ColorJitter(p=0.4),
    A.GaussNoise(var_limit=(5, 20), p=0.2),
    A.MotionBlur(blur_limit=3, p=0.2),
    A.Rotate(limit=15, p=0.4),
    A.Affine(scale=(0.9, 1.1), translate_percent=0.1, p=0.3)
])

def place_on_background(fg, mask):
    """More realistic background replacement using the mask"""
    bg_path = random.choice(backgrounds)
    bg = cv2.imread(bg_path)
    bg = cv2.resize(bg, (fg.shape[1], fg.shape[0]))

    mask3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Keep only hand
    hand = cv2.bitwise_and(fg, mask3)

    # Remove hand from background area
    inv_mask3 = cv2.bitwise_not(mask3)
    bg_area = cv2.bitwise_and(bg, inv_mask3)

    return cv2.add(bg_area, hand)


# -------------------------
# PROCESS CLASSES
# -------------------------
for class_name in os.listdir(INPUT_DIR):
    class_in = os.path.join(INPUT_DIR, class_name)
    class_out = os.path.join(OUTPUT_DIR, class_name)

    os.makedirs(class_out, exist_ok=True)

    images = glob(os.path.join(class_in, "*"))

    print(f"\nProcessing {class_name} - {len(images)} images")

    for img_path in tqdm(images):
        img = cv2.imread(img_path)
        if img is None:
            continue

        mask = extract_hand_mask(img)

        # Skip if hand mask missing
        if mask.sum() == 0:
            continue

        for i in range(10):  # 10 augmentations per image
            aug = augment(image=img)
            aug_img = aug["image"]

            # 50% chance to apply background
            if random.random() < 0.5:
                final_img = place_on_background(aug_img, mask)
            else:
                final_img = aug_img

            out_name = f"{os.path.splitext(os.path.basename(img_path))[0]}_aug{i}.jpg"
            cv2.imwrite(os.path.join(class_out, out_name), final_img)

print("\nDONE ✔ Augmentation completed realistically")

# %%
import torch
print(torch.cuda.is_available())


# %%
