import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from ultralytics import YOLO
import mediapipe as mp

# Check if CUDA (NVIDIA GPU) is available. If not, use CPU.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device for ResNet: {DEVICE}")
# ==================================

# =============================
# LOAD RESNET50
# =============================
num_classes = 26
resnet = models.resnet50(pretrained=False)
resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
state_dict = torch.load("alphabet_resnet50_signlang2.pth", map_location="cpu")
resnet.load_state_dict(state_dict)
resnet.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])


# =============================
# YOLO
# =============================
yolo_model = YOLO("runs/classify/train4/weights/best.pt")

CLASS_NAMES = [
    "A","B","C","D","E","F","G","H","I","J",
    "K","L","M","N","O","P","Q","R","S","T",
    "U","V","W","X","Y","Z"
]


# =============================
# MEDIAPIPE HANDS
# =============================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)


cap = cv2.VideoCapture(0)

# --- FULLSCREEN IMPLEMENTATION ---
WINDOW_NAME = "Sign Recognition"
cv2.namedWindow(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN) 
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
# ---------------------------------

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        h, w, c = frame.shape

        xs, ys = [], []

        # Collect ALL hand landmark points (both hands)
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                xs.append(int(lm.x * w))
                ys.append(int(lm.y * h))

        # Bounding box around BOTH hands
        x_min = max(min(xs) - 40, 0)
        y_min = max(min(ys) - 40, 0)
        x_max = min(max(xs) + 40, w)
        y_max = min(max(ys) + 40, h)

        combined_crop = frame[y_min:y_max, x_min:x_max]

        if combined_crop.size == 0:
            cv2.imshow(WINDOW_NAME, frame)
            continue

        # ---------- YOLO ----------
        # Set imgsz to 224 to match the ResNet input size
        yolo_res = yolo_model.predict(combined_crop, imgsz=224, verbose=False)[0]
        y_probs = yolo_res.probs.data
        y_cls = int(yolo_res.probs.top1)
        y_conf = float(y_probs[y_cls])
        y_label = CLASS_NAMES[y_cls]

        # ---------- RESNET ----------
        img_tensor = transform(combined_crop).unsqueeze(0)
        with torch.no_grad():
            out = resnet(img_tensor)
            probs = F.softmax(out, dim=1)
            r_conf, r_cls = torch.max(probs, dim=1)
            r_conf = float(r_conf)
            r_cls = int(r_cls)
            r_label = CLASS_NAMES[r_cls]

        # ---------- PICK BEST ----------
        if r_conf > y_conf:
            final_label = r_label
            final_conf = r_conf
            source = "ResNet50"
        else:
            final_label = y_label
            final_conf = y_conf
            source = "YOLO"

        # ---------- DRAW FULL BOX ----------
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,255,0), 2)
        cv2.putText(
            frame,
            f"{final_label} ({final_conf:.2f})",
            (x_min, y_min - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9, (0,255,0), 2
        )

    # Use the defined window name
    cv2.imshow(WINDOW_NAME, frame) 
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()