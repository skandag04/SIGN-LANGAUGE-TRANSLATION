import cv2
import mediapipe as mp
import numpy as np
import pickle

MODEL_PATH = "sign_model5.pkl"
ENCODER_PATH = "label_encoder5.pkl"

model = pickle.load(open(MODEL_PATH, "rb"))
encoder = pickle.load(open(ENCODER_PATH, "rb"))

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
WINDOW_NAME = "Numbers Detection" # Define a name for the window

# 1. Create the window with the defined name
cv2.namedWindow(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN) 
# 2. Set the window property to full screen
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN) 

def normalize_landmarks(landmarks):
    x_vals = landmarks[0::3]
    y_vals = landmarks[1::3]

    min_x, max_x = min(x_vals), max(x_vals)
    min_y, max_y = min(y_vals), max(y_vals)

    normalized = []
    for i in range(0, len(landmarks), 3):
        norm_x = (landmarks[i]   - min_x) / (max_x - min_x + 1e-6)
        norm_y = (landmarks[i+1] - min_y) / (max_y - min_y + 1e-6)
        norm_z = landmarks[i+2]   # Keeping Z unchanged
        normalized.extend([norm_x, norm_y, norm_z])
    return normalized


while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]

        # Extract raw coordinates
        data = []
        for lm in hand.landmark:
            data.extend([lm.x, lm.y, lm.z])

        # 🔹 Apply SAME normalization used in training
        data = normalize_landmarks(data)

        prediction = model.predict([data])[0]
        label = encoder.inverse_transform([prediction])[0]
        label = str(label)

        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        cv2.putText(frame, label, (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    # Use the defined window name
    cv2.imshow(WINDOW_NAME, frame) 
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()