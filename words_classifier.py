import cv2
import mediapipe as mp
import numpy as np
import pickle
import threading
import queue
import pythoncom
import win32com.client 

# ============================
# Load Model & Encoder
# ============================
MODEL_PATH = "sign_model6.pkl"
ENCODER_PATH = "label_encoder6.pkl"

model = pickle.load(open(MODEL_PATH, "rb"))
encoder = pickle.load(open(ENCODER_PATH, "rb"))

# ============================
# FAST AUDIO WORKER
# ============================
speech_queue = queue.Queue()

def tts_worker():
    pythoncom.CoInitialize()
    speaker = win32com.client.Dispatch("SAPI.SpVoice")
    while True:
        text = speech_queue.get()
        if text is None: break
        speaker.Speak(text) 
        speech_queue.task_done()

threading.Thread(target=tts_worker, daemon=True).start()

# ============================
# MediaPipe & Camera Setup
# ============================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
WINDOW_NAME = "Sign Language Detection"

cv2.namedWindow(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# ============================
# Landmark Normalization
# ============================
def normalize_landmarks(landmarks):
    x_vals = landmarks[0::3]
    y_vals = landmarks[1::3]
    min_x, max_x = min(x_vals), max(x_vals)
    min_y, max_y = min(y_vals), max(y_vals)
    normalized = []
    for i in range(0, len(landmarks), 3):
        norm_x = (landmarks[i]   - min_x) / (max_x - min_x + 1e-6)
        norm_y = (landmarks[i+1] - min_y) / (max_y - min_y + 1e-6)
        norm_z = landmarks[i+2]
        normalized.extend([norm_x, norm_y, norm_z])
    return normalized

# ============================
# Main Loop Variables
# ============================
last_spoken_label = None
current_candidate = None
stability_counter = 0

# ADJUST THIS: 5 frames is roughly 0.15 seconds. 10 frames is ~0.3 seconds.
STABILITY_THRESHOLD = 7 

no_hand_frames = 0
NO_HAND_THRESHOLD = 15

while True:
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        no_hand_frames = 0
        hand = results.multi_hand_landmarks[0]
        
        # Data Processing
        data = []
        for lm in hand.landmark:
            data.extend([lm.x, lm.y, lm.z])
        data = normalize_landmarks(data)

        # Predict
        prediction = model.predict([data])[0]
        detected_label = str(encoder.inverse_transform([prediction])[0])

        # --- STABILITY LOGIC ---
        if detected_label == current_candidate:
            stability_counter += 1
        else:
            current_candidate = detected_label
            stability_counter = 0

        # Only proceed to speak if the sign has been stable
        if stability_counter >= STABILITY_THRESHOLD:
            if current_candidate != last_spoken_label:
                # Clear previous pending audio
                while not speech_queue.empty():
                    try: speech_queue.get_nowait()
                    except: break
                
                speech_queue.put(current_candidate)
                last_spoken_label = current_candidate
        
        # UI Feedback
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
        cv2.putText(frame, f"SIGN: {detected_label}", (50, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
        
        # Optional: Progress bar for the "delay"
        bar_width = int((stability_counter / STABILITY_THRESHOLD) * 200)
        cv2.rectangle(frame, (50, 130), (50 + bar_width, 145), (0, 255, 0), -1)

    else:
        no_hand_frames += 1
        if no_hand_frames > NO_HAND_THRESHOLD:
            last_spoken_label = None
            current_candidate = None
            stability_counter = 0

    cv2.imshow(WINDOW_NAME, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()