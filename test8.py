#%%
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

DATASET_DIR = "/mnt/d/MAJOR PROJECT/new dataset/words"
OUTPUT_CSV = "landmarks/landmarks6.csv"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

def extract_landmarks(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if not results.multi_hand_landmarks:
        return None
    
    hand = results.multi_hand_landmarks[0]
    data = []

    for lm in hand.landmark:
        data.extend([lm.x, lm.y, lm.z])

    return data

def main():
    os.makedirs("landmarks", exist_ok=True)

    dataset = []

    for label in sorted(os.listdir(DATASET_DIR)):
        folder = os.path.join(DATASET_DIR, label)

        if not os.path.isdir(folder):
            continue

        for img_name in tqdm(os.listdir(folder), desc=f"Processing {label}"):
            img_path = os.path.join(folder, img_name)
            lm = extract_landmarks(img_path)

            if lm is not None:
                dataset.append([label] + lm)

    df = pd.DataFrame(dataset)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nLandmark dataset saved at {OUTPUT_CSV}")
    print("Total samples:", len(df))

if __name__ == "__main__":
    main()

# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pickle
import os

CSV_PATH = "landmarks/landmarks6.csv"
MODEL_PATH = "models/sign_model6.pkl"
ENCODER_PATH = "models/label_encoder6.pkl"

# Normalize keypoints (Min-Max normalization per sample)
def normalize_landmarks(row):
    x_vals = row[0::3]
    y_vals = row[1::3]

    min_x, max_x = min(x_vals), max(x_vals)
    min_y, max_y = min(y_vals), max(y_vals)

    normalized = []
    for i in range(0, len(row), 3):
        norm_x = (row[i]   - min_x) / (max_x - min_x + 1e-6)
        norm_y = (row[i+1] - min_y) / (max_y - min_y + 1e-6)
        norm_z = row[i+2]  # Keeping Z as original
        normalized.extend([norm_x, norm_y, norm_z])
    return normalized


def main():
    os.makedirs("models", exist_ok=True)

    df = pd.read_csv(CSV_PATH)

    y = df.iloc[:, 0]
    X = df.iloc[:, 1:]

    # 🔹 Apply normalization to each row
    X = X.apply(normalize_landmarks, axis=1, result_type='expand')

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.15, shuffle=True, stratify=y_encoded
    )

    clf = RandomForestClassifier(n_estimators=600)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print("Test Accuracy:", accuracy_score(y_test, y_pred))

    pickle.dump(clf, open(MODEL_PATH, "wb"))
    pickle.dump(encoder, open(ENCODER_PATH, "wb"))

    print(f"\nModel saved at {MODEL_PATH}")
    print(f"Label encoder saved at {ENCODER_PATH}")


if __name__ == "__main__":
    main()

# %%
import cv2
import mediapipe as mp
import numpy as np
import pickle

MODEL_PATH = "models/sign_model.pkl"
ENCODER_PATH = "models/label_encoder.pkl"

model = pickle.load(open(MODEL_PATH, "rb"))
encoder = pickle.load(open(ENCODER_PATH, "rb"))

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]

        data = []
        for lm in hand.landmark:
            data.extend([lm.x, lm.y, lm.z])

        prediction = model.predict([data])[0]
        label = encoder.inverse_transform([prediction])[0]

        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        cv2.putText(frame, label, (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    cv2.imshow("Sign Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# %%
import pandas as pd
import glob
import os

data_list = []
dataset_folder = "/mnt/d/MAJOR PROJECT/American Sign Language Digits Dataset"  # <-- update if different

# Loop through each digit CSV
for file in glob.glob(os.path.join(dataset_folder, "*.csv")):
    df = pd.read_csv(file)

    # Extract digit label from filename
    digit = int(os.path.basename(file).split("_")[-1].split(".")[0])
    df.insert(0, "label", digit)

    data_list.append(df)

# Combine all CSVs
combined_df = pd.concat(data_list, ignore_index=True)
combined_df.to_csv("keypoints_dataset.csv", index=False)

print("Dataset prepared ✔")
print("Saved file: keypoints_dataset.csv")
print("Total samples:", len(combined_df))

# %%
# train_classifier.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pickle

df = pd.read_csv("keypoints_dataset.csv")

X = df.iloc[:, 1:].values  # Keypoints
y = df.iloc[:, 0].values   # Labels (digits)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = KNeighborsClassifier(n_neighbors=7)
model.fit(X_train, y_train)

print("Accuracy:", model.score(X_test, y_test))

pickle.dump(model, open("sign_digit_model.pkl", "wb"))
print("Model saved as sign_digit_model.pkl")

# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pickle

df = pd.read_csv("keypoints_dataset.csv")

# Remove columns that are not keypoints or labels
df = df.select_dtypes(include=['float64', 'int64'])

# First column must be label
y = df.iloc[:, 0].values   # labels
X = df.iloc[:, 1:].values  # keypoints only

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = KNeighborsClassifier(n_neighbors=7)
model.fit(X_train, y_train)

print("Accuracy:", model.score(X_test, y_test))

pickle.dump(model, open("sign_digit_model.pkl", "wb"))
print("Model saved as sign_digit_model.pkl")

# %%
import pyttsx3

engine = pyttsx3.init()
engine.setProperty('rate', 150)    # speed of speech
engine.setProperty('volume', 1.0)  # volume (0.0 to 1.0)

engine.say("Hello, I am a speech synthesis engine.")
engine.runAndWait()
# %%
