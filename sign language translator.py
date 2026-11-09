#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ================= CELL 1 =================
import os, cv2, pickle, warnings
import numpy as np
import mediapipe as mp
import tensorflow as tf

from tqdm import tqdm
from collections import deque, Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

warnings.filterwarnings('ignore')

# In[3]:


# ================= CELL 2 =================
DATASET_PATH = r"C:\Users\malai\Downloads\archive (2)\asl_alphabet_train\asl_alphabet_train"

MODEL_FILE = "sign_model.h5"
ENCODER_FILE = "label_encoder.pkl"

# In[5]:


# ================= CELL 3 =================
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1
)

X, y = [], []

for label in os.listdir(DATASET_PATH):

    folder = os.path.join(DATASET_PATH, label)

    if not os.path.isdir(folder):
        continue

    for img_name in tqdm(os.listdir(folder), desc=label):

        path = os.path.join(folder, img_name)

        img = cv2.imread(path)

        if img is None:
            continue

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        result = hands.process(rgb)

        if result.multi_hand_landmarks:

            hand = result.multi_hand_landmarks[0]
            wrist = hand.landmark[0]

            points = []

            for lm in hand.landmark:

                points.extend([
                    lm.x - wrist.x,
                    lm.y - wrist.y,
                    lm.z - wrist.z
                ])

            X.append(points)
            y.append(label)

hands.close()

X = np.array(X)
y = np.array(y)

print("Samples:", len(X))
print("Classes:", np.unique(y))

# In[7]:


# ================= CELL 4 =================
le = LabelEncoder()

y_encoded = le.fit_transform(y)

y_onehot = to_categorical(y_encoded)

with open(ENCODER_FILE, 'wb') as f:
    pickle.dump(le, f)

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y_onehot,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print("Train:", X_train.shape)
print("Validation:", X_val.shape)

# In[9]:


# ================= CELL 5 =================
model = Sequential([

    Dense(256, activation='relu', input_shape=(63,)),
    Dropout(0.3),

    Dense(128, activation='relu'),
    Dropout(0.3),

    Dense(len(le.classes_), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# In[11]:


# ================= CELL 6 =================
history = model.fit(
    X_train,
    y_train,

    validation_data=(X_val, y_val),

    epochs=20,
    batch_size=64
)

model.save(MODEL_FILE)

print("Model Saved")

# In[15]:


# ================= CELL 7 =================
model = load_model(MODEL_FILE)

with open(ENCODER_FILE, 'rb') as f:
    le = pickle.load(f)

cap = cv2.VideoCapture(0)

pred_buffer = deque(maxlen=10)

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1
) as hands:

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = hands.process(rgb)

        if result.multi_hand_landmarks:

            hand = result.multi_hand_landmarks[0]

            wrist = hand.landmark[0]

            coords = []

            for lm in hand.landmark:

                coords.extend([
                    lm.x - wrist.x,
                    lm.y - wrist.y,
                    lm.z - wrist.z
                ])

            data = np.array(coords).reshape(1, -1)

            pred = model.predict(data, verbose=0)[0]

            idx = np.argmax(pred)

            pred_buffer.append(idx)

            smooth = Counter(pred_buffer).most_common(1)[0][0]

            label = le.classes_[smooth]

            conf = pred[smooth]

            cv2.putText(
                frame,
                f"{label} ({conf:.2f})",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                2
            )

            mp.solutions.drawing_utils.draw_landmarks(
                frame,
                hand,
                mp_hands.HAND_CONNECTIONS
            )

        cv2.imshow("Sign Language", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

# In[ ]:



