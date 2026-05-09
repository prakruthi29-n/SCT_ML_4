import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

dataset_path = "dataset/leapGestRecog"

X = []
y = []

print("Loading images...")

for folder in os.listdir(dataset_path):

    folder_path = os.path.join(dataset_path, folder)

    if not os.path.isdir(folder_path):
        continue

    for gesture in os.listdir(folder_path):

        gesture_path = os.path.join(folder_path, gesture)

        if not os.path.isdir(gesture_path):
            continue

        for image_name in os.listdir(gesture_path):

            image_path = os.path.join(gesture_path, image_name)

            try:
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                img = cv2.resize(img, (64, 64))

                X.append(img.flatten())

                y.append(gesture)

            except:
                pass

print("Dataset Loaded Successfully")

X = np.array(X)
y = np.array(y)

print("Splitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training model...")

model = RandomForestClassifier(n_estimators=100)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print(f"Model Accuracy: {accuracy * 100:.2f}%")

joblib.dump(model, "gesture_model.pkl")

print("Model saved as gesture_model.pkl")