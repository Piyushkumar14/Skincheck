import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
import cv2

# 🔹 Constants
MODEL_PATH = "model/skin_lesion_model.h5"
AUGMENTED_CSV = "Augmented_Metadata.csv"
IMAGE_FOLDER = "Augmented_Images"
IMG_SIZE = (224, 224)

# 🔹 Load dataset
def load_data(csv_path, image_folder):
    df = pd.read_csv(csv_path)

    # Label encoding
    label_mapping = {"melanoma": 0, "nevus": 1, "seborrheic_keratosis": 2}
    df["label"] = df["label"].map(label_mapping)

    images, labels = [], []
    for _, row in df.iterrows():
        img_path = os.path.join(image_folder, row["image_id"])
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            img = cv2.resize(img, IMG_SIZE)
            img = img / 255.0  # Normalize
            images.append(img)
            labels.append(row["label"])

    return np.array(images), np.array(labels)

# 🔹 Load model
def validate_model():
    print("🔹 Loading validation dataset...")
    X, y_true = load_data(AUGMENTED_CSV, IMAGE_FOLDER)

    # Split into train & validation
    from sklearn.model_selection import train_test_split
    _, X_val, _, y_val = train_test_split(X, y_true, test_size=0.2, random_state=42)

    print("🔹 Loading trained model...")
    model = tf.keras.models.load_model(MODEL_PATH)

    print("🔹 Evaluating model...")
    y_pred = np.argmax(model.predict(X_val), axis=1)

    print("\n✅ Classification Report:")
    print(classification_report(y_val, y_pred, target_names=["Melanoma", "Nevus", "Seborrheic Keratosis"]))
