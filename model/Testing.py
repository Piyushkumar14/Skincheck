import os
import pandas as pd
import numpy as np
import tensorflow as tf
import cv2
from sklearn.metrics import classification_report

# 🔹 Constants
MODEL_PATH = "model/skin_lesion_model.keras"
TEST_METADATA_CSV = "ISIC-2017_Test_v2_Data/ISIC-2017_Test_v2_Data_metadata.csv"
TEST_LABELS_CSV = "ISIC-2017_Test_v2_Data/ISIC-2017_Test_v2_Part3_GroundTruth.csv"
IMAGE_FOLDER = "ISIC-2017_Test_v2_Data"
IMG_SIZE = (224, 224)

# 🔹 Label Mapping
LABEL_MAPPING = {"melanoma": 0, "nevus": 1, "seborrheic_keratosis": 2}

def get_label(row):
    """ Convert ground truth CSV to label class """
    if row["melanoma"] == 1.0:
        return "melanoma"
    elif row["seborrheic_keratosis"] == 1.0:
        return "seborrheic_keratosis"
    else:
        return "nevus"

def load_test_data(metadata_path, labels_path, image_folder):
    """ Load test metadata, ground truth labels, and images """
    metadata_df = pd.read_csv(metadata_path)
    labels_df = pd.read_csv(labels_path)

    # Map ground truth labels
    labels_df["label"] = labels_df.apply(get_label, axis=1)

    # Merge metadata with labels
    df = pd.merge(metadata_df, labels_df[["image_id", "label"]], on="image_id")

    images, labels = [], []
    for _, row in df.iterrows():
        img_path = os.path.join(image_folder, row["image_id"] + ".jpg")
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            img = cv2.resize(img, IMG_SIZE)
            img = img / 255.0  # Normalize
            images.append(img)
            labels.append(LABEL_MAPPING[row["label"]])

    return np.array(images), np.array(labels)

def evaluate_model():
    """ Load model and evaluate on test data """
    print("🔹 Loading test dataset...")
    X_test, y_test = load_test_data(TEST_METADATA_CSV, TEST_LABELS_CSV, IMAGE_FOLDER)

    print("🔹 Loading trained model...")
    model = tf.keras.models.load_model(MODEL_PATH)

    print("🔹 Making predictions...")
    y_pred = np.argmax(model.predict(X_test), axis=1)

    print("\n✅ Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Melanoma", "Nevus", "Seborrheic Keratosis"]))

