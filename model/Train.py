import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
import cv2
from sklearn.model_selection import train_test_split

# 🔹 Constants
AUGMENTED_CSV = "Augmented_Metadata.csv"
IMAGE_FOLDER = "Augmented_Images"
MODEL_SAVE_PATH = "model/custom_skin_lesion_model.h5"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 100

# 🔹 Ensure model directory exists
os.makedirs("model", exist_ok=True)


def load_data(csv_path, image_folder):
    """ Load dataset from CSV and preprocess images & labels """
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

    return np.array(images), to_categorical(np.array(labels), num_classes=3)


def build_custom_cnn():
    """ Define the Custom CNN Model Architecture from the Research Paper """
    model = Sequential([
        # Block 1
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(3, 3)),
        Dropout(0.25),

        # Block 2
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Block 3
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Fully Connected Layers
        Flatten(),
        Dense(1024, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),

        # Output Layer (Softmax for multi-class classification)
        Dense(3, activation='softmax')  # 3 classes (melanoma, nevus, seborrheic keratosis)
    ])

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def train_and_save_model():
    """ Load data, train the Custom CNN, and save the trained model """
    print("🔹 Loading augmented dataset...")
    X, y = load_data(AUGMENTED_CSV, IMAGE_FOLDER)

    print("🔹 Splitting data into train & validation...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    print("🔹 Building Custom CNN model...")
    model = build_custom_cnn()

    print("🔹 Training model...")
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE)

    print(f"✅ Saving model to {MODEL_SAVE_PATH}")
    model.save(MODEL_SAVE_PATH)