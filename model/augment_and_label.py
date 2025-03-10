import os
import pandas as pd
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from collections import Counter

# 🔹 Paths
METADATA_PATH = "data/ISIC-2017_Training_Data/ISIC-2017_Training_Data_metadata.csv"
LABELS_PATH = "data/ISIC-2017_Training_Data/ISIC-2017_Training_Part3_GroundTruth.csv"
IMAGE_FOLDER = "data/ISIC-2017_Training_Data"
OUTPUT_FOLDER = "Augmented_Preprocessed_Images"
OUTPUT_CSV = "Augmented_Metadata.csv"

# 🔹 Create output directory if not exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# 🔹 Label Mapping Function
def get_label(row):
    """ Convert ground truth CSV to categorical labels """
    if row["melanoma"] == 1.0:
        return "melanoma"
    elif row["seborrheic_keratosis"] == 1.0:
        return "seborrheic_keratosis"
    else:
        return "nevus"


def load_and_merge_data(metadata_path, labels_path):
    """ Load metadata & labels, merge on image_id """
    metadata_df = pd.read_csv(metadata_path)
    labels_df = pd.read_csv(labels_path)

    # Map labels
    labels_df["label"] = labels_df.apply(get_label, axis=1)

    # Merge metadata with labels
    return pd.merge(metadata_df, labels_df[["image_id", "label"]], on="image_id")


def preprocess_image(img):
    """ Apply preprocessing: resize, histogram equalization, normalization """
    img = cv2.resize(img, (224, 224))  # Resize

    # Convert to grayscale & apply histogram equalization
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    img = img / 255.0  # Normalize (scale pixel values between 0 and 1)
    return img


def augment_image(img, image_id, datagen, num_aug=3):
    """ Generate augmented images and save them """
    img = np.expand_dims(img, axis=0)  # Expand dims for augmentation
    augmented_images = []

    for i, batch in enumerate(datagen.flow(img, batch_size=1)):
        aug_img = batch[0]
        aug_filename = f"{image_id}_aug_{i}.jpg"
        aug_path = os.path.join(OUTPUT_FOLDER, aug_filename)
        cv2.imwrite(aug_path, (aug_img * 255).astype(np.uint8))  # Convert back to 0-255
        augmented_images.append(aug_filename)

        if i >= num_aug - 1:  # Generate num_aug images per original
            break

    return augmented_images


def balance_dataset(df):
    """ Apply oversampling to balance melanoma & seborrheic keratosis """
    counts = Counter(df["label"])
    max_samples = max(counts.values())  # Find the max class count

    augmented_df = pd.DataFrame()
    for label in counts.keys():
        class_df = df[df["label"] == label]
        oversampled_df = class_df.sample(max_samples, replace=True, random_state=42)  # Oversample
        augmented_df = pd.concat([augmented_df, oversampled_df])

    return augmented_df.reset_index(drop=True)


def augment_and_preprocess():
    """ Process dataset: resize, enhance contrast, augment, and balance dataset """
    print("🔹 Loading dataset...")
    df = load_and_merge_data(METADATA_PATH, LABELS_PATH)

    print("🔹 Balancing dataset with oversampling...")
    df = balance_dataset(df)  # Handle class imbalance

    datagen = ImageDataGenerator(
        rotation_range=30, width_shift_range=0.2, height_shift_range=0.2,
        brightness_range=[0.8, 1.2], horizontal_flip=True
    )

    augmented_data = []
    for _, row in df.iterrows():
        img_id, label, age, sex = row["image_id"], row["label"], row["age_approximate"], row["sex"]
        img_path = os.path.join(IMAGE_FOLDER, img_id + ".jpg")

        if not os.path.exists(img_path):
            print(f"Skipping {img_id} (Image Not Found)")
            continue

        img = cv2.imread(img_path)
        img = preprocess_image(img)  # Apply preprocessing

        # Save original preprocessed image
        preprocessed_filename = f"{img_id}_preprocessed.jpg"
        preprocessed_path = os.path.join(OUTPUT_FOLDER, preprocessed_filename)
        cv2.imwrite(preprocessed_path, (img * 255).astype(np.uint8))
        augmented_data.append([preprocessed_filename, label, age, sex])

        # Generate augmented images
        aug_images = augment_image(img, img_id, datagen, num_aug=3)
        for aug_img in aug_images:
            augmented_data.append([aug_img, label, age, sex])

    # Save updated metadata
    augmented_df = pd.DataFrame(augmented_data, columns=["image_id", "label", "age_approximate", "sex"])
    augmented_df.to_csv(OUTPUT_CSV, index=False)

    print(f"✅ Augmented dataset created: {len(augmented_df)} images")
    print(f"✅ Metadata saved to {OUTPUT_CSV}")
