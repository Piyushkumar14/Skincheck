import model.augment_and_label as aug
from model.Testing import evaluate_model
from model.Train import train_and_save_model
from model.validation import validate_model

METADATA_PATH = "./data/ISIC-2017_Training_Data/ISIC-2017_Training_Data_metadata.csv"
LABELS_PATH = "./data/ISIC-2017_Training_Data/ISIC-2017_Training_Part3_GroundTruth.csv"
IMAGE_FOLDER = "./data/ISIC-2017_Training_Data"  # Folder with original images
OUTPUT_FOLDER = "./Augmented_Images"  # Where augmented images will be saved
OUTPUT_CSV = "./Augmented_Metadata.csv"  # Final CSV with labels, age, sex


def main():

    print("🔹 Loading and merging data...")
    aug.augment_and_preprocess()
    print("🔹 Augmentation complete!")


    print("🔹 Training model...")
    train_and_save_model()
    print("🔹 Model training complete!")

    # print("🔹 Model validation...")
    # validate_model()
    # print("🔹 Model validation complete!")


    # print("🔹 Testing model...")
    # evaluate_model()
    # print("🔹 Testing complete!")

if __name__ == "__main__":
    main()