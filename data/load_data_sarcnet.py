import pandas as pd
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import CLIPProcessor
from PIL import Image
import os
import gdown
import zipfile
# import io # not needed it seems

# Settings
SARCNET_DIR = "./data/sarcnet/"  # Change if needed
PROCESSED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sarcnet_processed")

# Google Drive settings
zip_file_id = '18m3KdDCXgkAlvTbjNhfftvU9LdhyDUTt'  # SarcNet file ID
zip_path = './data/sarcnet.zip'  # Temporary zip download location
extract_dir = './data/sarcnet/'  # Where the data will go

# Download and unzip SarcNet dataset if missing
if not os.path.exists(extract_dir):
    os.makedirs('./data', exist_ok=True)
    url = f'https://drive.google.com/uc?id={zip_file_id}'
    gdown.download(url, zip_path, quiet=False)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall('./data/')
    print("Downloaded and extracted SarcNet dataset.")

# Load SarcNet CSVs
train_df = pd.read_csv(os.path.join(SARCNET_DIR, "train.csv"), encoding='latin1')
val_df = pd.read_csv(os.path.join(SARCNET_DIR, "val.csv"), encoding='latin1')
test_df = pd.read_csv(os.path.join(SARCNET_DIR, "test.csv"), encoding='latin1')

# Make sure columns exist
assert all(col in train_df.columns for col in ["Imagepath", "Text", "Image_label", "Text_label", "Multi_label"])

# print(list(train_df.columns))
# print(list(train_dataset.column_names))

# Convert to Huggingface Dataset format
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)

# uses CLIP's processor
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Preprocessing function
def preprocess_sarcnet(example):
    # image_path = os.path.join(SARCNET_DIR, "Image", example["Imagepath"])
    # image = Image.open(image_path).convert("RGB")

    image_paths = [
        os.path.join(SARCNET_DIR, "Image", img_path) for img_path in example["Imagepath"]
    ]
    images = [Image.open(path).convert("RGB") for path in image_paths]

    inputs = processor(
        text=example["Text"],
        images=images,
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=77, # cut off captions after this length
    )

# ["Imagepath", "Text", "Image_label", "Text_label", "Multi_label"]
    return {
        # for clip
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "pixel_values": inputs["pixel_values"],
        "label": example["Multi_label"],

        # image-specific label
        "image_label": example["Image_label"],
        # test-specific label
        "text_label": example["Text_label"],

        # for bert
        "text_list": example["Text"],
        "image_list": example["Imagepath"],
        "label_list": example["Multi_label"],
        # "text_label": example["Text_label"],
        "samples": inputs["pixel_values"]
    }

# Preprocessing and saving
def main():
    if not os.path.exists(PROCESSED_DIR):
        processes = 2 # i changed this from 4 because my computer's a brick
        processed_train = train_dataset.map(preprocess_sarcnet, batched=True, num_proc=processes)
        processed_val = val_dataset.map(preprocess_sarcnet, batched=True, num_proc=processes)
        processed_test = test_dataset.map(preprocess_sarcnet, batched=True, num_proc=processes)

        for ds in [processed_train, processed_val, processed_test]:
            ds.set_format(
                type="numpy",
                columns=[
                    "input_ids", "attention_mask", "pixel_values",
                    "image_label", "text_label", "label",
                    "text_list", "image_list", "label_list", "samples"
                ]
            )

        full_dataset = DatasetDict({
            "train": processed_train,
            "val": processed_val,
            "test": processed_test,
        })

        full_dataset.save_to_disk(PROCESSED_DIR)
    else:
        # Load processed data if already exists
        full_dataset = load_from_disk(PROCESSED_DIR)

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
