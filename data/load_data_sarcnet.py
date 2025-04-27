import pandas as pd
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import CLIPProcessor
from PIL import Image
import os
# import io # not needed it seems

# Settings
SARCNET_DIR = "./data/sarcnet/"  # Change if needed
PROCESSED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sarcnet_processed")

# Load SarcNet CSVs
train_df = pd.read_csv(os.path.join(SARCNET_DIR, "train.csv"))
val_df = pd.read_csv(os.path.join(SARCNET_DIR, "val.csv"))
test_df = pd.read_csv(os.path.join(SARCNET_DIR, "test.csv"))

# Make sure columns exist: ['image_path', 'text', 'image_label', 'text_label', 'multimodal_label']
assert all(col in train_df.columns for col in ["image_path", "text", "image_label", "text_label", "multimodal_label"])

# Convert to Huggingface Dataset format
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)

# uses CLIP's processor
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Preprocessing function
def preprocess_sarcnet(example):
    image_path = os.path.join(SARCNET_DIR, "images", example["image_path"])
    image = Image.open(image_path).convert("RGB")

    inputs = processor(
        text=example["text"],
        images=image,
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=77,
    )

    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "pixel_values": inputs["pixel_values"],
        "image_label": example["image_label"],
        "text_label": example["text_label"],
        "multimodal_label": example["multimodal_label"],
        "text_list": example["text"],
        "image_list": example["image_path"],
    }

# Preprocessing and saving
def main():
    if not os.path.exists(PROCESSED_DIR):
        processed_train = train_dataset.map(preprocess_sarcnet, num_proc=4)
        processed_val = val_dataset.map(preprocess_sarcnet, num_proc=4)
        processed_test = test_dataset.map(preprocess_sarcnet, num_proc=4)

        for ds in [processed_train, processed_val, processed_test]:
            ds.set_format(
                type="numpy",
                columns=[
                    "input_ids", "attention_mask", "pixel_values",
                    "image_label", "text_label", "multimodal_label",
                    "text_list", "image_list"
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
