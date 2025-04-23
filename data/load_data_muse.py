import pandas as pd
from datasets import Dataset, load_dataset, load_from_disk, DatasetDict

from transformers import CLIPProcessor
from PIL import Image
import os

# Load TSV files using pandas
train_df = pd.read_csv('./data/muse/train_df.tsv', sep='\t')  # Use the correct path for your TSV
val_df = pd.read_csv('./data/muse/val_df.tsv', sep='\t')
test_df = pd.read_csv('./data/muse/test_df.tsv', sep='\t')

# Add a sarcasm label (1 for sarcasm) - all data is sarcastic
train_df['label'] = 1
val_df['label'] = 1
test_df['label'] = 1

# Convert the pandas DataFrame to the Hugging Face dataset format
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)

# rename columns, then remove second caption (not relevant for our task)
train_dataset = train_dataset.rename_columns({
    train_dataset.column_names[0]: "image",
    train_dataset.column_names[1]: "text",  # First caption
    train_dataset.column_names[2]: "text_2",  # Second caption to remove
    train_dataset.column_names[3]: "label"
})
train_dataset = train_dataset.remove_columns("text_2")

val_dataset = val_dataset.rename_columns({
    val_dataset.column_names[0]: "image",
    val_dataset.column_names[1]: "text",  # First caption
    val_dataset.column_names[2]: "text_2",  # Second caption to remove
    val_dataset.column_names[3]: "label"
})
val_dataset = val_dataset.remove_columns("text_2")

test_dataset = test_dataset.rename_columns({
    test_dataset.column_names[0]: "image",
    test_dataset.column_names[1]: "text",  # First caption
    test_dataset.column_names[2]: "text_2",  # Second caption to remove
    test_dataset.column_names[3]: "label"
})
test_dataset = test_dataset.remove_columns("text_2")

#### below - similar/same as original load_data file

# uses CLIP's preprocessor
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# preprocess each example
def preprocess_muse(example):
    # Build full image paths for each image ID in the batch
    image_paths = [
        os.path.join("./data/muse/images", image_id + ".jpg") for image_id in example["image"]
    ]
    
    # Open all images in the batch
    images = [Image.open(path).convert("RGB") for path in image_paths]

    # Process with CLIPProcessor
    inputs = processor(
        text=example["text"],
        images=images,
        return_tensors="np",  # Keep it NumPy for easier manipulation
        padding="max_length",
        truncation=True,
        max_length=77,
    )
    
    return {
        # fields for clip requirements (inputs, labels)
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "pixel_values": inputs["pixel_values"],
        "label": example["label"],

        # fields for bert and other things
        "text_list": example["text"],
        "image_list": example["image"],
        "label_list": example["label"],
        "samples": inputs["pixel_values"],  # already preprocessed image
    }

# Get the directory this script is in (i.e., the "data" folder)
current_dir = os.path.dirname(os.path.abspath(__file__))
processed_data_dir = os.path.join(current_dir, "muse_processed")

if not os.path.exists(processed_data_dir):
    # this doesn't actually batch the data, it just processes it in batches
    processed_train_dataset = train_dataset.map(preprocess_muse, batched=True, num_proc=4)
    processed_val_dataset = val_dataset.map(preprocess_muse, batched=True, num_proc=4)
    processed_test_dataset = test_dataset.map(preprocess_muse, batched=True, num_proc=4)

    processed_train_dataset.set_format(
        type="numpy",
        columns=["input_ids", "attention_mask", "pixel_values", "label", "text_list", "image_list", "label_list", "samples"]
    )
    processed_val_dataset.set_format(
        type="numpy",
        columns=["input_ids", "attention_mask", "pixel_values", "label", "text_list", "image_list", "label_list", "samples"]
    )
    processed_test_dataset.set_format(
        type="numpy",
        columns=["input_ids", "attention_mask", "pixel_values", "label", "text_list", "image_list", "label_list", "samples"]
    )

    full_dataset = DatasetDict({
        "train": processed_train_dataset,
        "val": processed_val_dataset,
        "test": processed_test_dataset,
    })

    full_dataset.save_to_disk(processed_data_dir)
    # processed_train_dataset.save_to_disk(os.path.join(processed_data_dir, "train"))
    # processed_val_dataset.save_to_disk(os.path.join(processed_data_dir, "val"))
    # processed_test_dataset.save_to_disk(os.path.join(processed_data_dir, "test"))

else:
    # load processed data if it exists
    full_dataset = load_from_disk(processed_data_dir)
    # processed_train_dataset = load_from_disk(os.path.join(processed_data_dir, "train"))
    # processed_val_dataset = load_from_disk(os.path.join(processed_data_dir, "val"))
    # processed_test_dataset = load_from_disk(os.path.join(processed_data_dir, "test"))