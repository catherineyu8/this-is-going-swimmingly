from datasets import Dataset
import pandas as pd
from PIL import Image
import os
from transformers import CLIPProcessor
import json

# create Dataset with out data
with open("data/our_data/captions_labels.json", "r") as f:
    data = json.load(f)

df = pd.DataFrame(data)
df["image_path"] = df["image_filename"].apply(lambda x: os.path.join("data/our_data/images", x))
dataset = Dataset.from_pandas(df)

# process it
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def preprocess(examples):
    # load images
    images = [Image.open(path).convert("RGB") for path in examples["image_path"]]
    
    # process image and caption
    inputs = processor(
        text=examples["caption"],
        images=images,
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=77,
    )
    
    return {
        # fields: label, input_ids, attention_mask, pixel_values, text_list
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "pixel_values": inputs["pixel_values"],
        "label": examples["label"],
        "text_list": examples["caption"],
    }
# Get the directory this script is in (i.e., the "data" folder)
current_dir = os.path.dirname(os.path.abspath(__file__))
processed_data_dir = os.path.join(current_dir, "our_processed")

if not os.path.exists(processed_data_dir):
    # this doesn't actually batch the data, it just processes it in batches
    # processed_train_dataset = train_dataset.map(preprocess, batched=True, num_proc=4)
    # processed_val_dataset = val_dataset.map(preprocess, batched=True, num_proc=4)
    processed_dataset = dataset.map(preprocess, batched=True, num_proc=4)

    processed_dataset.set_format(
        type="numpy",
        columns=["input_ids", "attention_mask", "pixel_values", "label", "text_list"]
    )

    processed_dataset.save_to_disk(processed_data_dir)

else:
    # load processed data if it exists
    processed_dataset.load_from_disk(processed_data_dir)