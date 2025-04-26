from datasets import load_dataset, load_from_disk, DatasetDict, Dataset
from transformers import CLIPProcessor
from PIL import Image
import numpy as np
import os
import pandas as pd
import io

# load dataset
dataset = load_dataset("tsystems/flickr8k")

# split data into 80 test, 20 train
train_temp_split = dataset['train'].train_test_split(test_size=0.2, shuffle=True, seed=42)
# further split the 20 train into half test half val (10 ea)
test_valid_split = train_temp_split['test'].train_test_split(test_size=0.5, shuffle=True, seed=42)

# Create the final dataset dictionary
split_dataset = DatasetDict({
    'train': train_temp_split['train'],
    'test': test_valid_split['train'],
    'validation': test_valid_split['test']
})

# converts PIL Image to bytes
def pil_image_to_bytes(image):
    with io.BytesIO() as output:
        image.save(output, format="JPEG")
        return output.getvalue()

# process each split
processed_splits = {}
# for split in ['train', 'test', 'validation']:
split = 'test'
# convert to pandas DataFrame
df = pd.DataFrame(split_dataset[split])
# convert PIL Images to bytes
df['image'] = df['image'].apply(pil_image_to_bytes)
# add a sarcasm label (0 for not sarcastic)
df['label'] = 0
# convert pandas DataFrame to Hugging Face dataset format
processed_splits[split] = Dataset.from_pandas(df)

# train_dataset = processed_splits['train']
# val_dataset = processed_splits['validation']
test_dataset = processed_splits['test']

# rename columns, then remove second caption (not relevant for our task)
# train_dataset = train_dataset.rename_columns({
#     train_dataset.column_names[3]: "text",
# })
# train_dataset = train_dataset.remove_columns("captions")

# val_dataset = val_dataset.rename_columns({
#     val_dataset.column_names[3]: "text",
# })
# val_dataset = val_dataset.remove_columns("captions")

test_dataset = test_dataset.rename_columns({
    test_dataset.column_names[3]: "text",
})
test_dataset = test_dataset.remove_columns("captions")


# uses CLIP's preprocessor
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# preprocess each example
def preprocess(batch):
    texts = batch["text"]
    images = [Image.open(io.BytesIO(img)).convert("RGB") for img in batch["image"]]

    inputs = processor(
        text=texts,
        images=images,
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=77,
    )

    return {
        # fields for clip requirements (inputs, labels)
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "pixel_values": inputs["pixel_values"],
        "label": batch["label"],

        # fields for bert and other things
        "text_list": texts,
        "image_list": batch["image"],
        "label_list": batch["label"],
        "samples": inputs["pixel_values"],
    }

# Get the directory this script is in (i.e., the "data" folder)
current_dir = os.path.dirname(os.path.abspath(__file__))
processed_data_dir = os.path.join(current_dir, "flickr_processed")

if not os.path.exists(processed_data_dir):
    # this doesn't actually batch the data, it just processes it in batches
    # processed_train_dataset = train_dataset.map(preprocess, batched=True, num_proc=4)
    # processed_val_dataset = val_dataset.map(preprocess, batched=True, num_proc=4)
    processed_test_dataset = test_dataset.map(preprocess, batched=True, num_proc=4)

    # processed_train_dataset.set_format(
    #     type="numpy",
    #     columns=["input_ids", "attention_mask", "pixel_values", "label", "text_list", "image_list", "label_list", "samples"]
    # )
    # processed_val_dataset.set_format(
    #     type="numpy",
    #     columns=["input_ids", "attention_mask", "pixel_values", "label", "text_list", "image_list", "label_list", "samples"]
    # )
    processed_test_dataset.set_format(
        type="numpy",
        columns=["input_ids", "attention_mask", "pixel_values", "label", "text_list", "image_list", "label_list", "samples"]
    )

    # full_dataset = DatasetDict({
    #     "train": processed_train_dataset,
    #     "val": processed_val_dataset,
    #     "test": processed_test_dataset,
    # })

    # full_dataset.save_to_disk(processed_data_dir)
    test_dataset.save_to_disk(processed_data_dir)

    
else:
    # load processed data if it exists
    # full_dataset = load_from_disk(processed_data_dir)
    test_dataset.save_to_disk(processed_data_dir)