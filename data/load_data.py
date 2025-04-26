from datasets import load_dataset, load_from_disk
from transformers import CLIPProcessor
import os

# load dataset
dataset = load_dataset("coderchen01/MMSD2.0", name="mmsd-v2")

# get just the first 64 examples: 2 batches
# dataset_chunk = dataset["train"].select(range(64))

# uses CLIP's preprocessor
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# preprocess each example
def preprocess(example):
    # process with CLIPProcessor
    inputs = processor(
        text=example["text"],
        images=example["image"],
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
processed_data_dir = os.path.join(current_dir, "mmsd_processed")

# preprocess small chunk (2 batches)
# small_chunk = dataset_chunk.map(preprocess)
# small_chunk.set_format(
#     type="numpy",
#     columns=["input_ids", "attention_mask", "pixel_values", "label", "text_list", "image_list", "label_list", "samples"]
# )
# small_chunk.save_to_disk(processed_data_dir)



if not os.path.exists(processed_data_dir):
    # this doesn't actually batch the data, it just processes it in batches
    dataset = dataset.map(preprocess, batched=True, num_proc=4)

    dataset.set_format(
        type="numpy",
        columns=["input_ids", "attention_mask", "pixel_values", "label", "text_list", "image_list", "label_list", "samples"]
    )
    dataset.save_to_disk(processed_data_dir)
else:
    # load processed data if it exists
    dataset = load_from_disk(processed_data_dir)

