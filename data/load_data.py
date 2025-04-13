from datasets import load_dataset, load_from_disk
from transformers import CLIPProcessor
import os

# Load the dataset
dataset = load_dataset("coderchen01/MMSD2.0", name="mmsd-v2")

# uses CLIP's preprocessor
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Define a function to preprocess each example
def preprocess(example):
    # Process text and image into tensor format
    inputs = processor(
        text=example["text"],
        images=example["image"],
        return_tensors="np",  # Or "tf" for TensorFlow if you prefer
        padding="max_length",
        truncation=True,
        max_length=77,
    )
    
    # Ensure we return the processed tensors in the correct shape
    return {
        "pixel_values": inputs["pixel_values"],  # This is already a batch of pixel values
        "input_ids": inputs["input_ids"],        # Tokenized text
        "attention_mask": inputs["attention_mask"],  # Attention mask
        "label": example["label"],  # Original label
    }

# Check if the processed dataset already exists to avoid reprocessing
if not os.path.exists("mmsd_processed"):
    # Apply preprocessing to the entire dataset
    dataset = dataset.map(preprocess, batched=True, num_proc=4)  # Adjust `num_proc` as needed
    
    # Save the processed dataset to disk
    dataset.save_to_disk("mmsd_processed")
else:
    # Load the processed dataset from disk if it exists
    dataset = load_from_disk("mmsd_processed")
