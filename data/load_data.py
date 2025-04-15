from datasets import load_dataset, load_from_disk
from transformers import CLIPProcessor
import os

# Load the dataset
dataset = load_dataset("coderchen01/MMSD2.0", name="mmsd-v2")

# uses CLIP's preprocessor
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# # Define a function to preprocess each example
# def preprocess(example):
#     # Process text and image into tensor format
#     inputs = processor(
#         text=example["text"],
#         images=example["image"],
#         return_tensors="np",  # Or "tf" for TensorFlow if you prefer
#         padding="max_length",
#         truncation=True,
#         max_length=77,
#     )
    
#     # Ensure we return the processed tensors in the correct shape
#     return {
#         "pixel_values": inputs["pixel_values"],  # This is already a batch of pixel values
#         "input_ids": inputs["input_ids"],        # Tokenized text
#         "attention_mask": inputs["attention_mask"],  # Attention mask
#         "label": example["label"],  # Original label
#     }

# Preprocessing function
def preprocess(example):
    # Process with CLIPProcessor
    inputs = processor(
        text=example["text"],
        images=example["image"],
        return_tensors="np",  # Keep it NumPy for easier manipulation
        padding="max_length",
        truncation=True,
        max_length=77,
    )
    
    return {
        # CLIP fields
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "pixel_values": inputs["pixel_values"],
        "label": example["label"],

        # Mimicking collate_func fields from RCLMuFN MyDataset Class
        "text_list": example["text"],
        "image_list": example["image"],
        "label_list": example["label"],
        "samples": inputs["pixel_values"],  # already preprocessed image
    }

# Get the directory this script is in (i.e., the "data" folder)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Build path to "mmsd_processed" inside the same folder
processed_data_dir = os.path.join(current_dir, "mmsd_processed")

# Check if the processed dataset already exists to avoid reprocessing
if not os.path.exists(processed_data_dir):
    # Apply preprocessing to the entire dataset
    dataset = dataset.map(preprocess, batched=True, num_proc=4)  # Adjust `num_proc` as needed

    dataset.set_format(
        type="numpy",
        columns=["input_ids", "attention_mask", "pixel_values", "label", "text_list", "image_list", "label_list", "samples"]
    )

    # Save the processed dataset to disk
    dataset.save_to_disk(processed_data_dir)
else:
    # Load the processed dataset from disk if it exists
    dataset = load_from_disk(processed_data_dir)
