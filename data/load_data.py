# from datasets import load_dataset
# from transformers import CLIPProcessor

# dataset = load_dataset("coderchen01/MMSD2.0", name="mmsd-v2")
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# def preprocess(example):
#     print("Processing a batch...")
#     inputs = processor(
#         text=example["text"],
#         images=example["image"],
#         return_tensors="np",  # or "tf" if using TensorFlow backend
#         padding="max_length",
#         truncation=True,
#         max_length=77,
#     )
#     return {
#         "pixel_values": inputs["pixel_values"][0],  # unwrap batch dim
#         "input_ids": inputs["input_ids"][0],
#         "attention_mask": inputs["attention_mask"][0],
#         "label": example["label"],
#     }

# dataset = dataset.map(preprocess)

# # Apply preprocessing to the entire dataset
# #dataset = dataset.map(preprocess, batched=True)

# # Save the processed dataset to disk
# dataset.save_to_disk("mmsd_processed")

# tf_dataset = dataset["train"].to_tf_dataset(
#     columns=["pixel_values", "input_ids", "attention_mask", "label"],
#     batch_size=32,
#     shuffle=True,
# )


from datasets import load_dataset, load_from_disk
from transformers import CLIPProcessor
import os

# Load the dataset
dataset = load_dataset("coderchen01/MMSD2.0", name="mmsd-v2")

# Initialize the CLIPProcessor
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

# Example to show how to access data
print(dataset["train"][0])  # Print the first item from the training set

# Access the datasets in a memory-efficient way
train_dataset = dataset["train"]
val_dataset = dataset["validation"]
test_dataset = dataset["test"]

# For example, if you want to use a DataLoader in TensorFlow or PyTorch:
# You can convert this dataset to a DataLoader format or use TensorFlow's Dataset API.
