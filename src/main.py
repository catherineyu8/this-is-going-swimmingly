from datasets import load_from_disk
import tensorflow as tf
from tqdm import tqdm
from transformers import CLIPProcessor
from model import RackleMuffin
import numpy as np

# def custom_collate_fn(batch):
#     # Stack uniform tensors
#     pixel_values = np.stack([example["pixel_values"] for example in batch])
#     input_ids = np.stack([example["input_ids"] for example in batch])
#     attention_mask = np.stack([example["attention_mask"] for example in batch])
#     labels = np.stack([example["label"] for example in batch])

#     # Keep these as lists (not stacking)
#     text_list = [example["text_list"] for example in batch]
#     image_list = [example["image_list"] for example in batch]
#     label_list = [example["label_list"] for example in batch]
#     samples = [example["samples"] for example in batch]

#     return {
#         "pixel_values": pixel_values,
#         "input_ids": input_ids,
#         "attention_mask": attention_mask,
#         "label": labels,
#         "text_list": text_list,
#         "image_list": image_list,
#         "label_list": label_list,
#         "samples": samples
    # }

def main():
    # Load the processed dataset
    dataset = load_from_disk("data/mmsd_processed")

    print("Loaded data from disk")

    # Convert training split to tf.data.Dataset
    train_dataset = dataset["train"].to_tf_dataset(
        columns=["input_ids", "attention_mask", "pixel_values", "label", "text_list", "image_list", "label_list", "samples"],
        label_cols="label",
        shuffle=True,
        batch_size=32,
    )

    print("created training dataset")

    # Convert validation split to tf.data.Dataset
    # val_dataset = dataset["validation"].to_tf_dataset(
    #     columns=["input_ids", "attention_mask", "pixel_values", "label", "text_list", "image_list", "label_list", "samples"],
    #     label_cols="label",
    #     shuffle=False,
    #     batch_size=32,
    #     collate_fn=tf_collate_fn  # You can define one if needed
    # )

    # print("created val dataset")

    # # Convert testing split to tf.data.Dataset
    # test_dataset = dataset["test"].to_tf_dataset(
    #     columns=["input_ids", "attention_mask", "pixel_values", "label", "text_list", "image_list", "label_list", "samples"],
    #     label_cols="label",
    #     shuffle=False,
    #     batch_size=32,
    #     collate_fn=tf_collate_fn  # You can define one if needed
    # )

    # print("created test dataset")

    # Create the model
    model = RackleMuffin()

    # Training loop
    for batch in tqdm(train_dataset, desc="Training"):
        inputs = {
            "input_ids": batch["input_ids"],
            "attention_matrix": batch["attention_matrx"],
            "pixel_values": batch["pixel_values"]
        }

        text_list = batch["text_list"]
        samples = batch["samples"]

        model(inputs)

        # You can also use the additional fields (text_list, image_list, etc.) for logging, debugging, etc.
        # For example, printing out the first text sample for debugging:
        print(f"First text in the batch: {text_list[0]}")
        
        # If you need to track any additional metrics or store extra data, you can do that here.
        # For example, tracking losses or printing image paths.
        
        # Forward pass for the model and loss calculation (if applicable)
        # outputs = model(inputs)  # If you need the model outputs for loss calculation
        
        # Calculate gradients and apply them if you're doing custom optimization steps
        # loss = outputs.loss  # This depends on your model's output format
        # grads = tape.gradient(loss, model.trainable_variables)
        # optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Optionally print loss for tracking
        # print(f"Loss: {loss.numpy()}")

    # Optionally save the model after every epoch
    model.save_pretrained("./saved_model")


if __name__ == '__main__':
    main()