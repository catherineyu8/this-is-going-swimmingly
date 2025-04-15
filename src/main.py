from datasets import load_from_disk
import tensorflow as tf
from tqdm import tqdm
from transformers import CLIPProcessor
from model import RackleMuffin
import numpy as np

def main():
    # Load the processed dataset
    dataset = load_from_disk("../data/mmsd_processed_chunk")

    print("loaded small data chunk from disk")


    two_batches = dataset.to_tf_dataset(
        columns=["input_ids", "attention_mask", "pixel_values", "label", "text_list", "image_list", "label_list", "samples"],
        label_cols="label",
        shuffle=True,
        batch_size=1,
    )

    # train_dataset = dataset["train"].to_tf_dataset(
    #     columns=["input_ids", "attention_mask", "pixel_values", "label", "text_list", "image_list", "label_list", "samples"],
    #     label_cols="label",
    #     shuffle=True,
    #     batch_size=32,
    # )

    # print("created training dataset")

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

    # breaking:
    # trying to take a batch (32) of examples
    # then, it tries to stack each dimension together (all text, all images)
    # it breaks because the text is different shapes
    for batch in tqdm(two_batches, desc="Training"):
        inputs = {
            "input_ids": batch["input_ids"],
            "attention_matrix": batch["attention_matrx"],
            "pixel_values": batch["pixel_values"]
        }

        text_list = batch["text_list"]
        samples = batch["samples"]

        model(inputs)

        # print for debugging
        print(f"First text in the batch: {text_list[0]}")
        

    # Optionally save the model after every epoch
    model.save_pretrained("./saved_model")


if __name__ == '__main__':
    main()