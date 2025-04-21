from datasets import load_from_disk
import tensorflow as tf
from tqdm import tqdm
from transformers import CLIPProcessor
from model import RackleMuffin
import numpy as np
from train import train, test

def main():
    # Load the processed dataset
    print(tf.config.list_physical_devices('GPU'))
    print(tf.sysconfig.get_build_info())

    dataset = load_from_disk("data/mmsd_processed")

    print("loaded small data chunk from disk")

    # two_clip_batches = dataset.to_tf_dataset(
    #     columns=["input_ids", "attention_mask", "pixel_values"],
    #     label_cols="label",
    #     shuffle=True,
    #     batch_size=32,
    # )

    train_dataset = dataset["train"].to_tf_dataset(
        columns=["input_ids", "attention_mask", "pixel_values"],
        label_cols="label",
        shuffle=False,
        batch_size=32,
    )
    text_list_train = dataset["train"]["text_list"]

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

    # Convert testing split to tf.data.Dataset
    test_dataset = dataset["test"].to_tf_dataset(
        columns=["input_ids", "attention_mask", "pixel_values"],
        label_cols="label",
        shuffle=False,
        batch_size=32,
    )
    text_list_test = dataset["test"]["text_list"]
    print("created test dataset")

    for inputs, _ in test_dataset.take(1):
        print("Batch structure:")
        print("input_ids shape:", inputs['input_ids'].shape)
        print("attention_mask shape:", inputs['attention_mask'].shape)
        print("pixel_values shape:", inputs['pixel_values'].shape)


    # attempt to run train! COMMENT OUT TEST STUFF FOR TRAIN
    model = RackleMuffin()
    train(model, train_dataset, text_list_train)

    # attempt to run test!

    # Do a dummy forward pass to build the model
    dummy_inputs = {
        "input_ids": tf.zeros((32, 77), dtype=tf.int32),
        "attention_mask": tf.ones((32, 77), dtype=tf.int32),
        "pixel_values": tf.zeros((32, 3, 224, 224), dtype=tf.float32),
    }
    dummy_text_list = np.array(["this is a caption"] * 32)
    _ = model(dummy_inputs, dummy_text_list)

    # Now load the weights and call test
    model.load_weights("racklemuffin_weights.h5") 
    test(model, test_dataset, text_list_test)
    

    # Create the model

    # breaking:
    # trying to take a batch (32) of examples
    # then, it tries to stack each dimension together (all text, all images)
    # it breaks because the text is different shapes
    # Group into batches of 32
    # batch_size = 32
    # text_list_batches = [text_list_train[i:i + batch_size] for i in range(0, len(text_list_train), batch_size)]

    # for (batch, text_batch) in tqdm(zip(train_dataset, text_list_batches), desc="Training"):
    #     clip_inputs, labels = batch
    #     # supposedly should be able to get text_batch now?

    #     # print("first clip_input_ids:", clip_inputs["input_ids"][0])
    #     # print("first text in batch:", text_batch[0])

    #     model(clip_inputs, text_batch)
    #     break
        
    # Optionally save the model after every epoch
    # model.save_pretrained("./saved_model")


if __name__ == '__main__':
    main()