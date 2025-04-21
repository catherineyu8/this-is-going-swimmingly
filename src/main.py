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

    print("Loaded data from disk.")

    # two_clip_batches = dataset.to_tf_dataset(
    #     columns=["input_ids", "attention_mask", "pixel_values"],
    #     label_cols="label",
    #     shuffle=True,
    #     batch_size=32,
    # )

    # split data into train/test and convert to tf.data.Dataset
    train_dataset = dataset["train"].to_tf_dataset(
        columns=["input_ids", "attention_mask", "pixel_values"],
        label_cols="label",
        shuffle=False,
        batch_size=32,
    )
    text_list_train = dataset["train"]["text_list"]

    test_dataset = dataset["test"].to_tf_dataset(
        columns=["input_ids", "attention_mask", "pixel_values"],
        label_cols="label",
        shuffle=False,
        batch_size=32,
    )
    text_list_test = dataset["test"]["text_list"]

    # for inputs, _ in test_dataset.take(1):
    #     print("Batch structure:")
    #     print("input_ids shape:", inputs['input_ids'].shape)
    #     print("attention_mask shape:", inputs['attention_mask'].shape)
    #     print("pixel_values shape:", inputs['pixel_values'].shape)

    # initialize model, do dummy forward pass to build model
    model = RackleMuffin()

    dummy_inputs = {
        "input_ids": tf.zeros((32, 77), dtype=tf.int32),
        "attention_mask": tf.ones((32, 77), dtype=tf.int32),
        "pixel_values": tf.zeros((32, 3, 224, 224), dtype=tf.float32),
    }
    dummy_text_list = np.array(["this is a caption"] * 32)
    _ = model(dummy_inputs, dummy_text_list)

    # train model
    train(model, train_dataset, text_list_train)

    # load model weights and test
    model.load_weights("racklemuffin_weights.h5") 
    test(model, test_dataset, text_list_test)
    

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