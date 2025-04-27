from datasets import load_from_disk
import tensorflow as tf
from tqdm import tqdm
from transformers import CLIPProcessor
from model import RackleMuffin
import numpy as np
from train import train, test
import argparse
import sys
from collections import Counter

def main():
    args = parse_args()

    # Load the processed dataset
    print(tf.config.list_physical_devices('GPU'))
    print(tf.sysconfig.get_build_info())

    mmsd_dataset = load_from_disk("data/mmsd_processed")

    print("Loaded data from disk.")

    # shuffle the MMSD training dataset
    mmsd_dataset["train"] = mmsd_dataset["train"].shuffle(seed=42)

    # split MMSD data into train/test/val
    # tf.data.Dataset contains input format needed for CLIP
        # pixel_values used for RESNET
    # text list contains input for BERT
    batched_mmsd_train = mmsd_dataset["train"].to_tf_dataset(
        columns=["input_ids", "attention_mask", "pixel_values"],
        label_cols="label",
        shuffle=False,
        batch_size=32,
    )
    mmsd_textlist_train = mmsd_dataset["train"]["text_list"].tolist()

    batched_mmsd_test = mmsd_dataset["test"].to_tf_dataset(
        columns=["input_ids", "attention_mask", "pixel_values"],
        label_cols="label",
        shuffle=False,
        batch_size=32,
    )
    mmsd_textlist_test = mmsd_dataset["test"]["text_list"].tolist()

    batched_mmsd_val = mmsd_dataset["validation"].to_tf_dataset(
        columns=["input_ids", "attention_mask", "pixel_values"],
        label_cols="label",
        shuffle=False,
        batch_size=32,
    )
    mmsd_textlist_val = mmsd_dataset["validation"]["text_list"].tolist()

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
    dummy_text_list = ["this is a caption"] * 32
    _ = model(dummy_inputs, text_data=dummy_text_list)

    # TRAIN model
    if args.mode == "train":
        '''
        # clip the train dataset to only 60 to make sure it works locally
        batched_mmsd_train_clipped = batched_mmsd_train.unbatch().take(60).batch(32)
        mmsd_textlist_train_clipped = mmsd_textlist_train[:60]
        train(model, batched_mmsd_train_clipped, mmsd_textlist_train_clipped)
        '''
        train(model, batched_mmsd_train, mmsd_textlist_train)

    # load weights and TEST model
    elif args.mode == "test":
        '''
        # clip test dataset to make sure it works locally
        batched_mmsd_test_clipped = batched_mmsd_test.unbatch().take(60).batch(32)
        mmsd_textlist_test_clipped = mmsd_textlist_test[:60]
        test(model, batched_mmsd_test_clipped, mmsd_textlist_test_clipped)
        '''
        
        model.load_weights("racklemuffin_weights.h5")
        
        if args.dataset == "muse":
            # get MUSE data
            muse_dataset = load_from_disk("data/muse_processed")
            
            batched_muse_test = muse_dataset["test"].to_tf_dataset(
                columns=["input_ids", "attention_mask", "pixel_values"],
                label_cols="label",
                shuffle=False,
                batch_size=32,
            )
            muse_textlist_test = muse_dataset["test"]["text_list"].tolist()

            test(model, batched_muse_test, muse_textlist_test)
        
        elif args.dataset == "mmsd2.0":
            test(model, batched_mmsd_test, mmsd_textlist_test)

        else:
            print("Invalid dataset. Use --dataset mmsd2.0 or --dataset muse.")

    else:
        print("Invalid mode. Use --mode train or --mode test.")
    

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


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode", 
        choices=["train", "test"], 
        required=True,
        help="Whether to train or test the model."
    )

    parser.add_argument(
        "--dataset", 
        choices=["mmsd2.0", "muse"], 
        help="Dataset to use when testing."
    )

    args = parser.parse_args()

    # Require --dataset only when mode is test
    if args.mode == "test" and args.dataset is None:
        parser.error("--dataset is required when mode is 'test'")

    return args

if __name__ == "__main__":
    main()