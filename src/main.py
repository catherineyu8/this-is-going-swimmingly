from datasets import load_from_disk, concatenate_datasets
import tensorflow as tf
from tqdm import tqdm
from transformers import CLIPProcessor
from model import RackleMuffin
import numpy as np
from train import train, test
import argparse
import sys
from collections import Counter
import random

def main():
    args = parse_args()

    print(tf.config.list_physical_devices('GPU'))
    print(tf.sysconfig.get_build_info())

    # initialize model, do dummy forward pass to build model
    model = RackleMuffin()

    dummy_inputs = {
        "input_ids": tf.zeros((32, 77), dtype=tf.int32),
        "attention_mask": tf.ones((32, 77), dtype=tf.int32),
        "pixel_values": tf.zeros((32, 3, 224, 224), dtype=tf.float32),
    }
    dummy_text_list = ["this is a caption"] * 32
    _ = model(dummy_inputs, dummy_text_list)

    # TRAIN model
    if args.mode == "train":
        '''
        # clip the train dataset to only 60 to make sure it works locally
        batched_mmsd_train_clipped = batched_mmsd_train.unbatch().take(60).batch(32)
        mmsd_textlist_train_clipped = mmsd_textlist_train[:60]
        train(model, batched_mmsd_train_clipped, mmsd_textlist_train_clipped)
        '''

        # Load the processed MMSD 2.0 dataset
        mmsd_dataset = load_from_disk("data/mmsd_processed")
        print("Loaded MMSD data from disk.")

        # shuffle the MMSD training dataset
        mmsd_dataset["train"] = mmsd_dataset["train"].shuffle(seed=42)

        # get train split of MMSD data
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

        train(model, batched_mmsd_train, mmsd_textlist_train)

    # load weights and TEST model
    elif args.mode == "test":
        '''
        # clip test dataset to make sure it works locally
        batched_mmsd_test_clipped = batched_mmsd_test.unbatch().take(60).batch(32)
        mmsd_textlist_test_clipped = mmsd_textlist_test[:60]
        test(model, batched_mmsd_test_clipped, mmsd_textlist_test_clipped)
        '''
        
        model.load_weights("saved_models/bert_resnet_clip_frozen/racklemuffin_weights_epoch_2.h5")
        
        if args.dataset == "muse_flickr":
            # get MUSE and FLICKR data
            muse_dataset = load_from_disk("data/muse_processed")["train"]
            flickr_dataset = load_from_disk("data/flickr_processed")

            # remove unneeded cols from dataset so we can combine them
            # only save these columns: text, label, input_ids, attention_mask, pixel_values, text_list
            muse_dataset = muse_dataset.remove_columns(["image_list", "label_list", "samples", "image"])
            flickr_dataset = flickr_dataset.remove_columns(["image_list", "label_list", "samples", "image_filename", "image"])
            
            # combine and shuffle data
            combined_dataset = concatenate_datasets([muse_dataset, flickr_dataset])
            combined_dataset = combined_dataset.shuffle(seed=42)

            # extract right formats for model
            batched_combined_data = combined_dataset.to_tf_dataset(
                columns=["input_ids", "attention_mask", "pixel_values"],
                label_cols="label",
                shuffle=False,
                batch_size=32,
            )
            combined_textlist = combined_dataset["text_list"].tolist()

            test(model, batched_combined_data, combined_textlist)
        
        elif args.dataset == "mmsd2.0":
            # Load the processed MMSD 2.0 dataset
            mmsd_dataset = load_from_disk("data/mmsd_processed")
            print("Loaded MMSD data from disk.")

            # get test/val splits of MMSD data
            batched_mmsd_test = mmsd_dataset["test"].to_tf_dataset(
                columns=["input_ids", "attention_mask", "pixel_values"],
                label_cols="label",
                shuffle=False,
                batch_size=32,
            )
            mmsd_textlist_test = mmsd_dataset["test"]["text_list"].tolist()

            # batched_mmsd_val = mmsd_dataset["validation"].to_tf_dataset(
            #     columns=["input_ids", "attention_mask", "pixel_values"],
            #     label_cols="label",
            #     shuffle=False,
            #     batch_size=32,
            # )
            # mmsd_textlist_val = mmsd_dataset["validation"]["text_list"].tolist()

            test(model, batched_mmsd_test, mmsd_textlist_test)

        elif args.dataset == "us":
            our_dataset = load_from_disk("data/our_processed")

            # extract right formats for model
            batched_data = our_dataset.to_tf_dataset(
                columns=["input_ids", "attention_mask", "pixel_values"],
                label_cols="label",
                shuffle=False,
                batch_size=32,
            )
            textlist = our_dataset["text_list"].tolist()

            test(model, batched_data, textlist)
        elif args.dataset == "sarcnet":
            sarcnet_dataset = load_from_disk("data/sarcnet_processed")

            # remove unneeded columns -- any other unneeded columns?
            sarcnet_dataset = sarcnet_dataset.remove_columns(["samples", "label_list", "image_list", "text_label", "image_label"])

            batched_sarcnet_test = sarcnet_dataset["test"].to_tf_dataset(
                columns=["input_ids", "attention_mask", "pixel_values"],
                label_cols="label",
                shuffle=False,
                batch_size = 32,
            )
            sarcnet_textlist_test = sarcnet_dataset["test"]["text_list"].tolist()

            test(model, batched_sarcnet_test, sarcnet_textlist_test)

        else:
            print("Invalid dataset. Use --dataset mmsd2.0 or --dataset muse_flickr or --dataset us.")

    else:
        print("Invalid mode. Use --mode train or --mode test.")
    

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
        choices=["mmsd2.0", "muse_flickr", "us", "sarcnet"],
        help="Dataset to use when testing."
    )

    args = parser.parse_args()

    # Require --dataset only when mode is test
    if args.mode == "test" and args.dataset is None:
        parser.error("--dataset is required when mode is 'test'")

    return args

if __name__ == "__main__":
    main()