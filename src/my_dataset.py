from datasets import load_dataset, load_from_disk
import tensorflow as tf
import json
import os
from PIL import Image

# dataset = load_dataset("coderchen01/MMSD2.0", name="mmsd-v2")


class MyTFDataset(tf.data.Dataset):
    @classmethod
    def load_or_process(cls, mode, text_name, limit=None, save_dir="./processed_data"):
        
        # Build path to "mmsd_processed" inside the data folder, relative to this location
        current_dir = os.path.dirname(os.path.abspath(__file__))
        processed_data_dir = os.path.join(current_dir, "../data/mmsd_processed")
        data_dir = os.path.join(current_dir, "../data", "mmsd")

        # Check if the processed dataset already exists to avoid reprocessing
        if os.path.exists(processed_data_dir):
            # Load the processed dataset from disk if it exists
            # processed_dataset = load_from_disk(processed_data_dir)
            print("data already processed, loading now!")
            return tf.data.Dataset.load(processed_data_dir)
        else:
            print("data not processed yet, processing now!")
            # Apply preprocessing to the entire dataset
            raw_dataset = load_dataset("coderchen01/MMSD2.0", name="mmsd-v2")
            processed_dataset = cls.extract_attributes(raw_dataset, data_dir)
            # processed_dataset = processed_dataset.map(cls.preprocess_sample, num_parallel_calls=tf.data.AUTOTUNE)

            # Save the processed dataset to disk
            processed_dataset.save(processed_data_dir)

            return processed_dataset
    
    @classmethod   
    def extract_attributes(cls, raw_data, data_dir):
        """
        Extract relevant attributes from raw dataset (text, image path, label, image ID).
        This will be used for processing the dataset into a more useful format.
        """
        dataset = []
        
        for example in raw_data['train']:  # Assuming you want to extract from the 'train' split
            text = example['text']
            label = example['label']
            image_id = example['image_id']

            # Assuming images are stored under a specific directory; adjust as needed
            image_path = os.path.join(data_dir, "dataset_image", f"{image_id}.jpg")
            
            dataset.append({
                'text': text,
                'image_path': image_path,
                'label': label,
                'image_id': image_id
            })

        return dataset

    # def __new__(cls, mode, text_name, limit=None):
    #     # parse data from file into a dict, maping image_id to the corresponding text, label, and image_path
    #     dataset = cls.load_data(raw_dataset_dir)

    #     # generate the dataset using the generator
    #     return tf.data.Dataset.from_generator(
    #         lambda: cls.generator(dataset),
    #         output_signature=(
    #             tf.TensorSpec(shape=(), dtype=tf.string),      # text
    #             tf.TensorSpec(shape=(), dtype=tf.string),      # image path (we'll load image later)
    #             tf.TensorSpec(shape=(), dtype=tf.int32),       # label
    #             tf.TensorSpec(shape=(), dtype=tf.int32),       # id
    #         )
    #     )

    # @staticmethod
    # def load_data(raw_dataset_path):
    #     cnt = 0
    #     data_set = dict()
    #     data_path = os.path.join(WORKING_PATH, text_name, f"{mode}.json") # open the train/val/test data file

    #     with open(data_path, 'r', encoding='utf-8') as f:
    #         datas = json.load(f)
    #         for data in datas:
    #             if limit is not None and cnt >= limit:
    #                 break

    #             image_id = int(data["image_id"])
    #             sentence = data["text"]
    #             label = int(data["label"])

    #             image_path = os.path.join(WORKING_PATH, "dataset_image", f"{image_id}.jpg") # get image filename
    #             if os.path.isfile(image_path):
    #                 # map image_id to its text, label, and image_path
    #                 data_set[image_id] = {
    #                     "text": sentence,
    #                     "label": label,
    #                     "image_path": image_path
    #                 }
    #                 cnt += 1
    #     return data_set
    
    # apply transformations to the images. to be used after creating a dataset
    @staticmethod
    def preprocess_sample(text, image_path, label, image_id):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [224, 224])
        image = tf.cast(image, tf.float32) / 255.0
        image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225] # normalize
        return text, image, label, image_id

    @classmethod
    def generator(cls, data_dict):
        """
        Generator function to yield processed data (text, image_path, label, image_id).
        """
        for entry in data_dict:
            yield entry['text'], entry['image_path'], entry['label'], entry['image_id']

    @classmethod
    def load_from_generator(cls, raw_data):
        """
        Use the generator to create a TensorFlow Dataset from raw data.
        """
        return tf.data.Dataset.from_generator(
            lambda: cls.generator(raw_data),
            output_signature=(
                tf.TensorSpec(shape=(), dtype=tf.string),  # text
                tf.TensorSpec(shape=(), dtype=tf.string),  # image path
                tf.TensorSpec(shape=(), dtype=tf.int32),   # label
                tf.TensorSpec(shape=(), dtype=tf.int32)    # image_id
            )
        )