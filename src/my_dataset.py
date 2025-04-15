import tensorflow as tf
import json
import os
from PIL import Image
import numpy as np

WORKING_PATH = "./MMSD2.0dataset/data"

class MyTFDataset(tf.data.Dataset):
    def __new__(cls, mode, text_name, limit=None):
        path = os.path.join(WORKING_PATH, text_name, f"{mode}.json")
        with open(path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        data = []
        for item in raw_data:
            image_id = item['image_id']
            image_path = os.path.join(WORKING_PATH, "dataset_image", f"{image_id}.jpg")
            if not os.path.isfile(image_path):
                continue
            if limit is not None and len(data) >= limit:
                break
            data.append({
                "id": image_id,
                "text": item['text'],
                "label": item['label'],
                "image_path": image_path
            })

        def gen():
            for item in data:
                yield item

        return tf.data.Dataset.from_generator(
            gen,
            output_signature={
                "text": tf.TensorSpec(shape=(), dtype=tf.string),
                "label": tf.TensorSpec(shape=(), dtype=tf.int32),
                "id": tf.TensorSpec(shape=(), dtype=tf.int32),
                "image_path": tf.TensorSpec(shape=(), dtype=tf.string),
            }
        )
