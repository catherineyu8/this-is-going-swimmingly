from datasets import load_from_disk
import tensorflow as tf
from transformers import CLIPProcessor
from model import RackleMuffin
from tqdm import tqdm

def main():
    # Load the processed dataset
    dataset = load_from_disk("data/mmsd_processed")

    print("loaded data from disk")

    # convert training split to tf.data.Dataset
    train_dataset = dataset["train"].to_tf_dataset(
        columns=["pixel_values", "input_ids", "attention_mask"],
        label_cols="label",
        shuffle=True,
        batch_size=32,
        collate_fn=None  # You can define one if needed
    )

    # convert validation split to tf.data.Dataset
    val_dataset = dataset["validation"].to_tf_dataset(
        columns=["pixel_values", "input_ids", "attention_mask"],
        label_cols="label",
        shuffle=False,
        batch_size=32
    )

    # convert testing split to tf.data.dataset
    test_dataset = dataset["test"].to_tf_dataset(
        columns=["pixel_values", "input_ids", "attention_mask"],
        label_cols="label",
        shuffle=False,
        batch_size=32
    )
    
    # create processor (only used in train, but I think we should have already done this in load_data
    # processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # create model
    model = RackleMuffin()

    # pass datasets into train()
   # train(args, model, device, train_data, dev_data, test_data, processor)

    for batch in tqdm(train_dataset, desc="Training"):
        # RCLMuFN unpacking
        # text_list, image_list, label_list, id_list, samples = batch
        # inputs = processor(text=text_list.numpy(), images=image_list.numpy(), padding='max_length',
        #            truncation=True, max_length=77, return_tensors="tf")
        inputs, labels = batch

        # inputs is immediately unpacked by CLIPModel and processed, needs to be in this expected format:
        # this should be exactly what we get from training_data!
        #     inputs = {
        #         "input_ids": tf.Tensor,         # shape: (batch_size, sequence_length)
        #         "attention_mask": tf.Tensor,    # shape: (batch_size, sequence_length)
        #         "pixel_values": tf.Tensor       # shape: (batch_size, 3, 224, 224)
        #     }

        # TODO: put data in dataloader?

        model(inputs)



        # with tf.GradientTape() as tape:
        #     loss, score = model(inputs, batch=batch, labels=labels, training=True)

        # gradients = tape.gradient(loss, model.trainable_variables)
        # optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # total_loss += loss.numpy()
        # step_count += 1

        #print(f"Epoch {epoch+1} loss: {total_loss / step_count:.4f}")

    return

if __name__ == '__main__':
    main()