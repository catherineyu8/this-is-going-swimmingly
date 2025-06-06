from tqdm import tqdm, trange
import tensorflow as tf
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# trains the model for some number of epochs
def train(model, train_clip, train_text):
    # train_clip is batched
    # need to batch train_text ourselves

    # learning rate is much higher here than in paper
    # paper uses AdamW (which has decoupled weight decay)
    # TODO: if this is flopping try switching to tf AdamW
    # clip_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=1e-6)
    rest_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=5e-4)

    # split trainable vars to use diff learning rates for clip and others
    clip_vars = model.clip_model.trainable_variables
    rest_vars = [var for var in model.trainable_variables if all(var is not clip_var for clip_var in clip_vars)]
        
    batch_size = 32
    num_epochs = 2 # TODO: adjust as needed
    losses = []

    for i in range(num_epochs):
        # define metric variables
        # tf vars must be defined per epoch since they accumulate data
        # using tf builtin metrics for now instead of writing custom ones
        accuracy = tf.keras.metrics.Accuracy()
        precision = tf.keras.metrics.Precision()
        recall = tf.keras.metrics.Recall()

        # batch raw text data (train_clip already batched)
        # NOTE: there are 620 batches in the training data set
        text_batched = [train_text[i:i + batch_size] for i in range(0, len(train_text), batch_size)]
        batch_counter = 0
        # iterate over batches of CLIP data and raw text
        for (batch, text_batch) in tqdm(zip(train_clip, text_batched), desc="Training"):
            # TODO: shuffle inputs, labels, and text together (make sure to take into account dictionaries)
            inputs, labels = batch

            # the very last batch is shorter so need this.
            # guess who figured that out after training the model for 619/620 batches and getting an index out of range error TvT
            this_batch_size = len(text_batch)

            shuffled_indices = tf.random.shuffle(tf.range(this_batch_size))

            inputs = {
                k: tf.gather(v, shuffled_indices, axis=0)
                for k, v in inputs.items()
            }
            labels = tf.gather(labels, shuffled_indices)
            text_batch = [text_batch[i] for i in shuffled_indices.numpy()]

            print(f"\ntraining batch: {batch_counter} of epoch {i+1}")            
        
            # do forward pass w/ gradient tape
            with tf.GradientTape() as tape:
                preds = model(inputs, text_batch)
                # print(f"labels shape: {labels.shape}, labels dtype: {labels.dtype}") # should be (32,)
                # print(f"preds shape: {preds.shape}, preds dtype: {preds.dtype}") # should be (32, 2)
                # apply sparse cross entropy loss
                loss = model.loss(labels, preds)
                losses.append(loss.numpy())
            
            print(f"completed forward pass of batch {batch_counter}")

            # get/apply gradients separately for CLIP and others and call corresponding optimizer (trainable_vars inherited))
            grads = tape.gradient(loss, model.trainable_variables)
            # clip_grads = [g for v, g in zip(model.trainable_variables, grads) if any(v is clip_var for clip_var in clip_vars)]
            rest_grads = [g for v, g in zip(model.trainable_variables, grads) if any(v is rest_var for rest_var in rest_vars)]


            # clip_optimizer.apply_gradients(zip(clip_grads, clip_vars))
            rest_optimizer.apply_gradients(zip(rest_grads, rest_vars))

            # TODO: define model accuracy function? (if this flops)
            pred_classes = tf.argmax(preds, axis=-1)
            accuracy.update_state(labels, pred_classes)
            precision.update_state(labels, pred_classes)
            recall.update_state(labels, pred_classes)

            print(f"completed metric calculation of batch {batch_counter}")
            batch_counter += 1
        
        # compute F1 score for epoch
        acc = accuracy.result().numpy()
        prec = precision.result().numpy()
        rec = recall.result().numpy()
        f1 = 2 * (prec * rec) / (prec + rec + 1e-8)
        print(f"Finished training epoch {i+1} with accuracy {acc} and F1 {f1}")

        # log data in json file for epoch
        log_data = {
            "epoch": i+1,
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1)
        }
        with open("training_log.json", "a") as f:
            f.write(json.dumps(log_data) + "\n")

        # save model's weights for each epoch (cannot save full model since it has custom classes/call method w 2 args)
        model.save_weights(f"racklemuffin_weights_epoch_{i+1}.h5")
        print(f"saved model weights to racklemuffin_weights_epoch_{i+1}.h5")

    # plot loss
    x_val_epochs = np.linspace(0, num_epochs, len(losses))

    plt.plot(x_val_epochs, losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs Epoch")
    plt.savefig("loss_vs_epoch.png")
    plt.show()

        
def test(model, test_clip, test_text):
    batch_size = 32
    text_batched = [test_text[i:i + batch_size] for i in range(0, len(test_text), batch_size)]
    
    # using tf builtin metrics for now instead of writing custom ones
    accuracy = tf.keras.metrics.Accuracy()
    precision = tf.keras.metrics.Precision()
    recall = tf.keras.metrics.Recall()

    batch_counter = 0
    all_preds = []
    all_labels = []

    for (batch, text_batch) in tqdm(zip(test_clip, text_batched), desc="Testing"):
        print(f"\ntesting batch: {batch_counter}")
        inputs, labels = batch
        preds = model(inputs, text_batch, training=False)

        # get predicted class labels and update metrics
        pred_classes = tf.argmax(preds, axis=-1)

        # Debug: Print a few predictions and labels
        # print(f"Predicted: {pred_classes.numpy()[:5]}")
        # print(f"Ground Truth: {labels.numpy()[:5]}")
        # print(f"Pred shape: {pred_classes.shape}, Label shape: {labels.shape}")

        # Optional: Manual accuracy check
        # correct = np.sum(pred_classes.numpy() == labels.numpy())
        # total = labels.shape[0]
        # print(f"Manual batch accuracy: {correct}/{total} = {correct / total:.2f}")


        accuracy.update_state(labels, pred_classes)
        precision.update_state(labels, pred_classes)
        recall.update_state(labels, pred_classes)

        all_preds.extend(pred_classes.numpy())
        all_labels.extend(labels.numpy())

        batch_counter += 1

    # compute F1 score
    acc = accuracy.result().numpy()
    prec = precision.result().numpy()
    rec = recall.result().numpy()
    f1 = 2 * (prec * rec) / (prec + rec + 1e-8)

    print(f"\nTest Accuracy: {acc}")
    print(f"Test Precision: {prec}")
    print(f"Test Recall: {rec}")
    print(f"Test F1 Score: {f1}")

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()
    plt.savefig("confusion_matrix.png")
