from tqdm import tqdm, trange
import tensorflow as tf


# trains the model for some number of epochs
def train(model, train_clip, train_text):
    # train_clip is batched
    # need to batch train_text ourselves

    # learning rate is much higher here than in paper
    # paper uses AdamW (which has decoupled weight decay)
    # TODO: if this is flopping try switching to tf AdamW
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)
    batch_size = 32
    num_epochs = 5

    for i in range(num_epochs):
        # define metric variables
        # tf vars must be defined per epoch since they accumulate data
        # using tf builtin metrics for now instead of writing custom ones
        accuracy = tf.keras.metrics.Accuracy()
        precision = tf.keras.metrics.Precision()
        recall = tf.keras.metrics.Recall()

        # batch raw text data (train_clip already batched)
        text_batched = [train_text[i:i + batch_size] for i in range(0, len(train_text), batch_size)]
        batch_counter = 0
        # iterate over batches of CLIP data and raw text
        for (batch, text_batch) in tqdm(zip(train_clip, text_batched), desc="Training"):
            inputs, labels = batch
            print(f"\ntraining batch: {batch_counter} of epoch {i}")

            # TODO: shuffle inputs, labels, and text together (make sure to take into account dictionaries)
        
            # do forward pass w/ gradient tape
            with tf.GradientTape() as tape:
                preds = model(inputs, text_batch)
                print(f"labels shape: {labels.shape}, labels dtype: {labels.dtype}") # should be (32,)
                print(f"preds shape: {preds.shape}, preds dtype: {preds.dtype}") # should be (32, 2)
                # apply sparse cross entropy loss
                loss = model.loss(labels, preds)
            
            print(f"completed forward pass of batch {batch_counter}")

            # get gradients and call optimizer (trainable_vars inherited))
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
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
        print(f"Finished training epoch {i} with accuracy {acc} and F1 {f1}")

        
def test(model, test_clip, test_text):
    batch_size = 32
    text_batched = [test_text[i:i + batch_size] for i in range(0, len(test_text), batch_size)]
    
    # using tf builtin metrics for now instead of writing custom ones
    accuracy = tf.keras.metrics.Accuracy()
    precision = tf.keras.metrics.Precision()
    recall = tf.keras.metrics.Recall()
    batch_counter = 0

    for (batch, text_batch) in tqdm(zip(test_clip, text_batched), desc="Testing"):
        print(f"\ntesting batch: {batch_counter}")
        inputs, labels = batch
        preds = model(inputs, batch, text_batch)

        # get predicted class labels and update metrics
        pred_classes = tf.argmax(preds, axis=-1)
        accuracy.update_state(labels, pred_classes)
        precision.update_state(labels, pred_classes)
        recall.update_state(labels, pred_classes)

    # compute F1 score
    acc = accuracy.result().numpy()
    prec = precision.result().numpy()
    rec = recall.result().numpy()
    f1 = 2 * (prec * rec) / (prec + rec + 1e-8)

    print(f"\nTest Accuracy: {acc:.4f}")
    print(f"Test Precision: {prec:.4f}")
    print(f"Test Recall: {rec:.4f}")
    print(f"Test F1 Score: {f1:.4f}")