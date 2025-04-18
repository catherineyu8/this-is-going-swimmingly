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
    # TODO: trying to get number of items in training data...this could be wrong
    num_batches = len(train_clip[0]["input_ids"]) // batch_size
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
        # iterate over batches of CLIP data and raw text
        for (batch, text_batch) in tqdm(zip(train_clip, text_batched), desc="Training"):
            inputs, labels = batch

            # TODO: shuffle inputs, labels, and text together (make sure to take into account dictionaries)
            
            # do forward pass w/ gradient tape
            with tf.GradientTape() as tape:
                preds = model(inputs, batch, text_batch)

                # did this in prev assignments but idk if necessary here
                labels = tf.cast(labels, tf.float32)
                # TODO: define model loss function
                loss = model.loss(preds, labels)

            # get gradients and call optimizer (trainable_vars inherited))
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            # assumes binary classiciation with 0.5 threshold (?)
            # should verify this is true
            predicted_labels = tf.round(preds)
            # TODO: define model accuracy function? (if this flops)
            accuracy.update_state(labels, predicted_labels)
            precision.update_state(labels, predicted_labels)
            recall.update_state(labels, predicted_labels)
        
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

    for (batch, text_batch) in tqdm(zip(test_clip, text_batched), desc="Testing"):
        inputs, labels = batch
        preds = model(inputs, batch, text_batch)

        # round probabilities from softmax to classifications
        # TODO: may not need based on model implementation
        pred_classes = tf.round(preds)

        # cast labels if necessary and also pred labels to match
        labels = tf.cast(labels, tf.int32)
        predicted_labels = tf.cast(pred_classes, tf.int32)

        # Update metrics
        accuracy.update_state(labels, predicted_labels)
        precision.update_state(labels, predicted_labels)
        recall.update_state(labels, predicted_labels)

    # compute F1 score
    acc = accuracy.result().numpy()
    prec = precision.result().numpy()
    rec = recall.result().numpy()
    f1 = 2 * (prec * rec) / (prec + rec + 1e-8)

    print(f"\nTest Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1 Score: {f1:.4f}")