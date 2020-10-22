
import tensorflow as tf
from tensorflow.keras import layers

def txt_cls_model(num_classes = 2, **kwargs):
    # A integer input for vocab indices.
    inputs = tf.keras.Input(shape=(None,), dtype="int64")

    max_features = kwargs["max_features"]
    embedding_dim = kwargs["embedding_dim"]

    # Next, we add a layer to map those vocab indices into a space of dimensionality
    # 'embedding_dim'.
    x = layers.Embedding(max_features, embedding_dim)(inputs)
    x = layers.Dropout(0.5)(x)

    # Conv1D + global max pooling
    x = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
    x = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
    x = layers.GlobalMaxPooling1D()(x)

    # We add a vanilla hidden layer:
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)

    # We project onto a single unit output layer, and squash it with a sigmoid:
    predictions = layers.Dense(1, activation="sigmoid", name="predictions")(x)

    model = tf.keras.Model(inputs, predictions)

    return model

