
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def vgg16(num_classes=2):

    inputs = keras.Input(shape=(224, 224, 1), name="input_1")

    x = keras.layers.Conv2D(64, 3, padding="same", activation="relu", name="conv_1")(inputs)
    x = keras.layers.Conv2D(64, 3, padding="same", activation="relu", name="conv_2")(x)
    x = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), name="maxpool_1")(x)

    x = keras.layers.Conv2D(128, 3, padding="same", activation="relu", name="conv_3")(x)
    x = keras.layers.Conv2D(128, 3, padding="same", activation="relu", name="conv_4")(x)
    x = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), name="maxpool_2")(x)

    x = keras.layers.Conv2D(256, 3, padding="same", activation="relu", name="conv_5")(x)
    x = keras.layers.Conv2D(256, 3, padding="same", activation="relu", name="conv_6")(x)
    x = keras.layers.Conv2D(256, 3, padding="same", activation="relu", name="conv_7")(x)
    x = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), name="maxpool_3")(x)

    x = keras.layers.Conv2D(512, 3, padding="same", activation="relu", name="conv_8")(x)
    x = keras.layers.Conv2D(512, 3, padding="same", activation="relu", name="conv_9")(x)
    x = keras.layers.Conv2D(512, 3, padding="same", activation="relu", name="conv_10")(x)
    x = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), name="maxpool_4")(x)

    x = keras.layers.Conv2D(512, 3, padding="same", activation="relu", name="conv_11")(x)
    x = keras.layers.Conv2D(512, 3, padding="same", activation="relu", name="conv_12")(x)
    x = keras.layers.Conv2D(512, 3, padding="same", activation="relu", name="conv_13")(x)
    x = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), name="maxpool_5")(x)

    x = keras.layers.Flatten(name="flatten_1")(x)
    x = keras.layers.Dense(4096, activation="relu", name="dense_1")(x)
    x = keras.layers.Dropout(0.2, name="dropout_1")(x)
    x = keras.layers.Dense(4096, activation="relu", name="dense_2")(x)
    x = keras.layers.Dropout(0.2, name="dropout_2")(x)

    outputs = keras.layers.Dense(num_classes, activation="softmax", name="softmax")(x)

    return keras.Model(inputs, outputs)


