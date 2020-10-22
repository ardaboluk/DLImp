
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def lstm_model(input_shape, num_classes = 2):
    inputs = keras.layers.Input(shape = input_shape)

    lstm_out = keras.layers.LSTM(32)(inputs)
    outputs = keras.layers.Dense(num_classes)(lstm_out)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model