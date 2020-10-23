
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def lstm_model(input_shape, num_classes = 1):
    inputs = keras.layers.Input(shape = input_shape)

    lstm_out = keras.layers.LSTM(60, activation="relu", return_sequences=True)(inputs)
    lstm_out = keras.layers.LSTM(30, activation="relu", return_sequences=True)(lstm_out)
    lstm_out = keras.layers.LSTM(15, activation="relu", return_sequences=True)(lstm_out)
    lstm_out = keras.layers.LSTM(7)(lstm_out)
    outputs = keras.layers.Dense(num_classes)(lstm_out)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model
