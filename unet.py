
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

"""
Notes:
    Keras tensors are channels last by default, that is, (batch_size, height, width, num_channels).
    However, keras.Input doesn't take batch_size normally.
    
"""

def contraction(shape, out_channels):
    contraction_inp = keras.Input(shape=shape, name = "contraction_input_1")
    x = keras.layers.Conv2D(out_channels, 3, padding="same", activation="relu", name="contraction_conv_1")(contraction_inp)
    x = keras.layers.BatchNormalization(name = "contraction_batchnorm_1")(x)
    x = keras.layers.Conv2D(out_channels, 3, padding="same", activation="relu", name="contraction_conv_2")(x)
    x = keras.layers.BatchNormalization(name = "contraction_batchnorm_2")(x)
    output = keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    return keras.Model(contraction_inp, output, name = "contraction")

def expansion(shape_1, shape_2, out_channels):
    tf.debugging.assert_equal(shape_1[0:2], shape_2[0:2])

    expansion_inp_1 = keras.Input(shape=shape_1, name = "expansion_input_1")
    expansion_inp_2 = keras.Input(shape=shape_2, name = "expansion_input_2")

    x = keras.layers.Concatenate(name = "expansion_concat")([expansion_inp_1, expansion_inp_2])
    x = keras.layers.Conv2D(out_channels, 3, padding="same", activation="relu", name="expansion_conv_1")(x)
    x = keras.layers.BatchNormalization(name="expansion_batchnorm_1")(x)
    x = keras.layers.Conv2D(out_channels, 3, padding="same", activation="relu", name="expansion_conv_2")(x)
    x = keras.layers.BatchNormalization(name="expansion_batchnorm_2")(x)
    output = keras.layers.Conv2DTranspose(out_channels // 2, kernel_size=(2, 2), name = "expansion_upconv")(x)

    return keras.Model([expansion_inp_1, expansion_inp_2], output)

def bottleneck(shape, out_channels):
    bottleneck_inp = keras.Input(shape=shape, name = "bottleneck_input_1")

    x = keras.layers.Conv2D(out_channels, 3, padding="same", activation="relu", name="bottleneck_conv_1")(bottleneck_inp)
    x = keras.layers.BatchNormalization(name="bottleneck_batchnorm_1")(x)
    x = keras.layers.Conv2D(out_channels, 3, padding="same", activation="relu", name="bottleneck_conv_2")(x)
    x = keras.layers.BatchNormalization(name="bottleneck_batchnorm_2")(x)
    output = keras.layers.Conv2DTranspose(out_channels // 2, kernel_size=(2, 2), name="bottleneck_upconv")(x)

    return keras.Model(bottleneck_inp, output)

def unet():

    contraction_conv2d_out_shapes = [[64, 64], [128, 128], [256, 256], [512, 512]]
    expansion_cond2d_out_shapes = [[1024, 1024], [512, 512], [256, 256], [128, 128], [64, 64]]

    x = keras.Input(shape=(300, 300, 3))

    contraction_counter = 1
    conv_counter = 1

    contraction_feature_maps = []

    for cur_conv_block in contraction_conv2d_out_shapes:
        for cur_conv in cur_conv_block:
            x = keras.layers.Conv2D(cur_conv, 3, padding="same", activation="relu", name=f"contraction{contraction_counter}_conv{conv_counter}")(x)
            x = keras.layers.BatchNormalization(name=f"block{contraction_counter}_batchnorm{conv_counter}")(x)
            conv_counter += 1
        x = keras.layers.MaxPool2D(pool_size=(2, 2))(x)
        contraction_counter += 1

    expansion_counter = 1
    conv_counter = 1

    for cur_conv_block in contraction_conv2d_out_shapes:
        for cur_conv in cur_conv_block:
            x = keras.layers.Conv2D(cur_conv, 3, padding="same", activation="relu", name=f"contraction{contraction_counter}_conv{conv_counter}")(x)
            x = keras.layers.BatchNormalization(name=f"block{contraction_counter}_batchnorm{conv_counter}")(x)
            conv_counter += 1
        x = keras.layers.MaxPool2D(pool_size=(2, 2))(x)
        contraction_counter += 1



    x = keras.layers.Conv2D(64, 3, padding="same", activation="relu", name="contraction_conv_1")(x)
    x = keras.layers.BatchNormalization(name="contraction_batchnorm_1")(x)
    x = keras.layers.Conv2D(64, 3, padding="same", activation="relu", name="contraction_conv_2")(x)
    x = keras.layers.BatchNormalization(name="contraction_batchnorm_2")(x)
    x = keras.layers.MaxPool2D(pool_size=(2, 2))(x)