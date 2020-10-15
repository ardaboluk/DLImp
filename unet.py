
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

"""
Notes:
    Keras tensors are channels last by default, that is, (batch_size, height, width, num_channels).
    However, keras.Input doesn't take batch_size normally.
"""

def unet():

    contraction_conv2d_out_shapes = [[64, 64], [128, 128], [256, 256], [512, 512]]
    bottleneck_conv2d_out_shapes = [1024, 1024]
    expansion_conv2d_out_shapes = [[512, 512], [256, 256], [128, 128], [64, 64]]

    x = keras.Input(shape=(300, 300, 3))

    # contraction blocks
    contraction_feature_maps = []
    contraction_counter = 1
    conv_counter = 1
    for cur_conv_block in contraction_conv2d_out_shapes:
        for cur_conv in cur_conv_block:
            x = keras.layers.Conv2D(cur_conv, 3, padding="same", activation="relu",
                                    name=f"contraction{contraction_counter}_conv{conv_counter}")(x)
            x = keras.layers.BatchNormalization(name=f"contraction{contraction_counter}_batchnorm{conv_counter}")(x)
            contraction_feature_maps.append(x)
            conv_counter += 1
        x = keras.layers.MaxPool2D(pool_size=(2, 2))(x)
        contraction_counter += 1

    # bottleneck_block
    conv_counter = 1
    for cur_bottleneck_conv in bottleneck_conv2d_out_shapes:
        x = keras.layers.Conv2D(cur_conv, 3, padding="same", activation="relu",
                                name=f"bottleneck_conv{conv_counter}")(x)
        x = keras.layers.BatchNormalization(name=f"bottleneck_batchnorm{conv_counter}")(x)
        conv_counter += 1
    x = keras.layers.Conv2DTranspose(x.shape[-1] // 2, kernel_size=(2, 2), name=f"bottleneck_upconv")(x)

    # expansion blocks
    expansion_counter = 1
    conv_counter = 1
    for cur_conv_block in expansion_conv2d_out_shapes:
        residual = contraction_feature_maps.pop()
        tf.debugging.assert_equal(residual[0:-1], x[0:-1])
        x = keras.layers.Concatenate(name=f"expansion{expansion_counter}_concat{conv_counter}")([residual, x])
        for cur_conv in cur_conv_block:
            x = keras.layers.Conv2D(cur_conv, 3, padding="same", activation="relu",
                                    name=f"expansion{expansion_counter}_conv{conv_counter}")(x)
            x = keras.layers.BatchNormalization(name=f"expansion{expansion_counter}_batchnorm{conv_counter}")(x)
            conv_counter += 1
        x = keras.layers.Conv2DTranspose(x.shape[-1] // 2, kernel_size=(2, 2),
                                         name=f"expansion{expansion_counter}_upconv")(x)
        contraction_counter += 1

    # final layer
    