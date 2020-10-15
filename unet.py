
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

"""
Notes:
    Keras tensors are channels last by default, that is, (batch_size, height, width, num_channels).
    However, keras.Input doesn't take batch_size normally.
"""

def unet_with_resize(input_shape=(300, 300, 3), num_classes = 2):

    contraction_conv2d_out_shapes = [[64, 64], [128, 128], [256, 256], [512, 512]]
    bottleneck_conv2d_out_shapes = [1024, 1024]
    expansion_conv2d_out_shapes = [[512, 512], [256, 256], [128, 128]]
    resize_coeff = 5

    unet_input = keras.Input(shape=input_shape)

    x = unet_input

    # contraction blocks
    contraction_feature_maps = []
    contraction_counter = 1
    conv_counter = 1
    for cur_conv_block in contraction_conv2d_out_shapes:
        for cur_conv in cur_conv_block:
            x = keras.layers.Conv2D(cur_conv, 3, padding="valid", activation="relu",
                                    name=f"contraction{contraction_counter}_conv{conv_counter}")(x)
            x = keras.layers.BatchNormalization(name=f"contraction{contraction_counter}_batchnorm{conv_counter}")(x)
            conv_counter += 1
        contraction_feature_maps.append(x)
        x = keras.layers.MaxPool2D(pool_size=(2, 2))(x)
        contraction_counter += 1

    # bottleneck_block
    conv_counter = 1
    for cur_bottleneck_conv in bottleneck_conv2d_out_shapes:
        x = keras.layers.Conv2D(cur_bottleneck_conv, 3, padding="valid", activation="relu",
                                name=f"bottleneck_conv{conv_counter}")(x)
        x = keras.layers.BatchNormalization(name=f"bottleneck_batchnorm{conv_counter}")(x)
        conv_counter += 1
    x = keras.layers.UpSampling2D(size=(2, 2), name = "bottleneck_upsampling")(x)
    x = keras.layers.Conv2D(x.shape[-1] // 2, padding="same", kernel_size=(2, 2), name = "bottleneck_upconv")(x)

    # expansion blocks
    expansion_counter = 1
    conv_counter = 1
    for cur_conv_block in expansion_conv2d_out_shapes:
        residual = contraction_feature_maps.pop()
        height_diff = residual.shape[1] - x.shape[1]
        width_diff = residual.shape[1] - x.shape[1]

        # DEBUG
        #print("width_diff ", width_diff)

        tf.debugging.assert_greater(height_diff, 0)
        tf.debugging.assert_greater(width_diff, 0)

        residual = keras.layers.Cropping2D(cropping=((int(np.ceil(height_diff / 2)), int(np.floor(height_diff / 2))),
                                                     (int(np.ceil(width_diff / 2)), int(np.floor(width_diff / 2)))),
                                           name=f"expansion{expansion_counter}_crop")(residual)

        x = keras.layers.Concatenate(name=f"expansion{expansion_counter}_concat")([residual, x])
        for cur_conv in cur_conv_block:
            x = keras.layers.Conv2D(cur_conv, 3, padding="valid", activation="relu",
                                    name=f"expansion{expansion_counter}_conv{conv_counter}")(x)
            x = keras.layers.BatchNormalization(name=f"expansion{expansion_counter}_batchnorm{conv_counter}")(x)
            conv_counter += 1

        if expansion_counter != len(expansion_conv2d_out_shapes):
            x = keras.layers.UpSampling2D(size=(2, 2), name=f"expansion{expansion_counter}_upsampling")(x)
            x = keras.layers.Conv2D(x.shape[-1] // 2, kernel_size=(2, 2), padding="same", name=f"expansion{expansion_counter}_upconv")(x)

        expansion_counter += 1

    # resize layer
    x = keras.layers.experimental.preprocessing.Resizing(x.shape[1] * resize_coeff, x.shape[2] * resize_coeff, name="resize")(x)

    # final layer
    output = keras.layers.Conv2D(num_classes, kernel_size=(1, 1), strides = (1, 1), padding="valid", name="final_layer")(x)

    return keras.Model(unet_input, output)

def vanilla_unet(input_shape = (300, 300, 3), num_classes = 2):

    contraction_conv2d_out_shapes = [[64, 64], [128, 128], [256, 256], [512, 512]]
    bottleneck_conv2d_out_shapes = [1024, 1024]
    expansion_conv2d_out_shapes = [[512, 512], [256, 256], [128, 128], [64, 64]]
    resize_coeff = 5

    unet_input = keras.Input(shape=input_shape)

    x = unet_input

    # contraction blocks
    contraction_feature_maps = []
    contraction_counter = 1
    conv_counter = 1
    for cur_conv_block in contraction_conv2d_out_shapes:
        for cur_conv in cur_conv_block:
            x = keras.layers.Conv2D(cur_conv, 3, padding="same", activation="relu",
                                    name=f"contraction{contraction_counter}_conv{conv_counter}")(x)
            x = keras.layers.BatchNormalization(name=f"contraction{contraction_counter}_batchnorm{conv_counter}")(x)
            conv_counter += 1
        contraction_feature_maps.append(x)
        x = keras.layers.MaxPool2D(pool_size=(2, 2))(x)
        contraction_counter += 1

    # bottleneck_block
    conv_counter = 1
    for cur_bottleneck_conv in bottleneck_conv2d_out_shapes:
        x = keras.layers.Conv2D(cur_bottleneck_conv, 3, padding="same", activation="relu",
                                name=f"bottleneck_conv{conv_counter}")(x)
        x = keras.layers.BatchNormalization(name=f"bottleneck_batchnorm{conv_counter}")(x)
        conv_counter += 1
    x = keras.layers.UpSampling2D(size=(2, 2), name = "bottleneck_upsampling")(x)
    x = keras.layers.Conv2D(x.shape[-1] // 2, padding="same", kernel_size=(2, 2), name = "bottleneck_upconv")(x)

    # expansion blocks
    expansion_counter = 1
    conv_counter = 1
    for cur_conv_block in expansion_conv2d_out_shapes:
        residual = contraction_feature_maps.pop()
        x = keras.layers.Concatenate(name=f"expansion{expansion_counter}_concat")([residual, x])
        for cur_conv in cur_conv_block:
            x = keras.layers.Conv2D(cur_conv, 3, padding="same", activation="relu",
                                    name=f"expansion{expansion_counter}_conv{conv_counter}")(x)
            x = keras.layers.BatchNormalization(name=f"expansion{expansion_counter}_batchnorm{conv_counter}")(x)
            conv_counter += 1

        if expansion_counter != len(expansion_conv2d_out_shapes):
            x = keras.layers.UpSampling2D(size=(2, 2), name=f"expansion{expansion_counter}_upsampling")(x)
            x = keras.layers.Conv2D(x.shape[-1] // 2, kernel_size=(2, 2), padding="same", name=f"expansion{expansion_counter}_upconv")(x)

        expansion_counter += 1

    # final layer
    output = keras.layers.Conv2D(num_classes, kernel_size=(3, 3), strides = (1, 1), activation="softmax", padding="same", name="final_layer")(x)

    # channel-wise softmax layer for determining the class of each output pixel
    # output = keras.layers.Softmax(axis = -1)(final_layer)

    return keras.Model(unet_input, output)
