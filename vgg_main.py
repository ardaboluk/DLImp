
import numpy as np
import scipy.ndimage as ndi

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import matplotlib as mpl
mpl.use("TkAgg")
import matplotlib.pyplot as plt

from vgg import vgg16

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

# DEBUG
# tf.enable_eager_execution()

batch_size = 10
shuffle_buffer_size = 60000

# Using the same seed is necessary if subset specified and shuffling is performed
# Otherwise, training and validation data may overlap due to shuffling
train_dataset = keras.preprocessing.image_dataset_from_directory(
    "./datasets/mnist/trainingData",
    validation_split=0.2,
    subset="training",
    seed=1137,
    batch_size=batch_size,
    image_size=(224,224),
    color_mode="grayscale"
)

validation_dataset = keras.preprocessing.image_dataset_from_directory(
    "./datasets/mnist/trainingData",
    validation_split=0.2,
    subset="validation",
    seed=1337,
    batch_size=batch_size,
    image_size=(224,224),
    color_mode="grayscale"
)

test_dataset = keras.preprocessing.image_dataset_from_directory(
    "./datasets/mnist/testData",
    batch_size=batch_size,
    image_size=(224,224),
    color_mode="grayscale"
)

plt.figure(figsize=(10, 10))
for images, labels in train_dataset.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(int(labels[i]))
        plt.axis("off")
plt.show()

vgg16_model = vgg16(num_classes = 10)

vgg16_model.summary()

keras.utils.plot_model(vgg16_model, show_shapes=True)

vgg16_model.compile(optimizer="adam",
                    loss=keras.losses.SparseCategoricalCrossentropy(),
                    metrics="sparse_categorical_accuracy",
                    run_eagerly=False)

checkpoint_dir = "./train_ckpt/cp-{epoch:04d}.ckpt"
cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir, save_weights_only=True, verbose=1)

history = vgg16_model.fit(train_dataset, validation_data=validation_dataset, epochs=5, callbacks=[cp_callback])

results = vgg16_model.evaluate(test_dataset)
