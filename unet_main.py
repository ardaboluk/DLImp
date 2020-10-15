
import os

import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import PIL
from PIL import Image, ImageOps
from tensorflow.keras.preprocessing.image import load_img

import random

from oxford_dataset import OxfordPets

from unet import vanilla_unet

import matplotlib as mpl
mpl.use("TkAgg")
import matplotlib.pyplot as plt

input_dir = "images/"
target_dir = "annotations/trimaps/"
img_size = (224, 224)
num_classes = 4
batch_size = 16

input_img_paths = sorted(
    [
        os.path.join(input_dir, fname)
        for fname in os.listdir(input_dir)
        if fname.endswith(".jpg")
    ]
)

target_img_paths = sorted(
    [
        os.path.join(target_dir, fname)
        for fname in os.listdir(target_dir)
        if fname.endswith(".png") and not fname.startswith(".")
    ]
)

print("Number of samples:", len(input_img_paths))

for input_path, target_path in zip(input_img_paths[:10], target_img_paths[:10]):
    print(input_path, "|", target_path)

sample_input_img = Image.open(input_img_paths[9])
sample_target_img = ImageOps.autocontrast(Image.open(target_img_paths[9]))

plt.imshow(sample_input_img.resize(img_size))
plt.show()
plt.imshow(sample_target_img.resize(img_size))
plt.show()

# Split our img paths into a training and a validation set
# arda: additionally get test set
#val_samples = 980
#test_samples = 20
val_samples = 7000
test_samples = 20
random.Random(1337).shuffle(input_img_paths)
random.Random(1337).shuffle(target_img_paths)
train_input_img_paths = input_img_paths[:-(val_samples+test_samples)]
train_target_img_paths = target_img_paths[:-(val_samples+test_samples)]
val_input_img_paths = input_img_paths[-(val_samples+test_samples):-test_samples]
val_target_img_paths = target_img_paths[-(val_samples+test_samples):-test_samples]
test_input_img_paths = input_img_paths[-test_samples:]
test_target_img_paths = target_img_paths[-test_samples:]

# Instantiate data Sequences for each split
train_gen = OxfordPets(
    batch_size, img_size, train_input_img_paths, train_target_img_paths
)
val_gen = OxfordPets(batch_size, img_size, val_input_img_paths, val_target_img_paths)
test_gen = OxfordPets(batch_size, img_size, test_input_img_paths, test_target_img_paths)

unet_model = vanilla_unet(input_shape=(img_size[0], img_size[1], 3), num_classes = num_classes)
unet_model.summary()
keras.utils.plot_model(unet_model, show_shapes=True)

print("Compiling the model..")
unet_model.compile(optimizer="rmsprop", loss=keras.losses.SparseCategoricalCrossentropy(), metrics="sparse_categorical_accuracy", run_eagerly=False)

#checkpoint_dir = "./train_ckpt/cp-{epoch:04d}.ckpt"
#cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir, save_weights_only=True, verbose=1)

epochs_num = 1

#history = unet_model.fit(train_gen, validation_data=val_gen, epochs=epochs_num, callbacks=[cp_callback])

print("Training..")
history = unet_model.fit(train_gen, validation_data=val_gen, epochs=epochs_num)

print("Testing..")
results = unet_model.predict(test_gen)

def display_mask(pred_results_list, i):
    """Quick utility to display a model's prediction."""
    mask = np.argmax(pred_results_list[i], axis=-1)
    mask = np.expand_dims(mask, axis=-1)
    imag = PIL.ImageOps.autocontrast(keras.preprocessing.image.array_to_img(mask))
    plt.imshow(imag)
    plt.show()


# Display results for test image #10
test_img_ind = 10

# Display input image
plt.imshow(Image.open(test_input_img_paths[test_img_ind]))
plt.show()

# Display ground-truth target mask
img = PIL.ImageOps.autocontrast(load_img(test_target_img_paths[test_img_ind]))
plt.imshow(img)
plt.show()

# Display mask predicted by our model
display_mask(results, test_img_ind)