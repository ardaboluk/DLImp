
import os

import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import PIL
from PIL import Image, ImageOps

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
batch_size = 32

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

plt.imshow(sample_input_img)
plt.show()
plt.imshow(sample_target_img)
plt.show()

# Split our img paths into a training and a validation set
val_samples = 1000
random.Random(1337).shuffle(input_img_paths)
random.Random(1337).shuffle(target_img_paths)
train_input_img_paths = input_img_paths[:-val_samples]
train_target_img_paths = target_img_paths[:-val_samples]
val_input_img_paths = input_img_paths[-val_samples:]
val_target_img_paths = target_img_paths[-val_samples:]

# Instantiate data Sequences for each split
train_gen = OxfordPets(
    batch_size, img_size, train_input_img_paths, train_target_img_paths
)
val_gen = OxfordPets(batch_size, img_size, val_input_img_paths, val_target_img_paths)

unet_model = vanilla_unet(input_shape=(img_size[0], img_size[1], 3), num_classes = num_classes)
unet_model.summary()
keras.utils.plot_model(unet_model)