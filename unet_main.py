
import os

import numpy as np

from tensorflow import keras

import PIL
from PIL import Image, ImageOps
from tensorflow.keras.preprocessing.image import load_img

import random

from oxford_dataset import OxfordPets

from unet import vanilla_unet

import matplotlib.pyplot as plt

input_dir = "images/"
target_dir = "annotations/trimaps/"
sample_outputs_path = "sample_output/"
img_size = (300, 300)
num_classes = 4
batch_size = 16
epochs_num = 15

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
plt.savefig(os.path.join(sample_outputs_path, "sample_input_1.jpeg"))
plt.imshow(sample_target_img.resize(img_size))
plt.savefig(os.path.join(sample_outputs_path, "sample_target_gt_1.jpeg"))

# Split our img paths into a training and a validation set
# arda: additionally get test set
val_samples = 980
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

checkpoint_dir = "./train_ckpt/cp-{epoch:04d}.ckpt"
cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir, save_weights_only=True, verbose=1)

print("Training..")
history = unet_model.fit(train_gen, validation_data=val_gen, epochs=epochs_num, callbacks=[cp_callback])

print("Testing..")
# results = unet_model.predict(test_gen)

# def getsingledatapair(input_img_path, target_img_path, target_img_size):
#     """Returns a single input-target pair, as opposed to __getitem__, which returns batch"""
#     x = load_img(input_img_path, target_size=target_img_size)
#     y = np.expand_dims(load_img(target_img_path, target_size=target_img_size, color_mode="grayscale"), 2)
#     return x, y


results = []
for i in range(test_samples):
    print("Predicting test image ", i, " ...")
    results.append(unet_model.predict(
        np.expand_dims(np.asarray(load_img(test_input_img_paths[i], target_size=img_size)), axis=0)
    ))
# for i in range(test_samples):
#     print("Predicting test image ", i, " ...")
#     results.append(unet_model.predict(
#         load_img(test_input_img_paths[i], target_size=img_size)
#     ))

pred_inputs_list = []
pred_gt_list = []
for test_ind in range(test_samples):
    pred_inputs_list.append(Image.open(test_input_img_paths[test_ind]).resize(img_size))
    pred_gt_list.append(load_img(test_target_img_paths[test_ind], target_size=img_size, color_mode="grayscale"))
    # pred_inputs_list.append(Image.open(test_input_img_paths[test_ind]))
    # pred_gt_list.append(Image.open(test_target_img_paths[test_ind]))

def save_input_gt_and_predicted_mask(pred_inputs, pred_gts, pred_results, num_preds):
    """Quick utility to display a model's prediction."""
    for pred_ind in range(num_preds):
        print("Saving results for test image ", str(pred_ind), "...")

        gt_mask = pred_gts[pred_ind]
        gt_mask = np.expand_dims(gt_mask, 2)
        gt_imag = PIL.ImageOps.autocontrast(keras.preprocessing.image.array_to_img(gt_mask))

        pred_mask = np.argmax(pred_results[pred_ind], axis=-1)
        pred_mask = np.expand_dims(pred_mask, axis=-1)
        pred_imag = PIL.ImageOps.autocontrast(keras.preprocessing.image.array_to_img(np.squeeze(pred_mask, axis=0)))

        pred_inputs[pred_ind].save(os.path.join(sample_outputs_path, str(pred_ind) + "_img.jpg"))
        gt_imag.save(os.path.join(sample_outputs_path, str(pred_ind) + "_gt.jpg"))
        pred_imag.save(os.path.join(sample_outputs_path, str(pred_ind) + "_pred.jpg"))


# Display mask predicted by our model
save_input_gt_and_predicted_mask(pred_inputs_list, pred_gt_list, results, test_samples)

