
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

from themodel import txt_cls_model

import sys
import os

import string
import re

batch_size = 32
raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
    "../datasets/lstm_datasets/aclImdb/train",
    batch_size=batch_size,
    validation_split=0.2,
    subset="training",
    seed=1337,
)
raw_val_ds = tf.keras.preprocessing.text_dataset_from_directory(
    "../datasets/lstm_datasets/aclImdb/train",
    batch_size=batch_size,
    validation_split=0.2,
    subset="validation",
    seed=1337,
)
raw_test_ds = tf.keras.preprocessing.text_dataset_from_directory(
    "../datasets/lstm_datasets/aclImdb/test", batch_size=batch_size
)

print(
    "Number of batches in raw_train_ds: %d"
    % tf.data.experimental.cardinality(raw_train_ds)
)
print(
    "Number of batches in raw_val_ds: %d" % tf.data.experimental.cardinality(raw_val_ds)
)
print(
    "Number of batches in raw_test_ds: %d"
    % tf.data.experimental.cardinality(raw_test_ds)
)

# It's important to take a look at your raw data to ensure your normalization
# and tokenization will work as expected. We can do that by taking a few
# examples from the training set and looking at them.
# This is one of the places where eager execution shines:
# we can just evaluate these tensors using .numpy()
# instead of needing to evaluate them in a Session/Graph context.
for text_batch, label_batch in raw_train_ds.take(1):
    for i in range(5):
        print(text_batch.numpy()[i])
        print(label_batch.numpy()[i])

# Having looked at our data above, we see that the raw text contains HTML break
# tags of the form '<br />'. These tags will not be removed by the default
# standardizer (which doesn't strip HTML). Because of this, we will need to
# create a custom standardization function.
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ")
    return tf.strings.regex_replace(
        stripped_html, "[%s]" % re.escape(string.punctuation), ""
    )


# Model constants.
max_features = 20000
embedding_dim = 128
sequence_length = 500

# Now that we have our custom standardization, we can instantiate our text
# vectorization layer. We are using this layer to normalize, split, and map
# strings to integers, so we set our 'output_mode' to 'int'.
# Note that we're using the default split function,
# and the custom standardization defined above.
# We also set an explicit maximum sequence length, since the CNNs later in our
# model won't support ragged sequences.
vectorize_layer = TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode="int",
    output_sequence_length=sequence_length,
)

# Now that the vocab layer has been created, call `adapt` on a text-only
# dataset to create the vocabulary. You don't have to batch, but for very large
# datasets this means you're not keeping spare copies of the dataset in memory.

# Let's make a text-only dataset (no labels):
text_ds = raw_train_ds.map(lambda x, y: x)
# Let's call `adapt`:
vectorize_layer.adapt(text_ds)

# There are 2 ways we can use our text vectorization layer:
# Option 1: Make it part of the model, so as to obtain a model that processes raw strings, like this:
# text_input = tf.keras.Input(shape=(1,), dtype=tf.string, name='text')
# x = vectorize_layer(text_input)
# x = layers.Embedding(max_features + 1, embedding_dim)(x)
# ...

# Option 2: Apply it to the text dataset to obtain a dataset of word indices, then feed it into a
# model that expects integer sequences as inputs. An important difference between the two is that
# option 2 enables you to do asynchronous CPU processing and buffering of your data when training
# on GPU. So if you're training the model on GPU, you probably want to go with this option to get
# the best performance. This is what we will do below.
#
# If we were to export our model to production, we'd ship a model that accepts raw strings as input,
# like in the code snippet for option 1 above. This can be done after training. We do this in the last section.

def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label

# Vectorize the data.
train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)

# Do async prefetching / buffering of the data for best performance on GPU.
# Caching loads whole dataset to host memory first, then loads batches to device memory
# as needed. Prefetching allows preparing of next data while processing the current one.
# Buffer size in prefetch method is in batches unit.
train_ds = train_ds.cache().prefetch(buffer_size=10)
val_ds = val_ds.cache().prefetch(buffer_size=10)
test_ds = test_ds.cache().prefetch(buffer_size=10)

# create the model
model = txt_cls_model(max_features = max_features, embedding_dim = embedding_dim)

callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath="./train_ckpt/cp-{epoch:04d}.ckpt", save_weights_only=True, verbose=1)]

# Compile the model with binary crossentropy loss and an adam optimizer.
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

epochs = 50
# Fit the model using the train and test datasets.
model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)

print()
print("Evaluating the model on the test set..")
model.evaluate(test_ds)

# Textvectorization layer can also be added to the model
