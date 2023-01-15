import os
import shutil

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization  # to create AdamW optimizer

import matplotlib.pyplot as plt

tf.get_logger().setLevel('ERROR')

AUTOTUNE = tf.data.AUTOTUNE
batch_size = 1
seed = 10

raw_train_ds = tf.keras.utils.text_dataset_from_directory(
    'TC-Financial-Report-Auto-Tagger-NLP-Ideation-Challenge/test_files',
    batch_size=batch_size,
    validation_split=0.2,
    subset='training',
    seed=seed)


class_names = raw_train_ds.class_names
train_ds = raw_train_ds.cache().prefetch(buffer_size=AUTOTUNE)


val_ds = tf.keras.utils.text_dataset_from_directory(
    'TC-Financial-Report-Auto-Tagger-NLP-Ideation-Challenge/test_files',
    batch_size=batch_size,
    validation_split=0.2,
    subset='validation',
    seed=seed)


val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

test_ds = tf.keras.utils.text_dataset_from_directory(
    'TC-Financial-Report-Auto-Tagger-NLP-Ideation-Challenge/test_files',
    batch_size=batch_size)

test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

for text_batch, label_batch in train_ds.take(1):
  for i in range(1):
    print(f'Review: {text_batch.numpy()[i]}')
    label = label_batch.numpy()[i]
    print(f'Label : {label} ({class_names[label]})')