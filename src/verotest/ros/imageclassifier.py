import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import random
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split

from batchcreator import Batchcreator


class Imageclassifier():

    batch_size = 32
    img_height = 74
    img_width = 74

    X = []
    y = []

    # Load training dataset

    # Load labels
    batchcreator = Batchcreator()
    batch_list_col, batch_list_depth = batchcreator.iterate_file()
    length = len(batch_list_col)

    for i in batch_list_col:
        image = i['color']
        label = i['label']

        X.append(image)
        y.append(label)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    train_dataset = train_dataset.batch(batch_size)
    test_dataset = test_dataset.batch(batch_size)

    num_classes = 3

    model = Sequential([
        layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(1, img_height, img_width, 24)),
        layers.Conv3D(1, 16, 24, padding='same', activation='relu'),
        layers.MaxPooling3D(),
        layers.Conv3D(1, 32, 24, padding='same', activation='relu'),

        layers.MaxPooling3D(),
        layers.Conv3D(1, 64, 24, padding='same', activation='relu'),
        layers.MaxPooling3D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])

    model.compile(optimizer=tf.keras.optimizers.RMSprop(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.summary()

    model.fit(train_dataset, epochs=10)

    model.evaluate(test_dataset)

    '''num_classes = 3

    model = Sequential([
        layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(img_height, img_width, 24)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.summary()

    epochs = 10
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()'''












