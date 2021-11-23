import matplotlib.pyplot as plt
import numpy as np


import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from batchcreator import Batchcreator

batch_size = 2
img_height = 90
img_width = 90
num_of_images = 8


def main(args=None):
    ### Read the training data
    batchcreator = Batchcreator()
    image_samples, depth_samples = batchcreator.iterate_file()

    img_height, img_width, img_depth = image_samples[0]['color'].shape
    X_train_image = []
    y_train = []
    for id, item in enumerate(image_samples):
        if id == 0:
            X_train_image = np.expand_dims(item['color'], axis=0)
        else:
            X_train_image = np.concatenate((X_train_image, np.expand_dims(item['color'], axis=0)), axis=0)
        y_train.append(item['label'][0])
    y_train = np.array(y_train)
    num_classes = 3
    batch_size = 1

    image_model = Sequential([
        layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(img_height, img_width, img_depth)),  # Input Layer: 74x74x24
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(16, activation='softmax'),
        layers.Dense(num_classes)
    ])

    image_model.compile(optimizer='adam',
                        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                        metrics=['accuracy'])

    image_model.summary()

    num_of_epochs = 50
    history = image_model.fit(x=X_train_image,
                              y=y_train,
                              epochs=num_of_epochs,
                              batch_size=batch_size,
                              validation_split=0.2)

    ###TODO Add the file name to the images. After shuffeling you can't say which image you currently predicting

    # Simple accuracy calculation
    counter = 0
    predictions = image_model.predict(X_train_image)
    for id, prediction in enumerate(predictions):
        score = tf.nn.softmax(prediction)
        result = np.argmax(score.numpy())
        if result == y_train[id]:
            counter += 1

    print("Currently {} percent of all images are predicted correctly!".format((counter / len(y_train)) * 100))
    print("Currently {} percent of all images are predicted correctly!".format((counter / len(y_train)) * 100))


if __name__ == '__main__':
    main()