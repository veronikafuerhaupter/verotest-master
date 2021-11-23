import imp
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
from PIL import Image
from batchcreator import Batchcreator

import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import concatenate

def main(args=None):
    ### Read the training data
    batchcreator = Batchcreator()
    image_samples, depth_samples = batchcreator.iterate_file()

    img_height, img_width, img_depth = image_samples[0]['color'].shape
    depth_height, depth_width, depth_depth = depth_samples[0]['depth'].shape
    X_train_image = []
    X_train_depth = []

    y_train = []
    for id, item in enumerate(image_samples):
        if id == 0:
            X_train_image = np.expand_dims(item['color'], axis=0)

        else:
            X_train_image = np.concatenate((X_train_image, np.expand_dims(item['color'], axis=0)), axis=0)

        y_train.append(item['label'])
    for id, item in enumerate(depth_samples):
        if id == 0:
            X_train_depth = np.expand_dims(item['depth'], axis=0)
        else:
            X_train_depth = np.concatenate((X_train_depth, np.expand_dims(item['depth'], axis=0)), axis=0)
    y_train = np.array(y_train)
    num_classes = 3
    batch_size = 8
    ##### Grid search #####
    result_optimizer = []
    param_optimizer = []

    # Grid Search
    for i in range(0, 17, 2):
        param_1 = 2 * i
        param_optimizer.append(param_1)
        print("Param 1: ", i * 2)

        # define two sets of inputs
        inputA = Input(shape=(img_height, img_width, img_depth))
        inputB = Input(shape=(depth_height, depth_width, depth_depth))

        # the first branch operates on the Image
        #x = Dense(param_1, activation="relu")(inputA)
        x = layers.Conv2D(param_1, 3, padding='same', activation='relu')(inputA)
        x = Dense(4, activation="relu")(x)
        x = MaxPooling2D()(x)
        x = Model(inputs=inputA, outputs=x)

        # the second branch operates on the depth
        y = Dense(128, activation="relu")(inputB)
        y = Dense(64, activation="relu")(y)
        y = Dense(32, activation="relu")(y)
        y = Dense(4, activation="relu")(y)
        y = Model(inputs=inputB, outputs=y)

        # combine the output of the two branches
        combined = concatenate([x.output, y.output])

        # apply a FC layer and then a regression prediction on the
        # combined outputs
        z = Dense(2, activation="relu")(combined)
        z = Dense(3, activation="relu")(z)
        z = layers.Flatten()(z)
        z = layers.Dense(num_classes)(z)

        # our model will accept the inputs of the two branches and
        # then output a single value
        image_model = Model(inputs=[x.input, y.input], outputs=z)

        image_model.compile(optimizer='adam',
                            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                            metrics=['accuracy'])

        image_model.summary()

        num_of_epochs = 10
        history = image_model.fit(x=[X_train_image, X_train_depth],
                                  y=y_train,
                                  epochs=num_of_epochs,
                                  batch_size=batch_size,
                                  validation_split=0.2,
                                  shuffle=True)

        ###TODO Add the file name to the images. After shuffeling you can't say which image you currently predicting

        # Simple accuracy calculation
        counter = 0
        predictions = image_model.predict([X_train_image, X_train_depth])
        for id, prediction in enumerate(predictions):
            score = tf.nn.softmax(prediction)
            result = np.argmax(score.numpy())
            if result == y_train[id]:
                counter += 1

        print("Currently {} percent of all images are predicted correctly!".format((counter / len(y_train)) * 100))
        result_optimizer.append((counter / len(y_train)) * 100)

    print("Results: ", result_optimizer)
    print("Parameters: ", param_optimizer)


if __name__ == '__main__':
    main()