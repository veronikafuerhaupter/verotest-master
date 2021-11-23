import random

import os
import re
import numpy as np
from PIL import Image
from gaussiannoisecreator import Gaussiannoisecreator

class Batchcreator:

    batch_list = None
    counter = 0

    def __init__(self):
        self.batch_list_col = []
        self.batch_list_depth = []
        self.directory_list = []

    def iterate_file(self):
        DIR_PATH = os.path.dirname(os.path.realpath(__file__))
        self.directory_list.append('training_material0')
        self.directory_list.append('training_material90')
        self.directory_list.append('training_material180')
        zähler = 0

        for runner, i in enumerate(self.directory_list):
            directory_name = self.directory_list[runner]
            DIR_PATH_TRAINING = os.path.join(DIR_PATH, directory_name+str('/'))

            for f in os.listdir(DIR_PATH_TRAINING):
                zähler = zähler + 1
                if f == 'observation60\\springmittel_depth_cropped1.npy':
                    continue
                DIR_PATH_OBSERVATIONS = os.path.join(DIR_PATH_TRAINING, f + str('/'))
                onlyfiles_col = [f for f in os.listdir(DIR_PATH_OBSERVATIONS) if "springmittel_col_cropped" in f]
                onlyfiles_depth = [f for f in os.listdir(DIR_PATH_OBSERVATIONS) if "springmittel_depth_cropped" in f]

                if len(onlyfiles_col) == 0:
                    continue

                onlyfiles_col.sort()
                onlyfiles_depth.sort()

                image_all_col = 0
                image_all_col_flipud = 0
                image_all_col_fliplr = 0
                image_all_depth = 0
                image_all_depth_flipud = 0
                image_all_depth_fliplr = 0

                if len(onlyfiles_col) == 8 and len(onlyfiles_depth) == 8:

                    for counter, filecol in enumerate(onlyfiles_col):
                        if counter == 0:

                            # Load 1st Color numpy array from directory and resize it
                            image_all_col = np.resize(np.load(os.path.join(DIR_PATH_OBSERVATIONS, filecol)), (74, 74, 3))

                            # Apply gaussian noise to images
                            gaussiannoisecreator = Gaussiannoisecreator()
                            image_all_col_noise = gaussiannoisecreator.gaussian_noise(image_all_col)

                            # Data augmentation via flipping the arrays horizontally and vertically
                            image_all_col_flipud = np.flipud(image_all_col)
                            image_all_col_fliplr = np.fliplr(image_all_col)

                        else:
                            # Load and 2nd to 8th Color numpy array from directory and resize it
                            image_col = np.resize(np.load(os.path.join(DIR_PATH_OBSERVATIONS, filecol)), (74, 74, 3))

                            # Data augmentation via flipping the arrays horizontally and vertically
                            image_col_noise = gaussiannoisecreator.gaussian_noise(image_col)
                            image_col_flipud = np.flipud(image_col)
                            image_col_fliplr = np.fliplr(image_col)

                            # Concatenate the sequence of 8 Springmittel
                            image_all_col = np.concatenate((image_all_col, image_col), axis=2)
                            image_all_col_flipud = np.concatenate((image_all_col_flipud, image_col_flipud), axis=2)
                            image_all_col_fliplr = np.concatenate((image_all_col_fliplr, image_col_fliplr), axis=2)
                            image_all_col_noise = np.concatenate((image_all_col_noise, image_col_noise), axis=2)


                    for counter, filedepth in enumerate(onlyfiles_depth):

                        if counter == 0:

                            # Load depth numpy array from directory, resize and exoand dimensions
                            image_all_depth = np.expand_dims(np.resize(np.load(os.path.join(DIR_PATH_OBSERVATIONS, filedepth)), (37, 37)), axis=2)

                            # Data augmentation via flipping the arrays horizontally and vertically
                            image_all_depth_flipud = np.flipud(image_all_depth)
                            image_all_depth_fliplr = np.fliplr(image_all_depth)
                            image_all_depth_noise = gaussiannoisecreator.gaussian_noise(image_all_depth)

                        else:
                            image_depth = np.expand_dims(np.resize(np.load(os.path.join(DIR_PATH_OBSERVATIONS, filedepth)), (37, 37)), axis=2)

                            # Data augmentation via flipping the arrays horizontally and vertically
                            image_depth_flipud = np.flipud(image_depth)
                            image_depth_fliplr = np.fliplr(image_depth)
                            image_depth_noise = gaussiannoisecreator.gaussian_noise(image_depth)

                            # Concatenate the sequence of 8 Springmittel
                            image_all_depth = np.concatenate((image_all_depth, image_depth), axis=2)
                            image_all_depth_flipud = np.concatenate((image_all_depth_flipud, image_depth_flipud), axis=2)
                            image_all_depth_fliplr = np.concatenate((image_all_depth_fliplr, image_depth_fliplr), axis=2)
                            image_all_depth_noise = np.concatenate((image_all_depth_noise, image_depth_noise),axis=2)

                    # Append the image sequences to lists
                    self.batch_list_col.append({'color': image_all_col, 'label': runner})
                    self.batch_list_col.append({'color': image_all_col_flipud, 'label': runner})
                    self.batch_list_col.append({'color': image_all_col_fliplr, 'label': runner})
                    self.batch_list_col.append({'color': image_all_col_noise, 'label': runner})
                    self.batch_list_depth.append({'depth': image_all_depth, 'label': runner})
                    self.batch_list_depth.append({'depth': image_all_depth_flipud, 'label': runner})
                    self.batch_list_depth.append({'depth': image_all_depth_fliplr, 'label': runner})
                    self.batch_list_depth.append({'depth': image_all_depth_noise, 'label': runner})

        random.shuffle(self.batch_list_col)
        random.shuffle(self.batch_list_depth)

        return self.batch_list_col, self.batch_list_depth

batchcreator = Batchcreator()
batchcreator.iterate_file()