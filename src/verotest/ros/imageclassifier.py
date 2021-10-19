import gensim.models as gm
import glob as gb
import keras.applications as ka
import keras.layers as kl
import keras.models as km
import keras.optimizers as ko
import keras_preprocessing.image as ki
import keras_preprocessing.sequence as ks
import keras_preprocessing.text as kt
import numpy as np
import pandas as pd
import pickle as pk
import tensorflow as tf
import utils as ut


# Data
Data_dir = np.array(gb.glob('../Data/MELD.Raw/train_splits/*'))
Validation_dir = np.array(gb.glob('../Data/MELD.Raw/dev_splits_complete/*'))
Test_dir = np.array(gb.glob('../Data/MELD.Raw/output_repeated_splits_test/*'))

# Parameters
BATCH = 16
EMBEDDING_LENGTH = 32