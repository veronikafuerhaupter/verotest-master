import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

class

# Data
Data_dir = np.array(gb.glob('../Data/MELD.Raw/train_splits/*'))
Validation_dir = np.array(gb.glob('../Data/MELD.Raw/dev_splits_complete/*'))
Test_dir = np.array(gb.glob('../Data/MELD.Raw/output_repeated_splits_test/*'))

# Parameters
BATCH = 16
EMBEDDING_LENGTH = 32