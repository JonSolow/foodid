
## Some standard imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.pyplot as plt
import os
from keras import models
from keras import layers
from sklearn.metrics import confusion_matrix, f1_score

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import sys

np.random.seed(123)

def file_path_dataframe(path):
    """Takes in path of json that defines the training and testing datasets and returns a pandas dataframe.
    The columns of the dataframe are the class (food label) and filename of the image."""
    df = pd.read_json(path).melt(var_name='class', value_name='filename')
    df['filename'] = df['filename'] + '.jpg'
    return df