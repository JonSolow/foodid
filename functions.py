
## Some standard imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

import os
from keras import models
from keras import layers

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


from sklearn.metrics import confusion_matrix, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit

# import the necessary packages
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
#from keras import backend as K

from keras.optimizers import SGD

from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau
from keras.metrics import top_k_categorical_accuracy

import functools


#import argparse
#import random
# import pickle


np.random.seed(123)

def file_path_dataframe(path):
    """Takes in path of json that defines the training and testing datasets and returns a pandas dataframe.
    The columns of the dataframe are the class (food label) and filename of the image."""
    df = pd.read_json(path).melt(var_name='class', value_name='filename')
    df['filename'] = df['filename'] + '.jpg'
    return df


def plot_confusion_matrix(actual, predictions, class_labels=[], figsize=(16,16), x_font_size=16, y_font_size=16, title_font_size=18):
    """Plots a confusion matrix using seaborn heatmap."""
    # Calculate Confusion Matrix
    cm = confusion_matrix(actual, predictions)

    # Figure adjustment and heatmap plot
    f = plt.figure(figsize=figsize)
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax, vmax=100, cbar=False, cmap='Paired', mask=(cm==0), fmt=',.0f', linewidths=2, linecolor='grey', ); 

    # labels
    ax.set_xlabel('Predicted labels', fontsize=x_font_size);
    ax.set_ylabel('True labels', labelpad=30, fontsize=y_font_size); 
    ax.set_title('Confusion Matrix', fontsize=title_font_size); 
    if len(class_labels)>0:
        ax.xaxis.set_ticklabels(class_labels, rotation=90); 
        ax.yaxis.set_ticklabels(class_labels, rotation=0);
    ax.set_facecolor('white')