# Import Python Package
import pandas as pd
import numpy as np
import seaborn as sns
import os
import cv2
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import tensorflow as tf
#from tensorflow.keras import layers, models
from plotly.subplots import make_subplots
from keras.layers import GlobalAveragePooling2D, Dense
#from tensorflow.keras.models import Model
from PIL import Image
from glob import glob
import plotly.offline as pyo
from IPython.display import display
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from pathlib import Path

#   images_list = []
    # for filename in glob.glob('/Users/ryancalderon/Desktop/CSUSB_Courses/Fall_2025_Classes/CSE 5160 - Machine Learning/project1Code/images/*.jpg'):
    # im = Image.open(filename)
    # images_list.append(im)

    



def in_data():
    # --- config ---
    DATA_DIR = Path('/Users/ryancalderon/Desktop/CSUSB_Courses/Fall_2025_Classes/CSE 5160 - Machine Learning/project1Code/images')  # path to dataset
    BATCH_SIZE = 64
    VAL_SPLIT = 0.3
    SEED = 1337

    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=VAL_SPLIT,
        subset="training",
        seed=SEED,
        batch_size=BATCH_SIZE,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=VAL_SPLIT,
        subset="validation",
        seed=SEED,
        batch_size=BATCH_SIZE,
    )
    
    
def main():
    in_data()