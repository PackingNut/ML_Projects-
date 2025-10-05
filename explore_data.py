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
    
def explore(data_path):
    classes = os.listdir(data_path)
    for class_name in classes:
        if class_name.startswith('.'):
            continue
        
        class_path = os.path.join(data_path, class_name)
        if not os.path.isdir(class_path):
            continue
        
        count = len([
            f for f in os.listdir(class_path)
            if not f.startswith('.')
        ])
        
        print(f"{class_name}: {count} images")

def split_data():
    # --- config ---
    DATA_DIR = Path('/Users/ryancalderon/Desktop/CSUSB_Courses/Fall_2025_Classes/CSE 5160 - Machine Learning/project1Code/images')  # path to dataset
    BATCH_SIZE = 64
    VAL_SPLIT = 0.2
    TEST_SPLIT = 0.2
    TRAIN_SPLIT = 0.6
    SEED = 1337
    assert abs(TRAIN_SPLIT + VAL_SPLIT + TEST_SPLIT - 1.0) < 1e-6
    TEMP_SPLIT = VAL_SPLIT + TEST_SPLIT 

    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=TEMP_SPLIT,
        subset="training",
        seed=SEED,
        batch_size=BATCH_SIZE,
    )

    temp_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=TEMP_SPLIT,
        subset="validation",
        seed=SEED,
        batch_size=BATCH_SIZE,
    )

    # splitting the val and test using batches
    temp_batches = tf.data.experimental.cardinality(temp_ds).numpy()
    val_fract_of_temp = VAL_SPLIT / (VAL_SPLIT + TEST_SPLIT)
    val_batches = int(temp_batches * val_fract_of_temp)

    val_ds = temp_ds.take(val_batches)
    test_ds = temp_ds.skip(val_batches)

    # tweaks to increase performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)
    test_ds = test_ds.cache().prefetch(AUTOTUNE)

    # Verify split
    print("Train batches:", tf.data.experimental.cardinality(train_ds).numpy())
    print("Val batches:", tf.data.experimental.cardinality(val_ds).numpy())
    print("Test batches:", tf.data.experimental.cardinality(test_ds).numpy())
    

    
    
def main():
    split_data()
    explore(data_path='/Users/ryancalderon/Desktop/CSUSB_Courses/Fall_2025_Classes/CSE 5160 - Machine Learning/project1Code/images')

main()