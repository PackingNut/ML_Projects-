# Import Python Packages
import numpy as np
import os
import tensorflow as tf
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
        
def count_examples(ds: tf.data.Dataset) -> int:
    return ds.unbatch().reduce(tf.constant(0, dtype=tf.int64), lambda acc, _: acc + 1).numpy()        

def split_memmap(
    ds: tf.data.Dataset,
    out_path_X: Path,
    out_path_y: Path,
    image_size: tuple[int, int],
    normalize: bool = False,
):
    H, W = image_size
    C = 3
    
    #count items to pre-size
    N = count_examples(ds)
    
    if normalize:
        x_dtype = np.float16
    else:
        x_dtype = np.uint8
    
    #Create on-disk .npy files and memory map them
    open_memmap = np.lib.format.open_memmap
    X_mm = open_memmap(out_path_X, mode='w+', dtype=x_dtype, shape=(N, H, W, C))
    y_mm = open_memmap(out_path_y, mode='w+', dtype=np.int64, shape=(N,))
    
    #stream through data and write
    idx = 0 
    for xb, yb in ds:
        xb = xb.numpy()
        if normalize:
            xb = (xb / 255.0).astype(np.float16)
        else:
            xb = xb.astype(np.uint8)
        
        b = xb.shape[0]
        X_mm[idx:idx+b] = xb
        y_mm[idx:idx+b] = yb.numpy().astype(np.int64)
        idx += b
    
    #Flush data
    del X_mm
    del y_mm
  

def split_data():
    # --- config ---
    DATA_DIR = Path('/Users/ryancalderon/Desktop/CSUSB_Courses/Fall_2025_Classes/CSE 5160 - Machine Learning/project1Code/images')  # path to dataset
    OUT_DIR = Path('./npy_out')
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    BATCH_SIZE = 32
    VAL_SPLIT = 0.2
    TEST_SPLIT = 0.2
    TRAIN_SPLIT = 0.6
    SEED = 1337
    assert abs(TRAIN_SPLIT + VAL_SPLIT + TEST_SPLIT - 1.0) < 1e-6
    TEMP_SPLIT = VAL_SPLIT + TEST_SPLIT 
    
    IMG_SIZE = (224, 224)
    COLOR_MODE = 'rgb'

    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=TEMP_SPLIT,
        subset="training",
        seed=SEED,
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE,
        color_mode=COLOR_MODE,
        shuffle=True,
    )

    temp_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=TEMP_SPLIT,
        subset="validation",
        seed=SEED,
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE,
        color_mode=COLOR_MODE,
        shuffle=True,
    )

    # splitting the val and test using batches
    temp_batches = tf.data.experimental.cardinality(temp_ds).numpy()
    val_fract_of_temp = VAL_SPLIT / (VAL_SPLIT + TEST_SPLIT)
    val_batches = int(temp_batches * val_fract_of_temp)

    val_ds = temp_ds.take(val_batches)
    test_ds = temp_ds.skip(val_batches)

    # Verify split
    print("Train batches:", tf.data.experimental.cardinality(train_ds).numpy())
    print("Val batches:", tf.data.experimental.cardinality(val_ds).numpy())
    print("Test batches:", tf.data.experimental.cardinality(test_ds).numpy())
    
    NORMALIZE = False
    split_memmap(train_ds, OUT_DIR/'X_train.npy', OUT_DIR/'y_train.npy', IMG_SIZE, normalize=NORMALIZE)
    split_memmap(val_ds, OUT_DIR/'X_val.npy', OUT_DIR/'y_val.npy', IMG_SIZE, normalize=NORMALIZE)
    split_memmap(test_ds, OUT_DIR/'X_test.npy', OUT_DIR/'y_test.npy', IMG_SIZE, normalize=NORMALIZE)
    
    print("Saved .npy files in: ", OUT_DIR.resolve())


def ds_to_numpy(ds: tf.data.Dataset):
    X_list, y_list = [], []
    for x, y in ds.unbatch():
        X_list.append(x.numpy()) #(H,W,C)
        y_list.append(y.numpy()) #scalar
    X = np.stack(X_list) if X_list else np.empty((0,), dtype=np.float32)
    y = np.asarray(y_list) if y_list else np.empty((0,), dtype=np.int64)
    
    return X,y
    
def main():
    split_data()
    explore(data_path='/Users/ryancalderon/Desktop/CSUSB_Courses/Fall_2025_Classes/CSE 5160 - Machine Learning/project1Code/images')

main()