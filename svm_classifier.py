# Import Python Packages
import numpy as np
import os
import tensorflow as tf
from pathlib import Path

#SVM imports
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
    
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
    
# get class names from directory; same way as keras    
def get_class_names(data_dir: Path):
    return sorted([d.name for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith('.')])

#SVM training
def train_svm(
    out_dir: Path = Path("./npy_out"),
    data_dir: Path = Path('/Users/ryancalderon/Desktop/CSUSB_Courses/Fall_2025_Classes/CSE 5160 - Machine Learning/project1Code/images'),
    normalize_from_uint8: bool = True,
    C: float = 1.0,
    max_iter: int = 5000,
):
    out_dir = Path(out_dir)
    
    # Load arrays from folder
    X_train = np.load(out_dir / "X_train.npy", mmap_mode="r")
    y_train = np.load(out_dir / "y_train.npy")
    X_val = np.load(out_dir / "X_val.npy", mmap_mode="r")
    y_val = np.load(out_dir / "y_val.npy")
    X_test = np.load(out_dir / "X_test.npy", mmap_mode="r")
    y_test = np.load(out_dir / "y_test.npy")
    
    #flatten
    n_train = X_train.shape[0]
    n_val = X_val.shape[0]
    n_test = X_test.shape[0]
    D = np.prod(X_train.shape[1:], dtype=int)
    
    def prep(X_mm):
        X = X_mm.reshape((X_mm.shape[0], D))
        if normalize_from_uint8:
            X = X.astype(np.float32) / 255.0
        else:
            X = X.astype(np.float32)
        return X
    X_train_2d = prep(X_train)
    X_val_2d = prep(X_val)
    X_test_2d = prep(X_test)
    
    print(f"Train shape: {X_train_2d.shape}, Val: {X_val_2d.shape}, Test: {X_test_2d.shape}")
    
    #model linear SVM and scale feature variance without centering
    clf = make_pipeline(
        StandardScaler(with_mean=False),
        LinearSVC(C=C, max_iter=max_iter, dual=True)
    )
    
    #Fit to the train
    clf.fit(X_train_2d, y_train)

    def evaluate(X, y, split_name):
        y_pred = clf.predict(X)
        acc = accuracy_score(y, y_pred)
        print(f"\n[{split_name}] accuracy: {acc: .4f}")
        print(classification_report(y, y_pred, target_names=get_class_names(data_dir)))
        return y_pred
    
    _ = evaluate(X_val_2d, y_val, "VAL")
    _ = evaluate(X_test_2d, y_test, "TEST")
    
    model_path = out_dir / "svm_linear.joblib"
    joblib.dump(clf, model_path)
    print(f"\nSaved model -> {model_path.resolve()}")
    
    return clf


def main():
    #ensure splits exist
    train_ds, val_ds, test_ds = split_data()
    train_svm(
        out_dir=Path("./.npy_out"),
        data_dir=Path('/Users/ryancalderon/Desktop/CSUSB_Courses/Fall_2025_Classes/CSE 5160 - Machine Learning/project1Code/images'),
        normalize_from_uint8=True,
        C=1.0,
        max_iter=5000,
    )
    explore(data_path='/Users/ryancalderon/Desktop/CSUSB_Courses/Fall_2025_Classes/CSE 5160 - Machine Learning/project1Code/images')

main()