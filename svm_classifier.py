# Import Python Packages
import numpy as np
import os
import tensorflow as tf
from pathlib import Path

#SVM imports
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import json
from PIL import Image

    
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

    # Force a deterministic class order that matches my folders
    CLASS_NAMES = [
        'DALL-E','DeepFaceLab','Face2Face','FaceShifter','FaceSwap',
        'Midjourney','NeuralTextures','Stable Diffusion','StyleGAN','Real'
    ]
    
    REAL_NAME = 'Real'
    REAL_IDX = CLASS_NAMES.index(REAL_NAME)
    
    
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=TEMP_SPLIT,
        subset="training",
        seed=SEED,
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE,
        color_mode=COLOR_MODE,
        shuffle=True,
        class_names=CLASS_NAMES,
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
        class_names=CLASS_NAMES, #mapping
    )
    
    # # Persist  mapping so predictions use the same indices
    # with open(OUT_DIR / "class_names.json", "w") as f:
    #     json.dump(CLASS_NAMES, f)
    # print("Label mapping:", dict(enumerate(CLASS_NAMES)))

    # splitting the val and test using batches
    temp_batches = tf.data.experimental.cardinality(temp_ds).numpy()
    val_fract_of_temp = VAL_SPLIT / (VAL_SPLIT + TEST_SPLIT)
    val_batches = int(temp_batches * val_fract_of_temp)

    val_ds = temp_ds.take(val_batches)
    test_ds = temp_ds.skip(val_batches)
    
    def to_binary(_, y):
        return _, tf.where(tf.equal(y, REAL_IDX), 1, 0)
    
    train_ds = train_ds.map(to_binary)
    val_ds   = val_ds.map(to_binary)
    test_ds  = test_ds.map(to_binary)

    # (Optional) save the binary label names for later reports/predictions
    with open(OUT_DIR / "class_names.json", "w") as f:
        json.dump(['fake', 'real'], f)

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

def load_class_names(out_dir: Path) -> list[str]:
    p = out_dir / "class_names.json"
    if p.exists():
        return json.load(open(p))
    return ['fake', 'real']
    
#SVM training
def train_svm(
    out_dir: Path = Path("./npy_out"),
    normalize_from_uint8: bool = True,
    C: float = 1.0,
    max_iter: int = 5000,
):
    class_names = load_class_names(out_dir)
    LABELS = [0, 1]
    
    # Load arrays from folder
    X_train = np.load(out_dir / "X_train.npy", mmap_mode="r")
    y_train = np.load(out_dir / "y_train.npy")
    X_val = np.load(out_dir / "X_val.npy", mmap_mode="r")
    y_val = np.load(out_dir / "y_val.npy")
    X_test = np.load(out_dir / "X_test.npy", mmap_mode="r")
    y_test = np.load(out_dir / "y_test.npy")
    
    #flatten & scale
    D = int(np.prod(X_train.shape[1:])) 
    def prep(X_mm):
        X = X_mm.reshape((X_mm.shape[0], D))
        return (X.astype(np.float32)/255.0) if normalize_from_uint8 else X.astype(np.float32)
    
    X_train_2d = prep(X_train)
    X_val_2d = prep(X_val)
    X_test_2d = prep(X_test)
    
    print(f"Train shape: {X_train_2d.shape}, Val: {X_val_2d.shape}, Test: {X_test_2d.shape}")
    
    #model linear SVM and scale feature variance without centering
    clf = make_pipeline(
        StandardScaler(with_mean=False),
        LinearSVC(C=C, max_iter=max_iter, dual=True, class_weight="balanced", random_state=1337)
    )
    
    #Fit to the train
    clf.fit(X_train_2d, y_train)
    
    def evaluate(X, y, split_name):
        y_pred = clf.predict(X)
        acc = accuracy_score(y, y_pred)
        print(f"\n[{split_name}] accuracy: {acc:.4f}")
        print(classification_report(y, y_pred, labels=LABELS, target_names=class_names, digits=4, zero_division=0))
        print("Confusion matrix (rows=true, cols=pred):\n", confusion_matrix(y, y_pred, labels=LABELS))
        return y_pred
    
    evaluate(X_val_2d, y_val, "VAL")
    evaluate(X_test_2d, y_test, "TEST")
    
    # Save Model
    joblib.dump(clf, out_dir  / "svm_linear.joblib")
    print("Saved model ->", (out_dir / "svm_linear.joblib").resolve())
    return clf

def eval_saved_test(
    model_path=Path("./npy_out/svm_linear.joblib"),
    out_dir=Path("./npy_out"),
    normalize_from_uint8=True,
):
    class_names = json.load(open(out_dir / "class_names.json"))
    LABELS = [0, 1]
    
    clf = joblib.load(model_path)
    X_test = np.load(out_dir / "X_test.npy", mmap_mode="r")
    y_test = np.load(out_dir / "y_test.npy")
    
    D = int(np.prod(X_test.shape[1:]))
    X = X_test.reshape((X_test.shape[0], D))
    X = (X.astype(np.float32)/255.0) if normalize_from_uint8 else X.astype(np.float32)

    y_pred = clf.predict(X)
    print(f"[TEST] accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, labels=LABELS, target_names=class_names, digits=4, zero_division=0))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred, labels=LABELS))
    

IMG_SIZE = (224, 224)
COLOR_MODE = "rgb"
NORMALIZE_FROM_UINT8 = True

def predict_image(image_path: str,
                  model_path="./npy_out/svm_linear.joblib",
                  out_dir="./npy_out"):
    import json, joblib
    class_names = json.load(open(Path(out_dir) / "class_names.json"))
    clf = joblib.load(model_path)

    img = Image.open(image_path).convert("RGB" if COLOR_MODE=="rgb" else "L")
    img = img.resize(IMG_SIZE)
    x = np.array(img).reshape(1, -1).astype(np.float32)
    if NORMALIZE_FROM_UINT8:
        x = x / 255.0

    pred_idx = int(clf.predict(x)[0])
    print(f"Prediction: {class_names[pred_idx].upper()}  (index {pred_idx})")

    # Optional: show decision margin for transparency
    if hasattr(clf, "decision_function"):
        margin = clf.decision_function(x)  # shape: (1, 2)
        print("Decision margins [fake, real]:", margin[0])



def main():
    #ensure splits exist
    split_data()
    train_svm(
        out_dir=Path("./npy_out"),
        normalize_from_uint8=True,
        C=1.0,
        max_iter=5000,
    )
    eval_saved_test()
    explore(data_path='/Users/ryancalderon/Desktop/CSUSB_Courses/Fall_2025_Classes/CSE 5160 - Machine Learning/project1Code/images')

main()