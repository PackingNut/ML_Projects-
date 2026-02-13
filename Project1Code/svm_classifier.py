# Import Python Packages
import numpy as np
import os
import tensorflow as tf
from pathlib import Path

#SVM + Plot imports
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
#SVM imports
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import joblib
import json
from PIL import Image
from tempfile import NamedTemporaryFile
from matplotlib import pyplot as plt

    
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
              

def split_memmap(
    ds,
    out_path_X: Path,
    out_path_y: Path,
    image_size,
    normalize: bool = False,
):
    H, W = image_size
    C = 3
    
    def num_examples(d):
        total = 0
        for xb, _ in d:
            total += int(xb.shape[0])
        return total
    #count items to pre-size
    N = num_examples(ds)
    
    if normalize:
        x_dtype = np.float16
    else:
        x_dtype = np.uint8
    
    # Write to temp files, then atomically move into place on success.

    with NamedTemporaryFile(delete=False, dir=str(out_path_X.parent), suffix=".tmp") as tx,\
         NamedTemporaryFile(delete=False, dir=str(out_path_y.parent), suffix=".tmp") as ty:
        tx_name, ty_name = tx.name, ty.name

    # Create memmaps using the temp paths (use str() for safety)
    X_mm = np.lib.format.open_memmap(str(tx_name), mode="w+", dtype=x_dtype, shape=(N, H, W, C))
    Y_mm = np.lib.format.open_memmap(str(ty_name), mode="w+", dtype=np.int64,  shape=(N,))

    idx = 0
    for xb, yb in ds:
        xb = xb.numpy()
        xb = (xb / 255.0).astype(np.float16) if normalize else xb.astype(np.uint8)
        b = xb.shape[0]
        X_mm[idx:idx+b] = xb
        Y_mm[idx:idx+b] = yb.numpy().astype(np.int64)
        idx += b

    # Ensure buffers are flushed and closed before rename
    del X_mm
    del Y_mm

    os.replace(tx_name, out_path_X)
    os.replace(ty_name, out_path_y)
  

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

def eval_saved_test(
    model_path=Path("./npy_out/svm_linear_streaming.joblib"),
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
<<<<<<< HEAD
    
    print(f"[TEST] accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, labels=LABELS, target_names=class_names, digits=4, zero_division=0))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred, labels=LABELS))
    # after computing y_pred
    y_score = clf.decision_function(X)  # shape (N,)
    cm = confusion_matrix(y_test, y_pred, labels=LABELS)

    plot_conf_mat(cm, class_names, Path(out_dir) / "test_cm.png")
    plot_roc_pr(y_test, y_score, str(Path(out_dir) / "test"))
    plot_score_regression(y_test, y_score, Path(out_dir) / "test_score_reg.png")
    print(f"Saved plots under {Path(out_dir).resolve()}")

=======
    print(f"[TEST] accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, labels=LABELS, target_names=class_names, digits=4, zero_division=0))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred, labels=LABELS))
>>>>>>> 67853e16185c53edbd3d9580bd5864ee25bb7cf6
    

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

<<<<<<< HEAD
def plot_conf_mat(cm, class_names, out_path):
    plt.figure(figsize=(4, 3))
    plt.imshow(cm, interpolation='nearest')
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)
    # annotate
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def plot_roc_pr(y_true, y_score, out_prefix):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    prec, rec, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    # ROC
    plt.figure(figsize=(4,3))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0,1], [0,1], linestyle='--')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_roc.png", dpi=140)
    plt.close()

    # PR
    plt.figure(figsize=(4,3))
    plt.plot(rec, prec, label=f"AP = {ap:.3f}")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision–Recall')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_pr.png", dpi=140)
    plt.close()


def plot_score_regression(y_true, y_score, out_path):
    """
    Simple 'regression line' visualization:
    fit y (0/1) ~ a * score + b and overlay on scatter of scores vs y.
    """
    scores = np.asarray(y_score).ravel()
    y = np.asarray(y_true).astype(float).ravel()

    # best-fit line y = a*x + b
    a, b = np.polyfit(scores, y, 1)
    xline = np.linspace(scores.min(), scores.max(), 200)
    yline = a * xline + b

    plt.figure(figsize=(5,3))
    # jitter y for visibility
    jitter = (np.random.RandomState(0).randn(len(y)) * 0.015)
    plt.scatter(scores, y + jitter, s=6, alpha=0.35, label='samples')
    plt.plot(xline, yline, linewidth=2, label=f'y≈{a:.3f}·score+{b:.3f}')
    plt.axhline(0.5, linestyle='--', linewidth=1, alpha=0.6, label='y=0.5')
    plt.xlabel('Decision score (margin)')
    plt.ylabel('y (0=fake, 1=real)')
    plt.title('Score vs Label with Regression Line')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()

=======
# def validate_npy(out_dir=Path("./npy_out")):
#     names = ["X_train.npy","y_train.npy","X_val.npy","y_val.npy","X_test.npy","y_test.npy"]
#     ok = True
#     for n in names:
#         p = out_dir / n
#         print("->", p)
#         if not p.exists():
#             print("   MISSING")
#             ok = False
#             continue
#         with open(p, "rb") as f:
#             magic = f.read(6)
#             print("   magic:", magic)
#             if magic != b"\x93NUMPY":
#                 print("   NOT A NPY FILE (bad magic header)")
#                 ok = False
#         try:
#             arr = np.load(p, mmap_mode=None)  # non-mmap to test the header
#             print("   shape:", arr.shape, "dtype:", arr.dtype)
#         except Exception as e:
#             print("   FAILED to load:", e)
#             ok = False
#     return ok
>>>>>>> 67853e16185c53edbd3d9580bd5864ee25bb7cf6

#SVM Training
def train_linear_svm_streaming(out_dir=Path("./npy_out"),
                               normalize_from_uint8=True,
                               batch_size=512, epochs=2, alpha=1e-4, rng_seed=1337):
    # load
    X_train = np.load(out_dir/"X_train.npy", mmap_mode="r")
    y_train = np.load(out_dir/"y_train.npy")
    X_val   = np.load(out_dir/"X_val.npy",   mmap_mode="r")
    y_val   = np.load(out_dir/"y_val.npy")
    X_test  = np.load(out_dir/"X_test.npy",  mmap_mode="r")
    y_test  = np.load(out_dir/"y_test.npy")

    D = int(np.prod(X_train.shape[1:]))
    classes = np.array([0, 1])

    # --- compute balanced class weights and pass as dict ---
    w = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight = {int(c): float(wi) for c, wi in zip(classes, w)}

    clf = SGDClassifier(loss="hinge",
                        alpha=alpha,
                        class_weight=class_weight,   # <-- dict, works with partial_fit
                        average=True,
                        random_state=rng_seed,
                        max_iter=1, tol=None)

    rng = np.random.default_rng(rng_seed)
    N = X_train.shape[0]

    for e in range(epochs):
        order = rng.permutation(N)
        for s in range(0, N, batch_size):
            idx = order[s:s+batch_size]
            Xb = X_train[idx].reshape(-1, D).astype(np.float32)
            if normalize_from_uint8: Xb /= 255.0
            yb = y_train[idx]
            if e == 0 and s == 0:
                clf.partial_fit(Xb, yb, classes=classes)
            else:
                clf.partial_fit(Xb, yb)
        print(f"epoch {e+1}/{epochs} done")

<<<<<<< HEAD
    def eval_split(Xmm, y, name, out_dir, batch_size, D, normalize_from_uint8):
        preds, scores = [], []
        for s in range(0, Xmm.shape[0], batch_size):
            Xb = Xmm[s:s+batch_size].reshape(-1, D).astype(np.float32)
            if normalize_from_uint8: 
                Xb /= 255.0
            preds.append(clf.predict(Xb))
            scores.append(clf.decision_function(Xb))  # signed margins
        y_pred  = np.concatenate(preds)
        y_score = np.concatenate(scores)

        acc = accuracy_score(y, y_pred)
        print(f"[{name}] acc: {acc:.4f}")
        print(classification_report(y, y_pred,
            labels=[0,1], target_names=['fake','real'], digits=4, zero_division=0))
        cm = confusion_matrix(y, y_pred, labels=[0,1])
        print("Confusion matrix:\n", cm)

        # plots
        out_prefix = str(Path(out_dir) / name.lower())
        plot_conf_mat(cm, ['fake','real'], Path(f"{out_prefix}_cm.png"))
        plot_roc_pr(y, y_score, out_prefix)  # saves *_roc.png and *_pr.png
        plot_score_regression(y, y_score, Path(f"{out_prefix}_score_reg.png"))

        return acc

    val_acc  = eval_split(X_val,  y_val,  "VAL",  out_dir, batch_size, D, normalize_from_uint8)
    test_acc = eval_split(X_test, y_test, "TEST", out_dir, batch_size, D, normalize_from_uint8)
=======
    def eval_split(Xmm, y, name):
        preds = []
        for s in range(0, Xmm.shape[0], batch_size):
            Xb = Xmm[s:s+batch_size].reshape(-1, D).astype(np.float32)
            if normalize_from_uint8: Xb /= 255.0
            preds.append(clf.predict(Xb))
        y_pred = np.concatenate(preds)
        print(f"[{name}] acc: {accuracy_score(y, y_pred):.4f}")
        print(classification_report(y, y_pred,
              labels=[0,1], target_names=['fake','real'], digits=4, zero_division=0))
        print("Confusion matrix:\n", confusion_matrix(y, y_pred, labels=[0,1]))

    eval_split(X_val,  y_val,  "VAL")
    eval_split(X_test, y_test, "TEST")
>>>>>>> 67853e16185c53edbd3d9580bd5864ee25bb7cf6

    joblib.dump(clf, out_dir/"svm_linear_streaming.joblib")
    print("Saved ->", (out_dir/"svm_linear_streaming.joblib").resolve())


<<<<<<< HEAD
    
=======
>>>>>>> 67853e16185c53edbd3d9580bd5864ee25bb7cf6
def main():
    #ensure splits exist
    split_data()
    #assert validate_npy(Path("./npy_out")), "Saved .npy files look invalid"
    train_linear_svm_streaming(out_dir=Path("./npy_out"), batch_size=512, epochs=2)
    eval_saved_test()
    explore(data_path='/Users/ryancalderon/Desktop/CSUSB_Courses/Fall_2025_Classes/CSE 5160 - Machine Learning/project1Code/images')

<<<<<<< HEAD
main()
=======
main()
>>>>>>> 67853e16185c53edbd3d9580bd5864ee25bb7cf6
