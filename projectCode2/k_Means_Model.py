# This script explores whether unsupervised clustering can naturally separate
# fraudulent transactions from legitimate ones, without being explicitly told which is which during training.

import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import json
import matplotlib.pyplot as plt
import itertools

# sklearn imports for clustering and evaluation
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    confusion_matrix,
)
from sklearn.decomposition import PCA

# Data importing and splitting

def load_csv_and_split(
    csv_path,
    label_col="Class",
    test_frac=0.2,
    val_frac=0.2,
    seed=1337,
):
    # Load CSV data
    df = pd.read_csv(csv_path)
    
    # Separate features from the label column
    # Everything except class is a feature (Time, V1-V28, Amount), 30 in total without class
    feature_cols = [c for c in df.columns if c != label_col]
    X_all = df[feature_cols].values.astype(np.float32)
    y_all = df[label_col].values.astype(np.int64) 
    
    # Shuffle data to avoid ordering bias
    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    
    X_all = X_all[idx]
    y_all = y_all[idx]
    
    # Calculate samples going into each split
    N = len(df)
    test_size = int(N * test_frac)
    val_size = int(N * val_frac)
    train_size = N - val_size - test_size
    
    # Split data into three chunks
    X_train = X_all[:train_size]
    y_train = y_all[:train_size]
    
    X_val = X_all[train_size:train_size + val_size]
    y_val = y_all[train_size:train_size + val_size]
    
    X_test = X_all[train_size + val_size:]
    y_test = y_all[train_size + val_size:]
    
    return (X_train, y_train, X_val, y_val, X_test, y_test, feature_cols)

# Preprocessing
def fit_scaler(X_train):
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def apply_scaler(scaler, X):
    return scaler.transform(X).astype(np.float32)


# Train K-Means (Unsupervised Clustering)
def train_kmeans(
    X_train_scaled,
    n_clusters=3,
    rng_seed=1337,
):
    km = KMeans(
        n_clusters=n_clusters,
        n_init=10,  # Run K-Means 10 times with different initializations
        random_state=rng_seed,
    )
    km.fit(X_train_scaled)
    return km

# Internal Metrics
def internal_metrics(model, X_scaled, split_name="TRAIN"):
    preds = model.predict(X_scaled)
    
    sil = silhouette_score(X_scaled, preds)
    ch = calinski_harabasz_score(X_scaled, preds)
    db = davies_bouldin_score(X_scaled, preds)
    
    print(f"\n[{split_name}] Internal cluster quality (how well-defined are the clusters?):")
    print(f"  Silhouette Score:           {sil:.4f}  (higher is better, range: -1 to 1)")
    print(f"  Calinski-Harabasz Index:    {ch:.4f}  (higher is better)")
    print(f"  Davies-Bouldin Index:       {db:.4f}  (lower is better)")
    
    return {
        "silhouette": sil,
        "calinski_harabasz": ch,
        "davies_bouldin": db,
        "cluster_assignments": preds,
    }



# External Metricss
def external_metrics(y_true, cluster_assignments, split_name="TRAIN"):
    ari = adjusted_rand_score(y_true, cluster_assignments)
    nmi = normalized_mutual_info_score(y_true, cluster_assignments)
    cm = confusion_matrix(y_true, cluster_assignments)
    
    print(f"\n[{split_name}] External alignment (how well do clusters match fraud labels?):")
    print(f"  Adjusted Rand Index (ARI):        {ari:.4f}  (1.0 = perfect)")
    print(f"  Normalized Mutual Info (NMI):     {nmi:.4f}  (1.0 = perfect)")
    print("  Confusion Matrix (rows=true label 0/1, cols=cluster IDs):")
    print(cm)
    
    return {
        "ARI": ari,
        "NMI": nmi,
        "confusion_matrix": cm,
    }



# Visualization (2D PCA Projection)
def plot_clusters_2d(X_scaled, cluster_assignments, y_true=None, out_path="clusters_2d.png"):
    # Reduce from 30D to 2D using Principal Component Analysis
    pca = PCA(n_components=2, random_state=0)
    X_2d = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(8, 6))
    
    # Plot each cluster with a different color
    for c in np.unique(cluster_assignments):
        mask = (cluster_assignments == c)
        plt.scatter(
            X_2d[mask, 0],
            X_2d[mask, 1],
            s=10,
            alpha=0.5,
            label=f"Cluster {c}",
        )
    
    # Overlay fraud transactions with 'x' markers if labels are available
    if y_true is not None:
        fraud_mask = (y_true == 1)
        plt.scatter(
            X_2d[fraud_mask, 0],
            X_2d[fraud_mask, 1],
            s=30,
            alpha=0.8,
            marker='x',
            linewidths=1.5,
            color='red',
            label="Fraud (True Label)",
        )
    
    plt.title("K-Means Clusters Visualized in 2D PCA Space")
    plt.xlabel("First Principal Component")
    plt.ylabel("Second Principal Component")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()
    print(f"\n Saved cluster visualization -> {Path(out_path).resolve()}")



#K selection (Elbow + Silhouette vs k)
def choose_k_elbow_silhouette(X_train_s, X_val_s=None, k_values=(2,3,4), out_dir=Path("./kmeans_out")):
    
    out_dir.mkdir(parents=True, exist_ok=True)
    inertia_list, sil_list, kvals = [], [], list(k_values)

    for k in kvals:
        km = KMeans(n_clusters=k, n_init=10, random_state=1337)
        km.fit(X_train_s)
        inertia_list.append(float(km.inertia_))
        X_for_sil = X_val_s if X_val_s is not None else X_train_s
        preds = km.predict(X_for_sil)
        sil = silhouette_score(X_for_sil, preds)
        sil_list.append(float(sil))

    # Elbow plot
    plt.figure(figsize=(5,4))
    plt.plot(kvals, inertia_list, marker='o')
    plt.title("Elbow (WCSS vs k)")
    plt.xlabel("k")
    plt.ylabel("Within-Cluster Sum of Squares")
    plt.tight_layout()
    plt.savefig(out_dir / "elbow.png", dpi=140)
    plt.close()

    # Silhouette vs k
    plt.figure(figsize=(5,4))
    plt.plot(kvals, sil_list, marker='o')
    plt.title("Silhouette (mean) vs k")
    plt.xlabel("k")
    plt.ylabel("Mean Silhouette Score")
    plt.tight_layout()
    plt.savefig(out_dir / "silhouette_vs_k.png", dpi=140)
    plt.close()

    return {k: {"inertia": i, "silhouette": s} for k, i, s in zip(kvals, inertia_list, sil_list)}



# Model stability using a split-half jaccard
def _same_cluster_pairs(labels):
    groups = {}
    for i, c in enumerate(labels):
        groups.setdefault(int(c), []).append(i)
    pairs = set()
    for idxs in groups.values():
        for i, j in itertools.combinations(idxs, 2):
            pairs.add((i, j))
    return pairs


def _jaccard_same_cluster(yhat1, yhat2):
    s1 = _same_cluster_pairs(yhat1)
    s2 = _same_cluster_pairs(yhat2)
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    inter = len(s1.intersection(s2))
    union = len(s1.union(s2))
    return inter / union if union else 0.0


def stability_split_half_jaccard(
    X_scaled, 
    k=2, 
    n_repeats=10, 
    sample_size=2000, 
    rng_seed=1337,
    out_dir=Path("./kmeans_out")
):
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(rng_seed)
    N = X_scaled.shape[0]
    scores = []

    for r in range(n_repeats):
        idx = rng.choice(N, size=min(sample_size, N), replace=False)
        S = X_scaled[idx]

        # split halves on the sampled pool
        rng.shuffle(idx)
        mid = len(idx)//2
        A_idx, B_idx = idx[:mid], idx[mid:]
        XA, XB = X_scaled[A_idx], X_scaled[B_idx]

        kma = KMeans(n_clusters=k, n_init=10, random_state=int(rng.integers(1e9)))
        kmb = KMeans(n_clusters=k, n_init=10, random_state=int(rng.integers(1e9)))
        kma.fit(XA)
        kmb.fit(XB)

        ya = kma.predict(S)
        yb = kmb.predict(S)

        j = _jaccard_same_cluster(ya, yb)
        scores.append(j)

    scores = np.asarray(scores, dtype=np.float32)
    mean_j, std_j = float(scores.mean()), float(scores.std())

    with open(out_dir / f"stability_k{k}.txt", "w") as f:
        f.write("Split-half Jaccard per repeat:\n")
        for i, v in enumerate(scores):
            f.write(f"  rep {i+1:02d}: {v:.4f}\n")
        f.write(f"\nMean: {mean_j:.4f}   Std: {std_j:.4f}\n")

    return mean_j, std_j



# Practical utility
def cluster_profiles(
    X_scaled, 
    feature_cols, 
    kmeans_model, 
    y_true=None, 
    out_dir=Path("./kmeans_out"), 
    prefix="train"
):
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = kmeans_model.predict(X_scaled)
    k = kmeans_model.n_clusters
    df_means, df_meds, summary_rows = [], [], []

    for c in range(k):
        mask = (labels == c)
        n = int(mask.sum())
        frac = n / len(labels)
        fraud_rate = None
        if y_true is not None and n > 0:
            y_sub = np.asarray(y_true)[mask]
            fraud_rate = float((y_sub == 1).mean())
        elif y_true is None:
            fraud_rate = None
        else:
            fraud_rate = float("nan")

        Xm = X_scaled[mask]
        means = Xm.mean(axis=0) if n>0 else np.zeros(len(feature_cols))
        meds  = np.median(Xm, axis=0) if n>0 else np.zeros(len(feature_cols))

        df_means.append(pd.Series(means, index=feature_cols, name=f"cluster_{c}"))
        df_meds.append(pd.Series(meds,  index=feature_cols, name=f"cluster_{c}"))

        summary_rows.append({
            "cluster": c,
            "count": n,
            "fraction": frac,
            "fraud_rate": fraud_rate
        })

    means_df = pd.DataFrame(df_means)
    meds_df  = pd.DataFrame(df_meds)
    means_df.to_csv(out_dir / f"{prefix}_cluster_feature_means.csv")
    meds_df.to_csv(out_dir / f"{prefix}_cluster_feature_medians.csv")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / f"{prefix}_cluster_summary.csv", index=False)

    # quick line plot of per-cluster means (scaled space)
    plt.figure(figsize=(8,4))
    for c in range(k):
        plt.plot(range(len(feature_cols)), means_df.loc[f"cluster_{c}"].values, marker='.')
    plt.title(f"{prefix}: per-cluster feature means (scaled)")
    plt.xlabel("feature index")
    plt.ylabel("mean (scaled)")
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_cluster_means_plot.png", dpi=140)
    plt.close()

    return summary_df, means_df, meds_df


def example_members_closest_to_centroid(
    X_scaled, 
    kmeans_model, 
    original_rows_df, 
    top_n=10, 
    out_dir=Path("./kmeans_out"), 
    prefix="test"
):
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = kmeans_model.predict(X_scaled)
    centers = kmeans_model.cluster_centers_
    rows = []

    for c in range(kmeans_model.n_clusters):
        idxs = np.where(labels == c)[0]
        if idxs.size == 0:
            continue
        Xc = X_scaled[idxs]
        dists = np.linalg.norm(Xc - centers[c], axis=1)
        order = np.argsort(dists)[:top_n]
        chosen = idxs[order]
        for rank, (i_global, d) in enumerate(zip(chosen, dists[order]), start=1):
            rec = {"cluster": int(c), "rank": rank, "index_in_split": int(i_global), "distance": float(d)}
            for col in original_rows_df.columns:
                rec[col] = original_rows_df.iloc[i_global][col]
            rows.append(rec)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"{prefix}_example_members.csv", index=False)
    return df



# Predict Clusters for new data
def predict_cluster_for_new_rows(model, scaler, new_rows_df, feature_cols, out_dir="./kmeans_out"):
    # Extract features in the correct order
    X_new = new_rows_df[feature_cols].values.astype(np.float32)
    
    # Apply the same scaling transformation used during training
    X_new_scaled = apply_scaler(scaler, X_new)
    
    # Predict cluster membership
    clusters = model.predict(X_new_scaled)
    
    print("\n Cluster assignments for new transactions:")
    for i, c in enumerate(clusters):
        print(f"Transaction {i}: Cluster {int(c)}")
    
    # Save predictions to CSV
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "transaction_index": np.arange(len(clusters)),
        "cluster": clusters.astype(int),
    }).to_csv(out_dir / "new_transactions_predictions.csv", index=False)
    print(f"Saved predictions -> {out_dir/'new_transactions_predictions.csv'}")



# Main Pipeline
def main():
    print("=" * 70)
    print("K-means clustering for fraud detection (Unsupervised)")
    print("=" * 70)
    
    # Configuration
    CSV_PATH = "./creditcard.csv" 
    OUT_DIR = Path("./kmeans_out")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load and split the data
    print("\n Loading and splitting dataset...")
    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        feature_cols
    ) = load_csv_and_split(
        csv_path=CSV_PATH,
        label_col="Class",
        test_frac=0.2,  # 20% for final testing
        val_frac=0.2,   # 20% for validation
        seed=1337,      # For reproducibility
    )
    
    print("\n Dataset shapes:")
    print(f"   Train: {X_train.shape[0]:,} samples x {X_train.shape[1]} features")
    print(f"   Val:   {X_val.shape[0]:,} samples x {X_val.shape[1]} features")
    print(f"   Test:  {X_test.shape[0]:,} samples x {X_test.shape[1]} features")
    print(f"   Fraud rate in train: {y_train.mean():.2%}")
    
    # Standardize features
    print("\n Standardizing features...")
    scaler = fit_scaler(X_train)
    X_train_s = apply_scaler(scaler, X_train)
    X_val_s   = apply_scaler(scaler, X_val)
    X_test_s  = apply_scaler(scaler, X_test)
    
    # Save scaler and feature order for later use
    joblib.dump(scaler, OUT_DIR / "scaler.joblib")
    with open(OUT_DIR / "feature_cols.json", "w") as f:
        json.dump(feature_cols, f)
    print(f"Saved scaler and feature columns to {OUT_DIR}")

    # ----- K selection: evaluate k=2..4 via elbow + silhouette on val
    print("\n Selecting k via elbow + silhouette (k = 2..4)...")
    k_eval = choose_k_elbow_silhouette(
        X_train_s, 
        X_val_s=X_val_s, 
        k_values=(2,3,4), 
        out_dir=OUT_DIR
    )
    print("k evaluation (inertia & silhouette):", k_eval)

    # Optional auto-select by best val silhouette:
    best_k = max(k_eval.keys(), key=lambda kk: k_eval[kk]["silhouette"])
    print(f"Best k by validation silhouette appears to be: k = {best_k}")
    
    # Train K-means with selected k
    print("\n Training K-Means model (unsupervised clustering)...")
    kmeans_model = train_kmeans(
        X_train_scaled=X_train_s,
        n_clusters=best_k,  # was fixed=3; now auto-selected
        rng_seed=1337,
    )
    
    # Save the trained model
    joblib.dump(kmeans_model, OUT_DIR / "kmeans_model.joblib")
    print(f"Saved K-Means model to {OUT_DIR}")
    
    # Evaluate cluster quality (internal metrics)
    print("\n" + "="*70)
    print("Internal Metrics (Cluster Quality - No Labels Used)")
    print("="*70)
    train_int = internal_metrics(kmeans_model, X_train_s, split_name="TRAIN")
    val_int   = internal_metrics(kmeans_model, X_val_s,   split_name="VAL")
    test_int  = internal_metrics(kmeans_model, X_test_s,  split_name="TEST")
    
    # Compare clusters to true fraud labels (external metrics)
    print("\n" + "="*70)
    print("External Metrics (Cluster Alignment with True Labels)")
    print("="*70)
    train_ext = external_metrics(y_train, train_int["cluster_assignments"], "TRAIN")
    val_ext   = external_metrics(y_val,   val_int["cluster_assignments"],   "VAL")
    test_ext  = external_metrics(y_test,  test_int["cluster_assignments"],  "TEST")
    
    # Visualize clusters in 2D
    print("\n" + "="*70)
    print("Visualization")
    print("="*70)
    plot_clusters_2d(
        X_scaled=X_test_s,
        cluster_assignments=test_int["cluster_assignments"],
        y_true=y_test,
        out_path=OUT_DIR / "clusters_test_2d.png",
    )

    # ----- Model stability (split-half Jaccard on TRAIN)
    print("\n" + "="*70)
    print("Model Stability: Split-half bootstrapped Jaccard (TRAIN)")
    print("="*70)
    mean_j, std_j = stability_split_half_jaccard(
        X_train_s, 
        k=kmeans_model.n_clusters, 
        n_repeats=10, 
        sample_size=2000, 
        rng_seed=1337,
        out_dir=OUT_DIR
    )
    print(f"Stability (k={kmeans_model.n_clusters}): mean Jaccard = {mean_j:.4f} ± {std_j:.4f}")
    print(f"Scores saved to: {OUT_DIR / ('stability_k'+str(kmeans_model.n_clusters)+'.txt')}")

    # Practical utility: profiles + fraud rate (TRAIN/VAL/TEST)
    print("\n" + "="*70)
    print("Practical Utility: Cluster profiles & Fraud Rate per cluster")
    print("="*70)

    # Use frames for readable example-member dumps
    train_df_raw = pd.DataFrame(X_train, columns=feature_cols)
    val_df_raw   = pd.DataFrame(X_val,   columns=feature_cols)
    test_df_raw  = pd.DataFrame(X_test,  columns=feature_cols)

    train_summary, _, _ = cluster_profiles(
        X_scaled=X_train_s,
        feature_cols=feature_cols,
        kmeans_model=kmeans_model,
        y_true=y_train,
        out_dir=OUT_DIR,
        prefix="train"
    )
    val_summary, _, _ = cluster_profiles(
        X_scaled=X_val_s,
        feature_cols=feature_cols,
        kmeans_model=kmeans_model,
        y_true=y_val,
        out_dir=OUT_DIR,
        prefix="val"
    )
    test_summary, _, _ = cluster_profiles(
        X_scaled=X_test_s,
        feature_cols=feature_cols,
        kmeans_model=kmeans_model,
        y_true=y_test,
        out_dir=OUT_DIR,
        prefix="test"
    )
    print("Test cluster summary (count, fraction, fraud_rate):\n", 
          test_summary[["cluster","count","fraction","fraud_rate"]])

    # Example members on TEST
    examples_df = example_members_closest_to_centroid(
        X_scaled=X_test_s,
        kmeans_model=kmeans_model,
        original_rows_df=test_df_raw,
        top_n=10,
        out_dir=OUT_DIR,
        prefix="test"
    )
    print("Saved example members per cluster ->", OUT_DIR / "test_example_members.csv")
    
    # Demo prediction on "new" data
    print("\n" + "="*70)
    print("Demo: Predicting clusters for new transactions")
    print("="*70)
    print("Using first 5 test samples as a demo of how you'd score new data...")
    
    # Create a DataFrame from the first 5 test samples
    new_rows_df = pd.DataFrame(
        X_test[:5],
        columns=feature_cols
    )
    
    predict_cluster_for_new_rows(
        model=kmeans_model,
        scaler=scaler,
        new_rows_df=new_rows_df,
        feature_cols=feature_cols,
        out_dir=OUT_DIR,
    )
    
    print("\n" + "="*70)
    print("Check the output directory for saved models and plots.")
    print("="*70)


if __name__ == "__main__":
    main()
