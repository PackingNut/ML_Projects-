#!/usr/bin/env python3
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram
import umap


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# UMAP 
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
UMAP_COMPONENTS = 2

# Sampling for hierarchical/dendrogram/silhouette (keeps visuals feasible)
SAMPLE_FOR_HIER = 5000

# number of clusters chosen
K = 2

#File load
def load_creditcard_csv():
    # try local file first
    local = Path("creditcard.csv")
    if local.exists():
        return str(local)
    # No file otherwise try kagglehub 
    try:
        import kagglehub
    except Exception:
        return None
    try:
        p = Path(kagglehub.dataset_download("mlg-ulb/creditcardfraud"))
        # if zip or folder, search for csv
        if p.is_file() and p.suffix.lower() == ".zip":
            import zipfile
            extract_dir = p.with_suffix("")
            with zipfile.ZipFile(p, "r") as zf:
                zf.extractall(extract_dir)
            p = extract_dir
        for f in p.rglob("creditcard*.csv"):
            return str(f)
    except Exception:
        return None
    return None

#%%
# Step 1: Load dataset and remove labels to make it unsupervised

csv_path = load_creditcard_csv()
if csv_path is None:
    print("ERROR: Cannot find 'creditcard.csv' locally and kagglehub unavailable.")
    print("Place 'creditcard.csv' in this folder or set up kagglehub.")
    sys.exit(1)

df = pd.read_csv(csv_path)
print(f"Loaded dataset with shape: {df.shape}")

if "Class" in df.columns:
    df = df.drop(columns=["Class"])
    print("Dropped 'Class' column — running fully unsupervised.")

#%%
# Step 2: Scale features

X = df.values.astype(float)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Standard scaling complete.")

#%%
# Step 3: UMAP -> 2D on full dataset
print("Running UMAP on full dataset (this may take a minute)...")
umap_reducer = umap.UMAP(
    n_neighbors=UMAP_N_NEIGHBORS,
    min_dist=UMAP_MIN_DIST,
    n_components=UMAP_COMPONENTS,
    metric="euclidean",
    random_state=RANDOM_SEED,
)
X_umap = umap_reducer.fit_transform(X_scaled)
print("UMAP embedding complete. Shape:", X_umap.shape)

# Show UMAP full scatter 
plt.figure(figsize=(8, 5))
plt.scatter(X_umap[:, 0], X_umap[:, 1], s=1.5, alpha=0.6)
plt.title("Step 3: UMAP 2D (full dataset) — unsupervised")
plt.xlabel("UMAP1"); plt.ylabel("UMAP2")
plt.grid(alpha=0.3)
plt.show()

#%%
# Step 4: Subset for hierarchical clustering, dendrogram, silhouette

n_total = X_umap.shape[0]
sample_size = min(SAMPLE_FOR_HIER, n_total)
rng = np.random.default_rng(RANDOM_SEED)
sample_idx = rng.choice(n_total, size=sample_size, replace=False)
X_sample = X_umap[sample_idx]
print(f"Using sampled subset of size {len(X_sample)} for hierarchical clustering visuals.")

#%%
# Step 5: Hierarchical clustering on thesubset 

print("Computing linkage matrix (Ward) for the sample (for dendrogram)...")
Z = linkage(X_sample, method="ward")

plt.figure(figsize=(10, 5))
dendrogram(Z, no_labels=True, color_threshold=None)
plt.title("Step 5: Dendrogram (Ward linkage) — sampled UMAP 2D")
plt.xlabel("Sample index")
plt.ylabel("Distance")
plt.show()

#%%
# Step 6: Silhouette scores on sampled subset to sanity-check k 

ks = range(2, 11)
sil_scores = []
for k in ks:
    labels_k = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X_sample)
    try:
        s = silhouette_score(X_sample, labels_k)
    except Exception:
        s = float("nan")
    sil_scores.append(s)

plt.figure(figsize=(7, 4))
plt.plot(list(ks), sil_scores, marker="o")
plt.title("Step 6: Silhouette scores (sampled subset)")
plt.xlabel("Number of clusters k")
plt.ylabel("Silhouette score")
plt.grid(alpha=0.3)
plt.show()

print("Selected number of clusters (fixed):", K)

#%%
# Step 7: Fit Agglomerative (k=2) on sampled subset to get cluster centroids

print(f"Fitting AgglomerativeClustering (k={K}) on the sampled subset to compute centroids...")
sample_labels = AgglomerativeClustering(n_clusters=K, linkage="ward").fit_predict(X_sample)

# compute centroids of clusters in the sample
centroids = np.vstack([X_sample[sample_labels == c].mean(axis=0) for c in range(K)])
cluster_sizes = [np.sum(sample_labels == c) for c in range(K)]
print("Sample cluster sizes:", cluster_sizes)
print("Centroids (sample):", centroids)

#%%
# Step 8: Assign every full UMAP point to nearest centroid for full dataset

def assign_to_nearest_centroid(points, centroids):
    # returns assigned labels and distances to centroid
    dists = np.linalg.norm(points[:, None, :] - centroids[None, :, :], axis=2)  # (N, K)
    labels = np.argmin(dists, axis=1)
    min_dists = dists[np.arange(dists.shape[0]), labels]
    return labels, min_dists

full_labels, full_dists = assign_to_nearest_centroid(X_umap, centroids)
# compute mean distance per cluster (on full assignment)
mean_dist_per_cluster = [full_dists[full_labels == c].mean() if np.any(full_labels==c) else np.inf for c in range(K)]
size_per_cluster = [np.sum(full_labels == c) for c in range(K)]
print("Full-assignment cluster sizes:", size_per_cluster)
print("Full-assignment mean distance to centroid per cluster:", mean_dist_per_cluster)

#%%
# Step 9: Identify fraud cluster for distance-based

# cluster with larger mean distance-to-centroid is considered more abnormal
fraud_cluster = int(np.argmax(mean_dist_per_cluster))
print(f"Identified cluster {fraud_cluster} as the 'fraud' cluster (higher mean distance).")

#mark top suspicious points within fraud cluster:
# We'll mark the top X% farthest points in that cluster as "suspected fraud"
SUSPECT_TOP_PERCENT = 0.015 
fraud_mask_full = (full_labels == fraud_cluster)
fraud_indices = np.where(fraud_mask_full)[0]
fraud_dists = full_dists[fraud_mask_full]
n_suspects = max(1, int(np.ceil(len(fraud_dists) * SUSPECT_TOP_PERCENT)))
# get global indices of top suspicious points
suspect_order = np.argsort(-fraud_dists)
top_suspect_idxs_in_fraud = fraud_indices[suspect_order[:n_suspects]]

print(f"Flagging {len(top_suspect_idxs_in_fraud)} suspected fraud points (top {SUSPECT_TOP_PERCENT*100:.2f}% of that cluster).")

#%%
#Plot final UMAP with cluster coloring and suspected fraud highlighted

plt.figure(figsize=(9, 6))
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
for c in range(K):
    mask = full_labels == c
    plt.scatter(X_umap[mask, 0], X_umap[mask, 1], s=2, alpha=0.5, c=colors[c % len(colors)], label=f"Cluster {c}")
#suspected fraud points with red stars
plt.scatter(X_umap[top_suspect_idxs_in_fraud, 0], X_umap[top_suspect_idxs_in_fraud, 1],
            s=40, c="red", marker="*", label="Suspected fraud (top outliers)")

#centroids
plt.scatter(centroids[:, 0], centroids[:, 1], s=80, c="black", marker="X", label="Centroids")
plt.title("UMAP 2D embedding — clusters (full) and suspected fraud highlights")
plt.xlabel("UMAP1"); plt.ylabel("UMAP2")
plt.legend(loc="best", markerscale=3)
plt.grid(alpha=0.2)
plt.show()

#Summary
print("\nSummary")
for c in range(K):
    print(f"Cluster {c}: size = {size_per_cluster[c]}, mean distance to centroid = {mean_dist_per_cluster[c]:.4f}")
print(f"Fraud cluster (by distance-based rule): {fraud_cluster}")
print(f"Number of flagged suspected fraud points (top {SUSPECT_TOP_PERCENT*100:.2f}% of that cluster): {len(top_suspect_idxs_in_fraud)}")
print("Note: This is unsupervised detection — suspected flags are heuristic and need human review.")
