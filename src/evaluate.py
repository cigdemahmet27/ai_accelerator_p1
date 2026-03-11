import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
import os


def check_results_dir(latent_dim=None):
    if latent_dim is not None:
        path = f'results/latent_{latent_dim}'
    else:
        path = 'results'
    if not os.path.exists(path):
        os.makedirs(path)


def plot_reconstructions(model, dataloader, device, latent_dim, num_images=10):
    model.eval()
    dataiter = iter(dataloader)
    images, labels = next(dataiter)
    images = images.to(device)

    with torch.no_grad():
        reconstructed = model(images)

    images = images.cpu().numpy()
    reconstructed = reconstructed.cpu().numpy()

    fig, axes = plt.subplots(
        nrows=2, ncols=num_images, figsize=(20, 4),
        sharex=True, sharey=True,
    )

    for img, ax in zip(images[:num_images], axes[0]):
        ax.imshow(np.squeeze(img), cmap='gray')
        ax.axis('off')

    for img, ax in zip(reconstructed[:num_images], axes[1]):
        ax.imshow(np.squeeze(img), cmap='gray')
        ax.axis('off')

    axes[0][0].set_ylabel('Original', fontsize=14, rotation=0, labelpad=70)
    axes[1][0].set_ylabel('Reconstructed', fontsize=14, rotation=0, labelpad=70)

    fig.suptitle(f'Reconstruction Results (latent_dim={latent_dim})', fontsize=16)
    plt.tight_layout()
    check_results_dir(latent_dim)
    plt.savefig(f'results/latent_{latent_dim}/reconstructions.png', dpi=150)
    plt.close()


def extract_latent_representations(model, dataloader, device):
    model.eval()
    latent_reps = []
    labels_list = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            z = model.encode(images)
            latent_reps.append(z.cpu().numpy())
            labels_list.append(labels.numpy())

    return np.concatenate(latent_reps), np.concatenate(labels_list)


def plot_latent_space(latent_reps, labels, latent_dim, method='PCA'):
    if method == 'PCA':
        reducer = PCA(n_components=2)
    elif method == 't-SNE':
        reducer = TSNE(n_components=2, random_state=42, perplexity=40, max_iter=1000)
    else:
        raise ValueError("Method must be 'PCA' or 't-SNE'")

    reduced_reps = reducer.fit_transform(latent_reps)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        reduced_reps[:, 0], reduced_reps[:, 1],
        c=labels, cmap='tab10', alpha=0.5, s=5,
    )
    cbar = plt.colorbar(scatter, ticks=range(10))
    cbar.set_label('Digit')
    plt.title(f'Latent Space – True Labels ({method}, latent_dim={latent_dim})')
    plt.xlabel(f'{method} Component 1')
    plt.ylabel(f'{method} Component 2')
    plt.tight_layout()

    check_results_dir(latent_dim)
    plt.savefig(f'results/latent_{latent_dim}/latent_space_{method.lower()}_true.png',
                dpi=150)
    plt.close()


def plot_cluster_space(latent_reps, cluster_labels, latent_dim, method='PCA'):
    """Visualize latent space colored by K-Means cluster assignments."""
    if method == 'PCA':
        reducer = PCA(n_components=2)
    elif method == 't-SNE':
        reducer = TSNE(n_components=2, random_state=42, perplexity=40, max_iter=1000)
    else:
        raise ValueError("Method must be 'PCA' or 't-SNE'")

    reduced_reps = reducer.fit_transform(latent_reps)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        reduced_reps[:, 0], reduced_reps[:, 1],
        c=cluster_labels, cmap='tab10', alpha=0.5, s=5,
    )
    cbar = plt.colorbar(scatter, ticks=range(10))
    cbar.set_label('Cluster')
    plt.title(f'Latent Space – K-Means Clusters ({method}, latent_dim={latent_dim})')
    plt.xlabel(f'{method} Component 1')
    plt.ylabel(f'{method} Component 2')
    plt.tight_layout()

    check_results_dir(latent_dim)
    plt.savefig(f'results/latent_{latent_dim}/latent_space_{method.lower()}_clusters.png',
                dpi=150)
    plt.close()


def cluster_and_evaluate(latent_reps, true_labels, latent_dim, num_clusters=10):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(latent_reps)
    centers = kmeans.cluster_centers_

    # --- Cluster visualizations ---
    print("  Plotting cluster assignments (PCA)...")
    plot_cluster_space(latent_reps, cluster_labels, latent_dim, method='PCA')
    print("  Plotting cluster assignments (t-SNE)...")
    plot_cluster_space(latent_reps, cluster_labels, latent_dim, method='t-SNE')

    # --- Hungarian algorithm: map clusters to true labels ---
    cm = confusion_matrix(true_labels, cluster_labels)
    row_ind, col_ind = linear_sum_assignment(-cm)
    mapping = {col_ind[i]: row_ind[i] for i in range(num_clusters)}
    mapped_cluster_labels = np.array([mapping[l] for l in cluster_labels])

    mapped_cm = confusion_matrix(true_labels, mapped_cluster_labels)

    # --- Metrics ---

    # PMS: Percentage of Mislabeled Samples
    mislabeled = np.sum(true_labels != mapped_cluster_labels)
    pms = (mislabeled / len(true_labels)) * 100

    # AD: Average L2 distance of samples to their corresponding cluster center
    ad = 0.0
    for i in range(len(latent_reps)):
        center = centers[cluster_labels[i]]
        ad += np.sum((latent_reps[i] - center) ** 2)
    ad /= len(latent_reps)

    # AVC: Average variation of samples within clusters
    avc = 0.0
    for k in range(num_clusters):
        cluster_points = latent_reps[cluster_labels == k]
        if len(cluster_points) > 0:
            cluster_center = centers[k]
            var = np.sum(np.sum((cluster_points - cluster_center) ** 2, axis=1))
            var /= len(cluster_points)
            avc += var
    avc /= num_clusters

    # TD: Total L2 distance between cluster centers
    td = 0.0
    for i in range(num_clusters):
        for j in range(i + 1, num_clusters):
            td += np.sum((centers[i] - centers[j]) ** 2)

    # --- Confusion matrix plot ---
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        mapped_cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=range(10), yticklabels=range(10),
    )
    plt.title(f'Confusion Matrix (latent_dim={latent_dim})')
    plt.xlabel('Predicted (Mapped Cluster)')
    plt.ylabel('True Label')
    plt.tight_layout()
    check_results_dir(latent_dim)
    plt.savefig(f'results/latent_{latent_dim}/confusion_matrix.png', dpi=150)
    plt.close()

    # --- Save metrics to file ---
    metrics_path = f'results/latent_{latent_dim}/metrics.txt'
    with open(metrics_path, 'w') as f:
        f.write(f"Clustering Metrics (latent_dim={latent_dim})\n")
        f.write(f"{'='*45}\n")
        f.write(f"Percentage of Mislabeled Samples (PMS): {pms:.2f}%\n")
        f.write(f"Average L2 dist. to cluster centers (AD): {ad:.4f}\n")
        f.write(f"Average variation within clusters (AVC): {avc:.4f}\n")
        f.write(f"Total L2 distance between centers (TD): {td:.4f}\n")

    print(f"\n  Clustering Metrics (latent_dim={latent_dim}):")
    print(f"  PMS : {pms:.2f}%")
    print(f"  AD  : {ad:.4f}")
    print(f"  AVC : {avc:.4f}")
    print(f"  TD  : {td:.4f}")

    return pms, ad, avc, td
