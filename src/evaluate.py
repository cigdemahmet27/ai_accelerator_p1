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

RESULTS_DIR = 'results'


def set_results_dir(path):
    global RESULTS_DIR
    RESULTS_DIR = path


def check_results_dir(latent_dim=None):
    if latent_dim is not None:
        path = f'{RESULTS_DIR}/latent_{latent_dim}'
    else:
        path = RESULTS_DIR
    if not os.path.exists(path):
        os.makedirs(path)


def plot_reconstructions(model, dataloader, device, latent_dim):
    model.eval()

    all_images = []
    all_labels = []
    for imgs, lbls in dataloader:
        all_images.append(imgs)
        all_labels.append(lbls)
    all_images = torch.cat(all_images)
    all_labels = torch.cat(all_labels)

    selected_indices = []
    for digit in range(10):
        digit_indices = (all_labels == digit).nonzero(as_tuple=True)[0]
        rand_idx = digit_indices[torch.randint(len(digit_indices), (1,)).item()]
        selected_indices.append(rand_idx)

    selected_images = all_images[torch.tensor(selected_indices)].to(device)

    with torch.no_grad():
        reconstructed = model(selected_images)

    orig_np = selected_images.cpu().numpy()
    recon_np = reconstructed.cpu().numpy()

    fig, axes = plt.subplots(
        nrows=2, ncols=10, figsize=(20, 4),
        sharex=True, sharey=True,
    )

    for i in range(10):
        axes[0][i].imshow(np.squeeze(orig_np[i]), cmap='gray')
        axes[0][i].set_title(str(i), fontsize=12)
        axes[0][i].axis('off')

        axes[1][i].imshow(np.squeeze(recon_np[i]), cmap='gray')
        axes[1][i].axis('off')

    axes[0][0].set_ylabel('Original', fontsize=14, rotation=0, labelpad=70)
    axes[1][0].set_ylabel('Reconstructed', fontsize=14, rotation=0, labelpad=70)

    fig.suptitle(f'Reconstruction Results (latent_dim={latent_dim})', fontsize=16)
    plt.tight_layout()
    check_results_dir(latent_dim)
    plt.savefig(f'{RESULTS_DIR}/latent_{latent_dim}/reconstructions.png', dpi=150)
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
    plt.savefig(f'{RESULTS_DIR}/latent_{latent_dim}/latent_space_{method.lower()}_true.png',
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
    plt.savefig(f'{RESULTS_DIR}/latent_{latent_dim}/latent_space_{method.lower()}_clusters.png',
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
    plt.savefig(f'{RESULTS_DIR}/latent_{latent_dim}/confusion_matrix.png', dpi=150)
    plt.close()

    # --- Save metrics to file ---
    metrics_path = f'{RESULTS_DIR}/latent_{latent_dim}/metrics.txt'
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


def plot_combined_loss_curves(all_loss_data):
    """Plot all latent dims' test loss curves on a single graph."""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(all_loss_data)))

    for i, data in enumerate(all_loss_data):
        dim = data['latent_dim']
        epochs = range(1, len(data['test_losses']) + 1)
        ax.plot(epochs, data['test_losses'], label=f'dim={dim}',
                color=colors[i], linewidth=2)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Test MSE Loss', fontsize=12)
    ax.set_title('Test Loss Convergence Across Latent Dimensions', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    plt.tight_layout()
    check_results_dir()
    plt.savefig(f'{RESULTS_DIR}/combined_loss_curves.png', dpi=150)
    plt.close()


def plot_metrics_comparison(all_results):
    """Bar charts comparing PMS and TD across latent dimensions."""
    dims = [r['latent_dim'] for r in all_results]
    pms_vals = [r['pms'] for r in all_results]
    td_vals = [r['td'] for r in all_results]
    test_loss_vals = [r['final_test_loss'] for r in all_results]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    x = np.arange(len(dims))
    dim_labels = [str(d) for d in dims]

    colors_pms = ['#27AE60' if v == min(pms_vals) else '#2E86C1' for v in pms_vals]
    bars = axes[0].bar(x, pms_vals, color=colors_pms, edgecolor='white', width=0.6)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(dim_labels)
    axes[0].set_xlabel('Latent Dimension')
    axes[0].set_ylabel('PMS (%)')
    axes[0].set_title('Percentage of Mislabeled Samples (lower is better)')
    for bar, val in zip(bars, pms_vals):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)

    colors_td = ['#27AE60' if v == max(td_vals) else '#E67E22' for v in td_vals]
    bars = axes[1].bar(x, td_vals, color=colors_td, edgecolor='white', width=0.6)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(dim_labels)
    axes[1].set_xlabel('Latent Dimension')
    axes[1].set_ylabel('TD')
    axes[1].set_title('Total Distance Between Centers (higher is better)')
    for bar, val in zip(bars, td_vals):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                     f'{val:.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)

    colors_loss = ['#27AE60' if v == min(test_loss_vals) else '#8E44AD' for v in test_loss_vals]
    bars = axes[2].bar(x, test_loss_vals, color=colors_loss, edgecolor='white', width=0.6)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(dim_labels)
    axes[2].set_xlabel('Latent Dimension')
    axes[2].set_ylabel('Test MSE Loss')
    axes[2].set_title('Reconstruction Error (lower is better)')
    for bar, val in zip(bars, test_loss_vals):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0002,
                     f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    axes[2].grid(axis='y', alpha=0.3)

    plt.suptitle('Metric Comparison Across Latent Dimensions', fontsize=16, y=1.02)
    plt.tight_layout()
    check_results_dir()
    plt.savefig(f'{RESULTS_DIR}/metrics_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_reconstruction_comparison(models_dict, dataloader, device):
    """Grid: rows = latent dims, columns = digits 0-9. Shows reconstruction quality across dims."""
    all_images = []
    all_labels = []
    for imgs, lbls in dataloader:
        all_images.append(imgs)
        all_labels.append(lbls)
    all_images = torch.cat(all_images)
    all_labels = torch.cat(all_labels)

    torch.manual_seed(42)
    selected_indices = []
    for digit in range(10):
        digit_indices = (all_labels == digit).nonzero(as_tuple=True)[0]
        rand_idx = digit_indices[torch.randint(len(digit_indices), (1,)).item()]
        selected_indices.append(rand_idx)

    selected_images = all_images[torch.tensor(selected_indices)]
    dims = sorted(models_dict.keys())
    n_rows = 1 + len(dims)

    fig, axes = plt.subplots(nrows=n_rows, ncols=10, figsize=(20, 2 * n_rows))

    for i in range(10):
        axes[0][i].imshow(np.squeeze(selected_images[i].numpy()), cmap='gray')
        axes[0][i].set_title(str(i), fontsize=11)
        axes[0][i].axis('off')
    axes[0][0].set_ylabel('Original', fontsize=11, rotation=0, labelpad=60)

    for row, dim in enumerate(dims, start=1):
        model = models_dict[dim]
        model.eval()
        with torch.no_grad():
            recon = model(selected_images.to(device)).cpu().numpy()
        for i in range(10):
            axes[row][i].imshow(np.squeeze(recon[i]), cmap='gray')
            axes[row][i].axis('off')
        axes[row][0].set_ylabel(f'dim={dim}', fontsize=11, rotation=0, labelpad=60)

    fig.suptitle('Reconstruction Comparison Across Latent Dimensions', fontsize=16)
    plt.tight_layout()
    check_results_dir()
    plt.savefig(f'{RESULTS_DIR}/reconstruction_comparison.png', dpi=150)
    plt.close()


def plot_tsne_comparison(models_dict, dataloader, device):
    """Side-by-side t-SNE plots colored by K-Means clusters for each latent dim."""
    dims = sorted(models_dict.keys())
    n_dims = len(dims)

    fig, axes = plt.subplots(1, n_dims, figsize=(5 * n_dims, 5))
    if n_dims == 1:
        axes = [axes]

    for idx, dim in enumerate(dims):
        model = models_dict[dim]
        model.eval()
        latent_reps = []
        with torch.no_grad():
            for images, _ in dataloader:
                images = images.to(device)
                z = model.encode(images)
                latent_reps.append(z.cpu().numpy())
        latent_reps = np.concatenate(latent_reps)

        kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(latent_reps)

        tsne = TSNE(n_components=2, random_state=42, perplexity=40, max_iter=1000)
        reduced = tsne.fit_transform(latent_reps)

        scatter = axes[idx].scatter(reduced[:, 0], reduced[:, 1],
                                    c=cluster_labels, cmap='tab10', alpha=0.4, s=3)
        axes[idx].set_title(f'dim={dim}', fontsize=13, fontweight='bold')
        axes[idx].set_xticks([])
        axes[idx].set_yticks([])

    fig.suptitle('t-SNE Cluster Comparison Across Latent Dimensions', fontsize=16)
    plt.tight_layout()
    check_results_dir()
    plt.savefig(f'{RESULTS_DIR}/tsne_comparison.png', dpi=150)
    plt.close()
