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

def check_results_dir():
    if not os.path.exists('results'):
        os.makedirs('results')

def plot_reconstructions(model, dataloader, device, num_images=10):
    model.eval()
    dataiter = iter(dataloader)
    images, labels = next(dataiter)
    images = images.to(device)
    
    with torch.no_grad():
        reconstructed = model(images)
        
    images = images.cpu().numpy()
    reconstructed = reconstructed.cpu().numpy()
    
    fig, axes = plt.subplots(nrows=2, ncols=num_images, sharex=True, sharey=True, figsize=(20, 4))
    
    for images_row, row_axes in zip([images, reconstructed], axes):
        for img, ax in zip(images_row, row_axes):
            ax.imshow(np.squeeze(img), cmap='gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            
    check_results_dir()
    plt.savefig('results/reconstructions.png')
    plt.close()

def extract_latent_representations(model, dataloader, device):
    model.eval()
    latent_reps = []
    labels_list = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            images = images.view(images.size(0), -1)
            z = model.encoder(images)
            latent_reps.append(z.cpu().numpy())
            labels_list.append(labels.numpy())
            
    return np.concatenate(latent_reps), np.concatenate(labels_list)

def plot_latent_space(latent_reps, labels, method='PCA'):
    if method == 'PCA':
        reducer = PCA(n_components=2)
    elif method == 't-SNE':
        reducer = TSNE(n_components=2, random_state=42)
    else:
        raise ValueError("Method must be 'PCA' or 't-SNE'")
        
    reduced_reps = reducer.fit_transform(latent_reps)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_reps[:, 0], reduced_reps[:, 1], c=labels, cmap='tab10', alpha=0.6)
    plt.colorbar(scatter, ticks=range(10))
    plt.title(f'Latent Space Visualization ({method})')
    plt.xlabel(f'{method} Component 1')
    plt.ylabel(f'{method} Component 2')
    
    check_results_dir()
    plt.savefig(f'results/latent_space_{method.lower()}.png')
    plt.close()

def cluster_and_evaluate(latent_reps, true_labels, num_clusters=10):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(latent_reps)
    centers = kmeans.cluster_centers_

    # Map cluster labels to true labels using Hungarian algorithm
    cm = confusion_matrix(true_labels, cluster_labels)
    # cm[i, j] means cost if we assign cluster j to label i. We want to maximize sum(cm[i, j])
    # linear_sum_assignment finds min cost, so we pass -cm
    row_ind, col_ind = linear_sum_assignment(-cm)
    
    # Create mapping dictionary from cluster_label (col_ind) to true_label (row_ind)
    mapping = {col_ind[i]: row_ind[i] for i in range(num_clusters)}
    
    # Map the cluster labels
    mapped_cluster_labels = np.array([mapping[l] for l in cluster_labels])
    
    # Recalculate Confusion Matrix after mapping
    mapped_cm = confusion_matrix(true_labels, mapped_cluster_labels)
    
    # 1. Percentage of MisLabeled Samples (PMS)
    mislabeled = np.sum(true_labels != mapped_cluster_labels)
    pms = (mislabeled / len(true_labels)) * 100
    
    # 2. Average L2 distance of samples to corresponding cluster centers (AD)
    # Distance to the assigned cluster center (before mapping)
    ad = 0
    for i in range(len(latent_reps)):
        center = centers[cluster_labels[i]]
        ad += np.sum((latent_reps[i] - center)**2)
    ad /= len(latent_reps)
    
    # 3. Average variation of samples within clusters (AVC)
    avc = 0
    for k in range(num_clusters):
        cluster_points = latent_reps[cluster_labels == k]
        if len(cluster_points) > 0:
            cluster_center = centers[k]
            var = np.sum(np.sum((cluster_points - cluster_center)**2, axis=1)) / len(cluster_points)
            avc += var
    avc /= num_clusters
    
    # 4. Total L2 distance between cluster centers (TD)
    td = 0
    for i in range(num_clusters):
        for j in range(i + 1, num_clusters):
            td += np.sum((centers[i] - centers[j])**2)
            
    # Visualize mapping and confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(mapped_cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix (Clusters mapped to True Labels)')
    plt.xlabel('Predicted (Mapped Cluster)')
    plt.ylabel('True Label')
    check_results_dir()
    plt.savefig('results/confusion_matrix.png')
    plt.close()
    
    print("Clustering Metrics:")
    print(f"Percentage of Mislabeled Samples (PMS): {pms:.2f}%")
    print(f"Average dist. to cluster centers (AD): {ad:.4f}")
    print(f"Average variation within clusters (AVC): {avc:.4f}")
    print(f"Total distance between centers (TD): {td:.4f}")
    
    return pms, ad, avc, td
