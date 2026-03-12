import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from autoencoder import Autoencoder, get_dataloaders
import evaluate


def compute_loss(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            outputs = model(images)
            loss = criterion(outputs, images)
            total_loss += loss.item() * images.size(0)
    return total_loss / len(dataloader.dataset)


def train_autoencoder(model, trainloader, testloader, device,
                      num_epochs=50, learning_rate=1e-3):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    test_losses = []

    model.to(device)

    print(f"Training on {device}...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, _ in trainloader:
            images = images.to(device)
            outputs = model(images)
            loss = criterion(outputs, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_train_loss = running_loss / len(trainloader.dataset)
        epoch_test_loss = compute_loss(model, testloader, criterion, device)

        train_losses.append(epoch_train_loss)
        test_losses.append(epoch_test_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}]  "
              f"Train Loss: {epoch_train_loss:.6f}  "
              f"Test Loss: {epoch_test_loss:.6f}")

    return model, train_losses, test_losses


def plot_loss_curves(train_losses, test_losses, latent_dim):
    evaluate.check_results_dir(latent_dim)
    plt.figure(figsize=(8, 5))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, test_losses, label='Test Loss', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title(f'Training & Test Loss Curve (latent_dim={latent_dim})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{evaluate.RESULTS_DIR}/latent_{latent_dim}/loss_curve.png', dpi=150)
    plt.close()


def run_experiment(latent_dim, trainloader, testloader, device, num_epochs=50):
    print(f"\n{'='*60}")
    print(f"  EXPERIMENT: latent_dim = {latent_dim}")
    print(f"{'='*60}\n")

    model = Autoencoder(latent_dim=latent_dim)

    model, train_losses, test_losses = train_autoencoder(
        model, trainloader, testloader, device, num_epochs=num_epochs
    )

    evaluate.check_results_dir(latent_dim)
    torch.save(model.state_dict(),
               f'{evaluate.RESULTS_DIR}/latent_{latent_dim}/autoencoder_model.pth')

    plot_loss_curves(train_losses, test_losses, latent_dim)

    print("\nVisualizing reconstructions...")
    evaluate.plot_reconstructions(model, testloader, device, latent_dim)

    print("Extracting latent representations...")
    latent_reps, true_labels = evaluate.extract_latent_representations(
        model, testloader, device
    )

    print("Plotting Latent Space (PCA)...")
    evaluate.plot_latent_space(latent_reps, true_labels, latent_dim, method='PCA')

    print("Plotting Latent Space (t-SNE)... (this may take a minute)")
    evaluate.plot_latent_space(latent_reps, true_labels, latent_dim, method='t-SNE')

    print("\nClustering and evaluating...")
    metrics = evaluate.cluster_and_evaluate(latent_reps, true_labels, latent_dim,
                                            num_clusters=10)

    return {
        'latent_dim': latent_dim,
        'final_train_loss': train_losses[-1],
        'final_test_loss': test_losses[-1],
        'train_losses': train_losses,
        'test_losses': test_losses,
        'pms': metrics[0],
        'ad': metrics[1],
        'avc': metrics[2],
        'td': metrics[3],
        'model': model,
    }


def print_comparison(all_results):
    print(f"\n{'='*60}")
    print("  COMPARISON ACROSS LATENT DIMENSIONS")
    print(f"{'='*60}\n")

    header = f"{'Metric':<30} "
    for r in all_results:
        header += f"{'dim=' + str(r['latent_dim']):>12} "
    print(header)
    print("-" * len(header))

    for key, label in [
        ('final_train_loss', 'Final Train Loss'),
        ('final_test_loss', 'Final Test Loss'),
        ('pms', 'PMS (%)'),
        ('ad', 'AD'),
        ('avc', 'AVC'),
        ('td', 'TD'),
    ]:
        row = f"{label:<30} "
        for r in all_results:
            row += f"{r[key]:>12.4f} "
        print(row)

    if not os.path.exists(evaluate.RESULTS_DIR):
        os.makedirs(evaluate.RESULTS_DIR)
    with open(f'{evaluate.RESULTS_DIR}/comparison.txt', 'w') as f:
        f.write(header + '\n')
        f.write("-" * len(header) + '\n')
        for key, label in [
            ('final_train_loss', 'Final Train Loss'),
            ('final_test_loss', 'Final Test Loss'),
            ('pms', 'PMS (%)'),
            ('ad', 'AD'),
            ('avc', 'AVC'),
            ('td', 'TD'),
        ]:
            row = f"{label:<30} "
            for r in all_results:
                row += f"{r[key]:>12.4f} "
            f.write(row + '\n')


def main(batch_size=256):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    evaluate.set_results_dir(f'results_bs{batch_size}')
    print(f"Batch size: {batch_size}")
    print(f"Results directory: {evaluate.RESULTS_DIR}")

    trainloader, testloader = get_dataloaders(batch_size=batch_size)

    latent_dims = [4, 8, 10, 16, 32, 64]
    all_results = []

    for ld in latent_dims:
        result = run_experiment(ld, trainloader, testloader, device, num_epochs=50)
        all_results.append(result)

    print_comparison(all_results)

    print("\nGenerating comparison plots...")

    all_loss_data = [{'latent_dim': r['latent_dim'], 'test_losses': r['test_losses']}
                     for r in all_results]
    evaluate.plot_combined_loss_curves(all_loss_data)
    print("  Combined loss curves saved.")

    evaluate.plot_metrics_comparison(all_results)
    print("  Metrics comparison bar chart saved.")

    models_dict = {r['latent_dim']: r['model'] for r in all_results}
    evaluate.plot_reconstruction_comparison(models_dict, testloader, device)
    print("  Reconstruction comparison grid saved.")

    print("  Generating t-SNE comparison (this may take several minutes)...")
    evaluate.plot_tsne_comparison(models_dict, testloader, device)
    print("  t-SNE comparison saved.")

    print(f"\nAll experiments completed. Results saved in '{evaluate.RESULTS_DIR}/' directory.")


if __name__ == "__main__":
    import sys
    bs = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    main(batch_size=bs)
