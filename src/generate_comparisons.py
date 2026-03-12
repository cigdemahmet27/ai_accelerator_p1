import sys
import torch
import numpy as np
from autoencoder import Autoencoder, get_dataloaders
import evaluate

batch_size = int(sys.argv[1]) if len(sys.argv) > 1 else 256
results_dir = f'results_bs{batch_size}'
evaluate.set_results_dir(results_dir)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print(f"Results directory: {results_dir}")

_, testloader = get_dataloaders(batch_size=batch_size)

latent_dims = [4, 8, 10, 16, 32, 64]
models_dict = {}

for dim in latent_dims:
    model = Autoencoder(latent_dim=dim)
    model.load_state_dict(torch.load(f'{results_dir}/latent_{dim}/autoencoder_model.pth',
                                     map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    models_dict[dim] = model
    print(f"  Loaded model for dim={dim}")

all_results = []
for dim in latent_dims:
    with open(f'{results_dir}/latent_{dim}/metrics.txt', 'r') as f:
        lines = f.readlines()
    metrics = {}
    for line in lines:
        if 'PMS' in line:
            metrics['pms'] = float(line.split(':')[1].strip().replace('%', ''))
        elif 'AD' in line:
            metrics['ad'] = float(line.split(':')[1].strip())
        elif 'AVC' in line:
            metrics['avc'] = float(line.split(':')[1].strip())
        elif 'TD' in line:
            metrics['td'] = float(line.split(':')[1].strip())

    with open(f'{results_dir}/comparison.txt', 'r') as f:
        comp_lines = f.readlines()

    train_loss = float(comp_lines[2].split()[3 + (latent_dims.index(dim))])
    test_loss = float(comp_lines[3].split()[3 + (latent_dims.index(dim))])

    all_results.append({
        'latent_dim': dim,
        'final_train_loss': train_loss,
        'final_test_loss': test_loss,
        **metrics,
    })
    print(f"  Loaded metrics for dim={dim}: PMS={metrics['pms']:.2f}%")

print("\n1/3 Generating metrics comparison bar chart...")
evaluate.plot_metrics_comparison(all_results)
print(f"  Saved: {results_dir}/metrics_comparison.png")

print("2/3 Generating reconstruction comparison grid...")
evaluate.plot_reconstruction_comparison(models_dict, testloader, device)
print(f"  Saved: {results_dir}/reconstruction_comparison.png")

print("3/3 Generating t-SNE comparison (this may take several minutes)...")
evaluate.plot_tsne_comparison(models_dict, testloader, device)
print(f"  Saved: {results_dir}/tsne_comparison.png")

print(f"\nDone! All comparison plots saved in {results_dir}/")
