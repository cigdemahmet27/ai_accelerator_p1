import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from autoencoder import Autoencoder, get_dataloaders
import evaluate

def train_autoencoder(model, trainloader, testloader, device, num_epochs=20, learning_rate=1e-3):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    
    model.to(device)
    
    print(f"Training on {device}...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, _ in trainloader:
            images = images.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, images) # Compare reconstructed to original image
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            
        epoch_loss = running_loss / len(trainloader.dataset)
        train_losses.append(epoch_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
        
    evaluate.check_results_dir()
    plt.figure()
    plt.plot(range(1, num_epochs+1), train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.savefig('results/loss_curve.png')
    plt.close()
    
    return model

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    trainloader, testloader = get_dataloaders(batch_size=256)
    
    model = Autoencoder(latent_dim=16)
    
    # Train the model
    model = train_autoencoder(model, trainloader, testloader, device, num_epochs=20)
    
    # Save the model
    evaluate.check_results_dir()
    torch.save(model.state_dict(), 'results/autoencoder_model.pth')
    
    print("\nVisualizing original vs reconstructed images...")
    evaluate.plot_reconstructions(model, testloader, device)
    
    print("Extracting latent representations...")
    latent_reps, true_labels = evaluate.extract_latent_representations(model, testloader, device)
    
    print("Plotting Latent Space (PCA)...")
    evaluate.plot_latent_space(latent_reps, true_labels, method='PCA')
    
    print("Plotting Latent Space (t-SNE)... (This might take a minute)")
    evaluate.plot_latent_space(latent_reps, true_labels, method='t-SNE')
    
    print("\nApplying K-Means clustering and calculating metrics...")
    evaluate.cluster_and_evaluate(latent_reps, true_labels, num_clusters=10)
    print("\nEvaluation completed. Results saved in 'results/' directory.")

if __name__ == "__main__":
    main()
