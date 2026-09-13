import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import CallDataset
from embeddings import Embeddings

class ResNetAblation(nn.Module):
    def __init__(self, feature_dims, latent_dim, entity_dim, dropout, feature_dropout, hidden_dim, num_blocks):
        super().__init__()

        # Reuses the exact same backbone (Gated Networks + ResNet) used in the LENS model to ensure a fair comparison.
        self.backbone = Embeddings(
            feature_dims = feature_dims, 
            latent_dim = latent_dim, 
            entity_dim = entity_dim, 
            dropout = dropout, 
            feature_dropout = feature_dropout, 
            hidden_dim = hidden_dim, 
            num_blocks = num_blocks
        )
        
        # Instead of calculating probabilities empirically via neighbors, 
        # this ablation baseline uses a standard linear layer to output a score directly.
        self.classifier = nn.Linear(latent_dim, 1)

    def forward(self, x):
        latent_vector = self.backbone(x) 
        logits = self.classifier(latent_vector)
        return logits

def train_ablation(train_path, test_path, epochs = 40):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    df_train = CallDataset(train_path)
    df_test = CallDataset(test_path)
    
    max_indices = torch.tensor(df_train.feature_dims, dtype = torch.long) - 1
    df_test.features = torch.clamp(df_test.features, min = 0)
    df_test.features = torch.minimum(df_test.features, max_indices)

    # Hardcoded hyperparameters depending on the segment, matching the main setup for a true apples-to-apples comparison.
    if "nocontact" in train_path:
        latent_dim, entity_dim = 1024, 16
        dropout, feature_dropout = 0.2, 0.15
        hidden_dim, num_blocks = 1024, 3
        batch_size_train = 2048
    else:
        latent_dim, entity_dim = 512, 32
        dropout, feature_dropout = 0.25, 0.2
        hidden_dim, num_blocks = 512, 2
        batch_size_train = 128

    dataloader_train = DataLoader(df_train, batch_size = batch_size_train, shuffle = True)
    dataloader_test = DataLoader(df_test, batch_size = 1024, shuffle = False)

    model = ResNetAblation(
        feature_dims = df_train.feature_dims, 
        latent_dim = latent_dim, 
        entity_dim = entity_dim, 
        dropout = dropout, 
        feature_dropout = feature_dropout, 
        hidden_dim = hidden_dim, 
        num_blocks = num_blocks).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr = 1e-4)
    
    # Standard Binary Cross Entropy is used here since we are treating the baseline as a standard classification problem.
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        accumulated_loss = 0.0
        
        for x, y in dataloader_train:
            x = x.to(device)
            y = y.to(device).float()
            
            optimizer.zero_grad()
            
            logits = model(x).view(-1) 
            y = y.view(-1)
            
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
            accumulated_loss += loss.item()

        print(f"Epoch {epoch}/{epochs} | Train Loss: {accumulated_loss / len(dataloader_train):.4f}")
    
    model.eval()
    probabilities_list = []
    labels_list = []
    
    with torch.no_grad():
        for x, y in dataloader_test:
            x = x.to(device)
            logits = model(x).view(-1)
            probs = torch.sigmoid(logits)
            
            probabilities_list.extend(probs.cpu().numpy())
            labels_list.extend(y.numpy())
            
    probabilities = np.array(probabilities_list)
    real_labels = np.array(labels_list)
    
    return probabilities, real_labels