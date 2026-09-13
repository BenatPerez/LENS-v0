import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import CallDataset
from faiss_sampler import FaissSampler
from loss import LensLoss
from embeddings import Embeddings
import numpy as np
import faiss
import random
import os
import time
from resnet_ablation import ResNetAblation
from sklearn.metrics import roc_auc_score, brier_score_loss

def set_seed(seed):
    # Ensures reproducibility across runs
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_time(model_name, split_prefix, time_seconds):
    file_path = "Models/training_times.txt"
    minutes = int(time_seconds // 60)
    seconds = int(time_seconds % 60)
    with open(file_path, "a", encoding = "utf-8") as f:
        f.write(f"{model_name: <20} | Segment: {split_prefix: <10} | Time: {minutes}m {seconds}s ({time_seconds:.2f} s)\n")

def evaluate_global_brier(anchor_dataloader, neighbor_dataloader, model, k, temp):
    device = next(model.parameters()).device
    model.eval()
    neighbors_embs_list = []
    neighbors_lbls_list = []

    # Extract embeddings and labels for the reference pool (the neighbors).
    with torch.no_grad():
        for x_v, y_v in neighbor_dataloader:
            x_v = x_v.to(device)
            emb_v = model(x_v)
            neighbors_embs_list.append(emb_v.cpu().numpy())
            neighbors_lbls_list.append(y_v.numpy())
            
    neighbors_embs = np.vstack(neighbors_embs_list).astype(np.float32)
    neighbors_lbls = np.concatenate(neighbors_lbls_list)

    # Check if we are evaluating a set against itself (e.g., Train on Train).
    # If they are the same dataloader, we reuse the embeddings to save computation.
    # If they are different (e.g., Test on Train), we calculate embeddings for the unseen anchors.
    if anchor_dataloader is neighbor_dataloader:
        anchors_embs = neighbors_embs
        anchors_lbls = neighbors_lbls
    else:
        anchors_embs_list = []
        anchors_lbls_list = []
        with torch.no_grad():
            for x_a, y_a in anchor_dataloader:
                x_a = x_a.to(device)
                emb_a = model(x_a)
                anchors_embs_list.append(emb_a.cpu().numpy())
                anchors_lbls_list.append(y_a.numpy())
        
        anchors_embs = np.vstack(anchors_embs_list).astype(np.float32)
        anchors_lbls = np.concatenate(anchors_lbls_list)

    # IndexFlatIP computes the Inner Product, which acts as Cosine Similarity since vectors are normalized.
    index_eval = faiss.IndexFlatIP(neighbors_embs.shape[1])
    index_eval.add(neighbors_embs)
    
    # When evaluating calls from the same pool, the closest match will inevitably be the call itself.
    # We increase the search space by one so we can later discard this self-match.
    k_search = k + 1 if anchor_dataloader is neighbor_dataloader else k
    distances, neighbor_indices = index_eval.search(anchors_embs, k_search)

    if anchor_dataloader is neighbor_dataloader:
        # Discard the first column (the self-match) to prevent the model from cheating by using its own exact outcome.
        calc_distances = torch.tensor(distances[:, 1:])
        calc_idx = neighbor_indices[:, 1:]
    else:
        # For unseen test/validation data, there is no self-match in the reference pool, so we keep all closest neighbors.
        calc_distances = torch.tensor(distances)
        calc_idx = neighbor_indices

    # Empirical probability estimation using the distances to nearest neighbors.
    weights = torch.nn.functional.softmax(calc_distances / temp, dim = 1).numpy()
    outcomes = neighbors_lbls[calc_idx]
    probabilities = np.sum(weights * outcomes, axis = 1)

    brier_score = np.mean((probabilities - anchors_lbls) ** 2)
    
    return brier_score, probabilities, anchors_lbls

def train_ResNet_LENS(train_path, val_path, split_prefix,
                      batch_size, latent_dim, entity_dim,
                      k_neighbors_train, k_eval,
                      temp_train, temp_eval,
                      dropout, feature_dropout, neighbor_dropout,
                      gaussian_noise, learning_rate, weight_decay,
                      max_norm, t, lambda1, hidden_dim, num_blocks,
                      epochs = 50, patience = 50):

    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    df_train = CallDataset(train_path)
    df_val = CallDataset(val_path)

    max_indices = torch.tensor(df_train.feature_dims, dtype = torch.long) - 1
    df_val.features = torch.clamp(df_val.features, min = 0)
    df_val.features = torch.minimum(df_val.features, max_indices)

    train_eval_loader = DataLoader(df_train, batch_size = 2048, shuffle = False)
    val_eval_loader = DataLoader(df_val, batch_size = 2048, shuffle = False)

    print(f"ResNet + LENS {split_prefix} | Starting training ---")
    set_seed(42)

    model = Embeddings(feature_dims = df_train.feature_dims, latent_dim = latent_dim, entity_dim = entity_dim,
                       dropout = dropout, feature_dropout = feature_dropout,
                       hidden_dim = hidden_dim, num_blocks = num_blocks).to(device)

    sampler_train = FaissSampler(df_train, model, k_neighbors = k_neighbors_train, batch_size = batch_size)
    dataloader_train = DataLoader(df_train, batch_sampler = sampler_train)
    
    loss_fn = LensLoss(k_neighbors = k_neighbors_train, temperature = temp_train,
                       neighbor_dropout = neighbor_dropout, gaussian_noise = gaussian_noise, t = t, lambda1 = lambda1).to(device)

    # AdamW allows decoupling L2 regularization (weight_decay) from the rest of the algorithm.
    optimizer = optim.AdamW(model.parameters(), lr = learning_rate, weight_decay = weight_decay)
    
    # Smoothly decreases the learning rate following a cosine curve to ensure stable convergence.
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = epochs)

    best_val_brier = float("inf")
    patience_counter = 0
    save_path = f"Models/ResNet_LENS_weights_{split_prefix}.pth"

    for epoch in range(1, epochs + 1):
        model.train()
        accumulated_similarity = 0.0

        for x, y in dataloader_train:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            embeddings = model(x)
            loss, mean_sim = loss_fn(embeddings, y)
            loss.backward()
            
            # Gradient clipping prevents a badly placed neighbor from violently shifting position vectors.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = max_norm)
            optimizer.step()

            accumulated_similarity += mean_sim.item()

        scheduler.step()
        epoch_similarity = accumulated_similarity / len(dataloader_train)

        brier_train_global, _, _ = evaluate_global_brier(train_eval_loader, train_eval_loader, model, k_eval, temp_eval)
        brier_val_global, _, _ = evaluate_global_brier(val_eval_loader, train_eval_loader, model, k_eval, temp_eval)
        print(f"Epoch {epoch} | TRAIN Brier: {brier_train_global:.4f} | VAL Brier: {brier_val_global:.4f} | Similarity: {epoch_similarity:.4f}")

        if brier_val_global < best_val_brier:
            best_val_brier = brier_val_global
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

    end_time = time.time()
    total_time = end_time - start_time
    save_time("ResNet + LENS", split_prefix, total_time)

def train_ResNet_Ablation(train_path, val_path, split_prefix,
                          batch_size, latent_dim, entity_dim,
                          dropout, feature_dropout,
                          learning_rate, weight_decay,
                          hidden_dim, num_blocks,
                          epochs = 50, patience = 50):

    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    df_train = CallDataset(train_path)
    df_val = CallDataset(val_path)

    max_indices = torch.tensor(df_train.feature_dims, dtype = torch.long) - 1
    df_val.features = torch.clamp(df_val.features, min = 0)
    df_val.features = torch.minimum(df_val.features, max_indices)

    dataloader_train = DataLoader(df_train, batch_size = batch_size, shuffle = True)
    dataloader_val = DataLoader(df_val, batch_size = 1024, shuffle = False)

    print(f"Standard ResNet {split_prefix} | Starting training ---")
    set_seed(42)

    model = ResNetAblation(
        feature_dims = df_train.feature_dims, 
        latent_dim = latent_dim, 
        entity_dim = entity_dim, 
        dropout = dropout, 
        feature_dropout = feature_dropout, 
        hidden_dim = hidden_dim, 
        num_blocks = num_blocks).to(device)

    optimizer = optim.AdamW(model.parameters(), lr = learning_rate, weight_decay = weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = epochs)

    best_val_brier = float("inf")
    patience_counter = 0
    save_path = f"Models/ResNet_Standard_weights_{split_prefix}.pth"

    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in dataloader_train:
            x = x.to(device)
            y = y.to(device).float()
            
            optimizer.zero_grad()
            logits = model(x).view(-1) 
            y = y.view(-1)
            
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        scheduler.step()

        model.eval()
        probabilities_list = []
        labels_list = []
        with torch.no_grad():
            for x, y in dataloader_val:
                x = x.to(device)
                logits = model(x).view(-1)
                probs = torch.sigmoid(logits)
                
                probabilities_list.extend(probs.cpu().numpy())
                labels_list.extend(y.numpy())
        
        val_auc = roc_auc_score(labels_list, probabilities_list)
        val_brier = brier_score_loss(labels_list, probabilities_list)

        print(f"Epoch {epoch} | VAL AUC: {val_auc:.4f} | VAL Brier: {val_brier:.4f}")

        if val_brier < best_val_brier:
            best_val_brier = val_brier
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

    end_time = time.time()
    total_time = end_time - start_time
    save_time("Standard ResNet", split_prefix, total_time)

if __name__ == "__main__":
    os.makedirs("Models", exist_ok = True)

    train_ResNet_Ablation(
        train_path = "data/bank_contact_train.parquet", 
        val_path = "data/bank_contact_val.parquet",
        split_prefix = "contact",
        batch_size = 128, latent_dim = 512, entity_dim = 32,
        dropout = 0.25, feature_dropout = 0.2,
        learning_rate = 1e-4, weight_decay = 0.008, 
        hidden_dim = 512, num_blocks = 2, epochs = 50, patience = 50
        )
    
    train_ResNet_Ablation(
        train_path = "data/bank_nocontact_train.parquet", 
        val_path = "data/bank_nocontact_val.parquet", 
        split_prefix = "nocontact",
        batch_size = 2048, latent_dim = 1024, entity_dim = 16,
        dropout = 0.2, feature_dropout = 0.15,
        learning_rate = 1e-4, weight_decay = 0.0002,
        hidden_dim = 1024, num_blocks = 3, epochs = 50, patience = 50
        )
    
    train_ResNet_LENS(
        train_path = "data/bank_contact_train.parquet", 
        val_path = "data/bank_contact_val.parquet",
        split_prefix = "contact",
        batch_size = 128,
        latent_dim = 512,
        entity_dim = 32,
        k_neighbors_train = 63,
        k_eval = 63,
        temp_train = 0.13,
        temp_eval = 0.06,
        dropout = 0.25,
        feature_dropout = 0.2,
        neighbor_dropout = 0.12,
        gaussian_noise = 0.005,
        learning_rate = 3e-5,
        weight_decay = 0.008,
        max_norm = 1.85,
        t = 11,
        lambda1 = 0.0085,
        hidden_dim = 512,
        num_blocks = 2,
        epochs = 50,
        patience = 50
    )

    train_ResNet_LENS(
        train_path = "data/bank_nocontact_train.parquet", 
        val_path = "data/bank_nocontact_val.parquet", 
        split_prefix = "nocontact",
        batch_size = 2048,
        latent_dim = 1024,
        entity_dim = 16,
        k_neighbors_train = 255,
        k_eval = 63,
        temp_train = 0.075,
        temp_eval = 0.075,
        dropout = 0.2,
        feature_dropout = 0.15,
        neighbor_dropout = 0.3,
        gaussian_noise = 0.003,
        learning_rate = 6e-5,
        weight_decay = 0.0002,
        max_norm = 2.75,
        t = 18,
        lambda1 = 0.009,
        hidden_dim = 1024,
        num_blocks = 3,
        epochs = 50,
        patience = 50
    )