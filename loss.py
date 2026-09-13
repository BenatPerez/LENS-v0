import torch
import torch.nn as nn
import torch.nn.functional as F

class LensLoss(nn.Module):
    def __init__(self, k_neighbors, t = 11, lambda1 = 0.0035, temperature = 0.12, neighbor_dropout = 0.2, gaussian_noise = 0.001):
        super().__init__()
        self.k_neighbors = k_neighbors
        self.t = t
        self.lambda1 = lambda1
        self.temperature = temperature
        self.neighbor_dropout = neighbor_dropout
        self.gaussian_noise = gaussian_noise

    def forward(self, embeddings, y):
        # Adding slight Gaussian noise during training helps generalize without destroying the original embeddings.
        if self.training:
            noise = torch.randn_like(embeddings) * self.gaussian_noise
            embeddings = embeddings + noise
            embeddings = F.normalize(embeddings, p = 2, dim = 1)

        # N is the number of anchors. Reshape embeddings to group each anchor with its k_neighbors.
        N = embeddings.shape[0] // (self.k_neighbors + 1)
        embeddings_group = embeddings.view(N, self.k_neighbors + 1, -1)
        y_group = y.view(N, self.k_neighbors + 1)

        anchors_emb = embeddings_group[:, 0:1, :]
        neighbors_emb = embeddings_group[:, 1:, :]
        anchors_y = y_group[:, 0]
        neighbors_y = y_group[:, 1:]

        # Calculate cosine similarity (inner product of normalized vectors).
        similarity = torch.sum(anchors_emb * neighbors_emb, dim = 2)
        mean_similarity = torch.mean(similarity)
        
        # Neighbor dropout randomly removes neighbors so the network organizes the global space instead of overfitting to specific cases.
        if self.training and self.neighbor_dropout > 0.0:
            dropout_mask = torch.rand_like(similarity) < self.neighbor_dropout
            # Setting similarity to -1e9 ensures the softmax weight becomes 0 for dropped neighbors.
            similarity = torch.where(dropout_mask, torch.tensor(-1e9).to(similarity.device), similarity)
    
        # Softmax normalizes the weights, scaled by the temperature parameter which defines the neighborhood scale.
        weights = torch.softmax(similarity / self.temperature, dim = 1)
        
        # Calculate empirical probability based on neighbors' outcomes.
        p = torch.sum(weights * neighbors_y, dim = 1)
        
        # Main Loss: Brier Score calculated over the N anchors.
        main_loss = torch.mean((p - anchors_y)**2)

        # Geometry Loss: An exponential repulsion term (similar to nuclear forces) applied to all M calls in the batch.
        # It only repels calls that get too close to stabilize predictions.
        M = embeddings.shape[0]
        all_squared_distances = torch.cdist(embeddings, embeddings) ** 2
        repulsion = torch.exp(-self.t * all_squared_distances)
        diagonal_mask = torch.eye(M, device = embeddings.device).bool()
        repulsion = repulsion.masked_fill(diagonal_mask, 0.0)
        
        geometry_loss = self.lambda1 * (torch.sum(repulsion) / (M * (M - 1)))

        total_loss = main_loss + geometry_loss

        return total_loss, mean_similarity