import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNetBlock(nn.Module):
    def __init__(self, dim, dropout = 0.1):
        super().__init__()

        # LayerNorm is used to prevent losing information across layers.
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return x + self.block(x)

class Embeddings(nn.Module):
    def __init__(self, feature_dims, entity_dim = 32, latent_dim = 1024, dropout = 0.1, feature_dropout = 0.05, hidden_dim = 2048, num_blocks = 5):
        super(Embeddings, self).__init__()

        # Feature dropout with a low rate forces the network to not rely solely on a single variable.
        self.tabular_masking = nn.Dropout(feature_dropout)
        
        # Converts each categorical variable into a vector/matrix representation.
        self.entity_list = nn.ModuleList([nn.Embedding(num_embeddings = dims, embedding_dim = entity_dim) for dims in feature_dims])
        concat_dim = len(feature_dims) * entity_dim
        
        # The gate network learns which facets of the variables are important for each case.
        # Hardsigmoid is used instead of standard Sigmoid to make the gating more aggressive (not stuck around 0.4-0.6).
        self.gate = nn.Sequential(
            nn.Linear(concat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, concat_dim),
            nn.Hardsigmoid()
        )

        # Initialize with a large bias (2.0) so all gates start fully open and close gradually as learning progresses.
        nn.init.constant_(self.gate[6].bias, 2.0)
        nn.init.xavier_uniform_(self.gate[6].weight, gain = 0.1)

        mlp_layers = []
        mlp_layers.extend([
            nn.Linear(concat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        ])

        # A ResNet structure handles the large size and prevents vanishing gradients.
        for _ in range(num_blocks):
            mlp_layers.append(ResNetBlock(hidden_dim, dropout = dropout))
        
        mlp_layers.append(nn.Linear(hidden_dim, latent_dim))
        self.final_mlp = nn.Sequential(*mlp_layers)

    def forward(self, x):
        entity_embeddings = []
        for i, network in enumerate(self.entity_list):
            variable = x[:, i]
            vector = network(variable)
            entity_embeddings.append(vector)
        
        concat_vector = torch.cat(entity_embeddings, dim = 1)

        # Applying gates over vectors instead of single variables allows shutting down only specific "facets" of a variable.
        mask = self.gate(concat_vector)
        filtered_vector = concat_vector * mask
        filtered_vector = self.tabular_masking(filtered_vector)

        embedding = self.final_mlp(filtered_vector)
        
        # Placing embeddings on a unit hypersphere ensures only the angles dictate their semantic differences, not distances.
        final_embedding = F.normalize(embedding, p = 2, dim = 1)

        return final_embedding