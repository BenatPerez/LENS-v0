import torch
import faiss
import numpy as np
from torch.utils.data import Sampler

class FaissSampler(Sampler):
    def __init__(self, dataset, model, batch_size = 4096, k_neighbors = 63):
        self.dataset = dataset
        self.model = model
        self.batch_size = batch_size
        self.k_neighbors = k_neighbors

        # Calculate how many anchors fit in a single batch.
        # The +1 accounts for the anchor itself alongside its k_neighbors.
        self.num_anchors = batch_size // (k_neighbors + 1)
        self.num_batches = len(dataset) // batch_size

    def __iter__(self):
        # Retrieve the full dataset features to compute current embeddings
        features = self.dataset.features
        
        self.model.eval()
        with torch.no_grad():
            embeddings = self.model(features)
        self.model.train()

        vectors_np = embeddings.cpu().numpy().astype(np.float32)
        d = vectors_np.shape[1]
        
        # We use Inner Product (IP) for Faiss. Since our embeddings are L2 normalized 
        index = faiss.IndexFlatIP(d)
        index.add(vectors_np)

        # Step 1: Pick a random starting anchor for each batch
        first_anchor_idx = np.random.choice(len(self.dataset), self.num_batches, replace = False)
        first_anchor_vectors = vectors_np[first_anchor_idx]

        # Step 2: Find the closest anchors to that first random anchor.
        # Creating these groups of close anchors ensures that they act as both 
        # anchors and neighbors for each other. This is crucial for the Brier Score to converge globally.
        _, close_anchors_idx = index.search(first_anchor_vectors, self.num_anchors)
        anchors_idx = close_anchors_idx.flatten()
        anchor_vectors = vectors_np[anchors_idx]

        # Step 3: For every anchor in the group, find its k nearest neighbors
        _, neighbors_idx = index.search(anchor_vectors, self.k_neighbors + 1)

        # Yield batches sequentially
        for i in range(self.num_batches):
            start = i * self.num_anchors
            end = start + self.num_anchors
            
            # Flatten the (num_anchors, k_neighbors + 1) matrix into a 1D list for the DataLoader
            batch_indices = neighbors_idx[start:end].flatten().tolist()
            yield batch_indices

    def __len__(self):
        return self.num_batches