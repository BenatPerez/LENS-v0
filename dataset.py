import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class CallDataset(Dataset):
    def __init__(self, file_path):
        # Load the preprocessed parquet file
        df = pd.read_parquet(file_path)
        
        # Separate features (X) and target outcome (Y)
        X = df.drop(columns = ["y"]).values
        Y = df["y"].values

        # Features are cast to long integers to be used in the Entity Embedding layers
        self.features = torch.tensor(X, dtype = torch.long)
        self.labels = torch.tensor(Y, dtype = torch.float32)
        
        self.feature_dims = []
        
        # Calculate the number of unique categories per feature to define embedding sizes later
        for col in df.drop(columns = ["y"]).columns:
            num_categories = df[col].nunique()
            self.feature_dims.append(num_categories)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]