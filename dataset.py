import numpy as np
import torch
from torch.utils.data import Dataset

class DrowsinessSequenceDataset(Dataset):
    def __init__(self, npz_path: str):
        data = np.load(npz_path)
        self.X = data["X"]
        self.y = data["y"]
        assert len(self.X) == len(self.y)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx].astype(np.float32)
        y = int(self.y[idx])
        # retorna tensores
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)

