from torch.utils.data import Dataset
from pathlib import Path
import torch

class DrowsinessImageSequenceDataset(Dataset):
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.files = sorted(self.root_dir.glob("*.pt"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx])
        video = data["video"]     
        label = data["label"]

        video = video.float()
        label = torch.tensor(label, dtype=torch.long)

        return video, label

