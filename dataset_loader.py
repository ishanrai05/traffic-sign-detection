from torch.utils.data import Dataset
from PIL import Image
import torch

class GTSRB(Dataset):
    def __init__(self, Cells, labels, transform=None):
        self.X = Cells
        self.y = labels
        
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        # Load data and get label
        X = self.X[index]
        X = Image.fromarray(X)
        y = torch.tensor(int(self.y[index]))
        if self.transform:
            X = self.transform(X)

        return X, y