"""
---------------------------------
BY : Haoyu Tang
Github : Jerry_Haoyu 
---------------------------------
"""
import torch 
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

class pendulumPINNDataSet(Dataset):
    """
        Theta time series for pendulumn with varying arm length dataset
        Sparsely choose num_data
    """
    def __init__(self, npy_data, num_data=128):
        """
        Args:
            npy_data(str): path to npy file 
        """
        self.len = num_data
        self.all_data = np.load(npy_data)
        total_len = self.all_data.shape[0]
        self.time_series = self.all_data[np.random.choice(np.arange(total_len), size=num_data), :]
        
    def __len__(self):
        return self.len 
    
    def __getitem__(self, idx):
        t = torch.tensor(self.time_series[idx, 0], requires_grad=True, dtype=torch.float32)
        x1 = torch.tensor(self.time_series[idx, 1], requires_grad=True, dtype=torch.float32)
        x2 = torch.tensor(self.time_series[idx, 2], requires_grad=True, dtype=torch.float32)
        L = torch.tensor(self.time_series[idx, 3], requires_grad=True, dtype=torch.float32)
        return t, x1, x2, L

        