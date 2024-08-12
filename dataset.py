import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

    
class Data(Dataset):
    def __init__(self, 
                 V1,
                 padded_X, 
                 X_lengths,
                 outcomes
                ):
        self.V1 = V1 #df, dense, continuous
        self.X = padded_X # subjects *features * max_timestamp
        self.X_lengths = X_lengths # list of lengths
        self.outcomes = outcomes # four outcomes in prediction task 
        
    def __len__(self):
        return len(self.V1)
    
    def __getitem__(self, idx):
        
        max_sequence_length=max(self.X_lengths)
        mask=np.zeros(max_sequence_length)
        mask[:self.X_lengths[idx]]=1
        
        return {
            'V1': self.V1.iloc[idx].values,
            'X': self.X[idx, :, :],
            'seq_len': self.X_lengths[idx],
            'mask': mask,
            'outcomes': self.outcomes[idx]
        }
