__author__ = 'Fabio'

import torch
from torch.utils.data import Dataset
import pandas as pd

class BostonCrimeDataset(Dataset):
    def __init__(self, data: pd.DataFrame, len_sequence: int = 14, len_target: int = 1):
        self.data = data
        self.len_seq = len_sequence
        self.len_target = len_target
        self.len = len(data) - len_sequence - len_target

    def __len__(self):
        return self.len

    def __getitem__(self, i):
        assert i < len(self) and i >= 0
        src = torch.tensor(self.data.iloc[i:i+self.len_seq].values)
        tgt = torch.tensor(self.data.iloc[i+self.len_seq:i+self.len_seq+self.len_target].values)
        return src, tgt


if __name__ == '__main__':
    df = pd.read_csv('./aggregated_crimes.csv', index_col=0)
    df.fillna(0, inplace=True)

    train_data = BostonCrimeDataset(df)
    val_data = BostonCrimeDataset(df)
    test_data = BostonCrimeDataset(df)

