from torch.utils.data import Dataset
from typing import Any

class BaseDataset(Dataset):
    """For this use case, one record of data will have: items, ages, prices of each customer
        Examples:
        # this below means at the index 0, we return the sequences of items/ages/prices of one customer
        >>> items, ages, prices = data[0] 
    """
    def __init__(self, data:list):
        self.data = data

    def __len__(self)->int:
        return len(self.data)

    def __getitem__(self, idx:int) -> Any:
        """Return one record"""
        return self.data[idx]