import torch
from torch.utils.data import Dataset 
from datasets import load_from_disk

class Data (Dataset):
    def __init__(self, path_to_data: str = "projet-mla/dataset_50" , test_val_split_ratio : float = 0.5) -> None:
       
        self.data = load_from_disk(path_to_data)
        self.train_data = self.data['train']
        self.test_val_data = self.data['test']
        self.test_data, self.val_data = self.test_val_data.train_test_split(test_size=test_val_split_ratio).values()
                
    def __len__(self):
        ...

    def __getitem__(self, idx):
       ...



