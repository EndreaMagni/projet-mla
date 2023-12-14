import torch
from torch.utils.data import Dataset,DataLoader 
from datasets import load_from_disk
import torch.nn.functional as F

class Seq2seqData(Dataset):
    def __init__(self, data, word_to_id_eng, word_to_id_fr):
        self.data = data
        self.word_to_id_eng = word_to_id_eng
        self.word_to_id_fr = word_to_id_fr

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pair = self.data[idx]
        input_sequence = [self.word_to_id_eng.get(word, 0) for word in pair['translation']['en']]
        output_sequence = [self.word_to_id_fr.get(word, 0) for word in pair['translation']['fr']]
        output_onehot = F.one_hot(torch.tensor(output_sequence), num_classes=30000).float()
        return torch.tensor(input_sequence, dtype=torch.long), torch.tensor(output_sequence, dtype=torch.long),output_onehot

