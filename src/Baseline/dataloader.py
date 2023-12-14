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
        output_onehot = F.one_hot(torch.tensor(output_sequence), num_classes=len(self.word_to_id_fr)).float()
        return torch.tensor(input_sequence, dtype=torch.long), torch.tensor(output_sequence, dtype=torch.long),output_onehot

fr_vocab_path = '../../30k_fr.txt'  # Path to the French vocab file
with open(fr_vocab_path, 'r') as file:
    fr_words = [line.strip() for line in file]

# Cleaning words and creating the dictionary
word_to_id_fr = {word: i for i, word in enumerate(fr_words)}

eng_vocab_path= '../../30k_eng.txt'  # Path to the English vocab file
with open(eng_vocab_path, 'r') as file:
    eng_words = [line.strip() for line in file]

word_to_id_eng = { word.strip() : i for i, word in enumerate(eng_words)}

train_data = load_from_disk('/home/linda/dataset_50/train')
dataset = Seq2seqData(train_data, word_to_id_eng, word_to_id_fr)

# Create a DataLoader
batch_size = 32  # Adjust as needed
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Iterate through the DataLoader to test loading
for batch in data_loader:
    input_sequence, output_sequence, output_onehot = batch

    # Print the shape of the loaded data to check if it's as expected
    print(f"Input Sequence Shape: {input_sequence.shape}")
    print(f"Output Sequence Shape: {output_sequence.shape}")
    print(f"Output One-Hot Shape: {output_onehot.shape}")

    # You can also print some sample data if needed
    print("Sample Input Sequence:", input_sequence[0])
    print("Sample Output Sequence:", output_sequence[0])
    print("Sample Output One-Hot:", output_onehot[0])

    # Break the loop after testing one batch if needed
    break



