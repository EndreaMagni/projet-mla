import torch
from torch.utils.data import Dataset 
from datasets import load_from_disk

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
        return input_sequence, output_sequence

fr_vocab_path = '../../30k_fr.txt'  # Path to the French vocab file
with open(fr_vocab_path, 'r') as file:
    fr_words = [line.strip() for line in file]

# Cleaning words and creating the dictionary
word_to_id_eng = {word: i for i, word in enumerate(fr_words)}

eng_vocab_path= '../../30k_eng.txt'  # Path to the English vocab file
# Open the file and read lines into a list
with open(eng_vocab_path, 'r') as file:
    eng_words = [line.strip() for line in file]

word_to_id_fr = { word.strip() : i for i, word in enumerate(eng_words)}

train_data = load_from_disk('/home/linda/dataset_50/train')
test_val_data=load_from_disk('/home/linda/dataset_50/test')
test_data, val_data = test_val_data.train_test_split(test_size=0.5).values()

# Create a TranslationDataset instance for training and validation
train_dataset = Seq2seqData(train_data, word_to_id_eng, word_to_id_fr)
val_dataset = Seq2seqData(val_data, word_to_id_eng, word_to_id_fr)