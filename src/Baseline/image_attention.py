import sys
sys.path.append("RNNsearch")
import os

from RNNsearch.RNNsearch import * 
from RNNsearch.Encoder_search import * 
from RNNsearch.Decoder_search import * 
from RNNsearch.Allignement import * 


from datasets import load_from_disk 
import torch
import torch.nn as nn

import matplotlib.pyplot as plt

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
        
        return torch.tensor(input_sequence, dtype=torch.long), torch.tensor(output_sequence, dtype=torch.long)

class Seq2seqDataEval(Dataset):
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
        size = pair['translation']['size']

        return torch.tensor(input_sequence, dtype=torch.long), torch.tensor(output_sequence, dtype=torch.long), torch.tensor(size, dtype=torch.long)

path = os.getcwd().replace("projet-mla/src","")
print(path)
# Load French vocabulary
fr_vocab_path = path+'projet-mla/new30k_fr.txt'  # Path to the French vocab file
with open(fr_vocab_path, 'r') as file:
    fr_words = [line.strip() for line in file]

# Cleaning words and creating the French dictionary
word_to_id_fr = {word: i for i, word in enumerate(fr_words)}

# Load English vocabulary
eng_vocab_path= path+'projet-mla/new30k_eng.txt'  # Path to the English vocab file
with open(eng_vocab_path, 'r') as file:
    eng_words = [line.strip() for line in file]

# Creating the English dictionary
word_to_id_eng = { word.strip() : i for i, word in enumerate(eng_words)}

# Load and split the test dataset
test_val_data = load_from_disk(path+'dataset_30/test')
test_data, val_data = test_val_data.train_test_split(test_size=0.5).values()

# Prepare test dataset
test_data = Seq2seqData(test_data, word_to_id_eng, word_to_id_fr)

# Load training dataset
train_data = load_from_disk(path+'dataset_50/train')
train_dataset = Seq2seqData(train_data, word_to_id_eng, word_to_id_fr)

# Define model hyperparameters
batch_size = 80       # Number of sequences in a mini-batch
vocab_size = 30000    # Size of the input vocabulary
hidden_size = 1000    # Number of features in the hidden state
embedding_dim = 620   # Word embedding dimension
maxout_units= 500     # Number of units in the maxout layer
allign_dim=50         # Number of features in the alignment model

device = "cpu"

# Initialize encoder and decoder
encoder = Encoder(vocab_size, hidden_size, embedding_dim,device)
decoder = Decoder(vocab_size, hidden_size, embedding_dim, maxout_units,device)
model = RNNsearch(encoder, decoder, device=device).to(device)


# Load model state
#file_path = 'saves/Search_50_last_version/best_model.pth'
file_path="~/projet-mla/src/Models/Search_50_last_version/best_model.pth"
file_path = os.path.expanduser(file_path)
state_dict = torch.load(file_path, map_location=device)
model.load_state_dict(state_dict["model_state_dict"])

# Create reverse lookup dictionaries for French and English vocabularies
trg_vocab_reverse_fr = {i: word for word, i in word_to_id_fr.items()}
trg_vocab_reverse_eng = {i: word for word, i in word_to_id_eng.items()}

# Perform inference on an example from the training dataset
source = test_data[1000][0][test_data[1000][0] != 0]

results , alignement = model(source.unsqueeze(0))

# Decode the predicted sequence
pred_idx = []
for i in range(results.shape[1]):
    pred_idx.append(results[:, i, :].argmax().item())
output_sequence = [trg_vocab_reverse_fr.get(idx) for idx in pred_idx]
input_sequence = [trg_vocab_reverse_eng.get(int(idx)) for idx in source]

print("Input sequence:", input_sequence)
print("Output sequence:", output_sequence)

# Function to display attention between input and output sequences
def display_attention(sentence, translation, attention, file_path):
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111)

    # Squeeze the attention to make it 2D and detach from torch
    attention = attention.squeeze(0).cpu().detach().numpy()

    cax = ax.matshow(attention, cmap='bone')

    ax.tick_params(labelsize=15)

    x_ticks_labels = sentence
    y_ticks_labels = translation

    ax.set_xticks(range(len(x_ticks_labels)))
    ax.set_yticks(range(len(y_ticks_labels)))

    ax.set_xticklabels(x_ticks_labels, rotation=45)
    ax.set_yticklabels(y_ticks_labels)

    plt.savefig(file_path)
    plt.close()
    
output_folder = 'Baseline/figures_attention'
file_name = 'attention_plot_example4.png'
file_path = os.path.join(output_folder, file_name)

display_attention(input_sequence, output_sequence, alignement, file_path)
