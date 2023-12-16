import sys
sys.path.append("RNNsearch")
import os
from train import *
from RNNsearch.RNNsearch import *
from RNNsearch.Encoder_search import *
from RNNsearch.Decoder_search import *
from RNNsearch.Allignement import *
from datasets import load_from_disk
import torch
import torch.nn as nn
from dataloader import *

fr_vocab_path = '../../new30k_fr.txt'  # Path to the French vocab file
with open(fr_vocab_path, 'r') as file:
    fr_words = [line.strip() for line in file]

# Cleaning words and creating the dictionary
word_to_id_fr = {word: i for i, word in enumerate(fr_words)}

eng_vocab_path= '../../new30k_eng.txt'  # Path to the English vocab file
# Open the file and read lines into a list
with open(eng_vocab_path, 'r') as file:
    eng_words = [line.strip() for line in file]

word_to_id_eng = { word.strip() : i for i, word in enumerate(eng_words)}


test_val_data = load_from_disk('/home/linda/dataset_50/test')
test_data, val_data = test_val_data.train_test_split(test_size=0.5).values()
val_dataset = Seq2seqData(val_data, word_to_id_eng, word_to_id_fr)
train_data = load_from_disk('/home/linda/dataset_50/train')
train_dataset = Seq2seqData(train_data, word_to_id_eng, word_to_id_fr)

# Define your model
#hyperparameters
batch_size = 80       # Number of sequences in a mini-batch
vocab_size = 30000     # Size of the input vocabulary
hidden_size = 1000     # Number of features in the hidden state
embedding_dim = 620   # Word embedding dimension
maxout_units= 500     # Number of units in the maxout layer
allign_dim=50        # Number of features in the allignment model Tx


device = "cpu"

encoder = Encoder(vocab_size, hidden_size, embedding_dim, device=device)
decoder = Decoder(vocab_size, hidden_size, embedding_dim, maxout_units, device=device)
model = RNNsearch(encoder, decoder, device=device).to(device)
#file_path = '~/projet-mla/src/Baseline/saves/Search_50/best_model.pth'
file_path='saves/Search_50_last_version/best_model.pth'

file_path = os.path.expanduser(file_path)

state_dict = torch.load(file_path, map_location=device)

model.load_state_dict(state_dict["model_state_dict"])

# Create reverse lookup dictionary for French vocabulary
trg_vocab_reverse_fr = {i: word for word, i in word_to_id_fr.items()}
trg_vocab_reverse_eng = {i: word for word, i in word_to_id_eng.items()}

source=train_dataset[100][0][train_dataset[100][0] != 0]
results,alignement=model(source.unsqueeze(0))

pred_idx=[]

for i in range (results.shape[1]) :
    pred_idx.append(results[:, i, :].argmax().item())
    
output_sequence = [trg_vocab_reverse_fr.get(idx) for idx in pred_idx]
input_sequence= [trg_vocab_reverse_eng.get(int(idx) )for idx in (source)]
print("input_sequence :" , input_sequence)
print( "output_sequence :" ,output_sequence)

breakpoint()

def beam_search():
    pass