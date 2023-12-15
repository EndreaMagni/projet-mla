import sys
sys.path.append("RNNsearch")

from train import *
from RNNsearch.RNNsearch import *
from RNNsearch.Encoder_search import *
from RNNsearch.Decoder_search import *
from RNNsearch.Allignement import *
from datasets import load_from_disk
import torch
import torch.nn as nn
from train import *
from dataloader import *


fr_vocab_path = '../../30k_fr.txt'  # Path to the French vocab file
with open(fr_vocab_path, 'r') as file:
    fr_words = [line.strip() for line in file]

# Cleaning words and creating the dictionary
word_to_id_fr = {word: i for i, word in enumerate(fr_words)}

eng_vocab_path= '../../30k_eng.txt'  # Path to the English vocab file
# Open the file and read lines into a list
with open(eng_vocab_path, 'r') as file:
    eng_words = [line.strip() for line in file]

word_to_id_eng = { word.strip() : i for i, word in enumerate(eng_words)}


#hyperparameters
batch_size = 80       # Number of sequences in a mini-batch
vocab_size = 30000     # Size of the input vocabulary
hidden_size = 1000     # Number of features in the hidden state
embedding_dim = 620   # Word embedding dimension
maxout_units= 500     # Number of units in the maxout layer
allign_dim=50        # Number of features in the allignment model Tx

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

encoder = Encoder(vocab_size, hidden_size, embedding_dim, device=device)
decoder = Decoder(vocab_size, hidden_size, embedding_dim, maxout_units, device=device)
model = RNNsearch(encoder, decoder, device=device).to(device)

train_data = load_from_disk('/home/linda/dataset_50/train')
test_val_data = load_from_disk('/home/linda/dataset_50/test')

#test_val_data = load_from_disk('/home/linda/projet-mla/mini_dataset')
#train_data = load_from_disk('/home/linda/projet-mla/mini_dataset')

test_data, val_data = test_val_data.train_test_split(test_size=0.5).values()

# Create a TranslationDataset instance for training and validation
train_dataset = Seq2seqData(train_data, word_to_id_eng, word_to_id_fr)
val_dataset = Seq2seqData(val_data, word_to_id_eng, word_to_id_fr)

# Move the data loaders to the same device as the model
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=device=="cuda")
val_data_loader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=device=="cuda")

learning_rate = 0.1
epochs = 100

train(model, train_data_loader, val_data_loader, vocab_size, learning_rate, epochs, device, print_every=1)

