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


fr_vocab_path = '../../30k_fr.txt'  # Path to the French vocab file
with open(fr_vocab_path, 'r') as file:
    fr_words = [line.strip() for line in file]

# Cleaning words and creating the dictionary
word_dict_fr = {word: i for i, word in enumerate(fr_words)}

eng_vocab_path= '../../30k_eng.txt'  # Path to the English vocab file
# Open the file and read lines into a list
with open(eng_vocab_path, 'r') as file:
    eng_words = [line.strip() for line in file]

word_dict_eng = { word.strip() : i for i, word in enumerate(eng_words)}


#hyperparameters
batch_size = 80       # Number of sequences in a mini-batch
vocab_size = 30000     # Size of the input vocabulary
hidden_size = 1000     # Number of features in the hidden state
embedding_dim = 620   # Word embedding dimension
maxout_units= 500     # Number of units in the maxout layer
allign_dim=50        # Number of features in the allignment model Tx

if torch.cuda.is_available():           device = "cuda"
else:                                   device = "cpu"


encoder=Encoder(vocab_size,hidden_size,embedding_dim ,device=device).to(device)
decoder=Decoder(vocab_size,hidden_size,embedding_dim,maxout_units,device=device).to(device)
model=RNNsearch(encoder,decoder,device).to(device)

train_data = load_from_disk('/home/linda/dataset_50/train')
test_val_data=load_from_disk('/home/linda/dataset_50/test')
#train_data = load_from_disk('..\\dataset_50\\train')
#test_val_data=load_from_disk('..\dataset_50\\test')
test_data, val_data = test_val_data.train_test_split(test_size=0.5).values()

learning_rate=1                            
epochs=2

batches=train(model,train_data,val_data,word_dict_eng,word_dict_fr,batch_size,vocab_size,learning_rate,epochs,print_every=1,device=device)