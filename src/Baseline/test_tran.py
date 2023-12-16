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
from train import *
from dataloader import *


def translate_sentence(sentence, src_vocab, trg_vocab, model, trg_vocab_reverse, device, max_len=50):
    model.eval()

    # Tokenization and conversion to indices
    tokens = sentence.split()
    src_indexes = [src_vocab.get(token, src_vocab['<unk>']) for token in tokens]
    src_indexes = [src_vocab["<sos>"]] + src_indexes + [src_vocab["<eos>"]]

    # Convert to tensor
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)

    # Process source sentence with the encoder
    with torch.no_grad():
        results, _ = model(src_tensor)

    # Generate translation
    pred_idx = [results[:, i, :].argmax().item() for i in range(results.shape[1])]

    # Convert indices back to words
    trg_tokens = [trg_vocab_reverse.get(idx, '<unk>') for idx in pred_idx]

    # Filter out <sos> and <eos> tokens
    trg_tokens = [token for token in trg_tokens if token not in [trg_vocab["<sos>"], trg_vocab["<eos>"]]]

    return trg_tokens

batch_size = 80   

device = "cpu"

# Load the French vocabulary and create a dictionary mapping words to indices
fr_vocab_path = '../../new30k_fr.txt'
with open(fr_vocab_path, 'r') as file:
    fr_words = [line.strip() for line in file]
word_to_id_fr = {word: i for i, word in enumerate(fr_words)}

# Load the English vocabulary and create a dictionary mapping words to indices
eng_vocab_path = '../../new30k_eng.txt'
with open(eng_vocab_path, 'r') as file:
    eng_words = [line.strip() for line in file]
word_to_id_eng = {word.strip(): i for i, word in enumerate(eng_words)}

# Check and add special tokens (start of sentence, end of sentence) to the dictionaries
for special_token in ['<sos>', '<eos>', '<unk>']:
    if special_token not in word_to_id_fr:
        word_to_id_fr[special_token] = len(word_to_id_fr)-3
    if special_token not in word_to_id_eng:
        word_to_id_eng[special_token] = len(word_to_id_eng)-2

# Load the datasets from disk
train_data = load_from_disk('/home/linda/dataset_50/train')
test_val_data = load_from_disk('/home/linda/dataset_50/test')

# Split the test/validation dataset into separate test and validation sets
test_data, val_data = test_val_data.train_test_split(test_size=0.5).values()

# Create instances of the Seq2seqData class for training and validation
train_dataset = Seq2seqData(train_data, word_to_id_eng, word_to_id_fr)
val_dataset = Seq2seqData(val_data, word_to_id_eng, word_to_id_fr)

# Create data loaders for the training and validation datasets
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=device=="cuda")
val_data_loader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=device=="cuda")

seq2seq_dataset = Seq2seqData(train_data, word_to_id_eng, word_to_id_fr)
input_sequence_tensor, output_sequence_tensor = seq2seq_dataset[0]

def ids_to_words(ids, id_to_word_map):
    return ' '.join([id_to_word_map.get(id, '') for id in ids])

id_to_word_eng = {i: word for word, i in word_to_id_eng.items()}
id_to_word_fr = {i: word for word, i in word_to_id_fr.items()}

input_sequence = ids_to_words(input_sequence_tensor.tolist(), id_to_word_eng)
output_sequence = ids_to_words(output_sequence_tensor.tolist(), id_to_word_fr)

# print("english:", input_sequence)
# print("france:", output_sequence)

sentence_to_translate = "we also need time for the relations between the ombudsman and the other institutions to mature in order to establish the desired balances and resolve the obvious tensions which may occur until things are working properly."

id_to_word_fr = {i: word for word, i in word_to_id_fr.items()}

#hyperparameters
batch_size = 80       # Number of sequences in a mini-batch
vocab_size = 30000     # Size of the input vocabulary
hidden_size = 1000     # Number of features in the hidden state
embedding_dim = 620   # Word embedding dimension
maxout_units= 500     # Number of units in the maxout layer
allign_dim=50        # Number of features in the allignment model Tx

encoder = Encoder(vocab_size, hidden_size, embedding_dim, device=device)
decoder = Decoder(vocab_size, hidden_size, embedding_dim, maxout_units, device=device)
model = RNNsearch(encoder, decoder, device=device).to(device)
file_path = '~/projet-mla/src/Baseline/saves/Search_50/best_model.pth'

file_path = os.path.expanduser(file_path)

state_dict = torch.load(file_path, map_location=device)

model.load_state_dict(state_dict["model_state_dict"])

torch.save(model, 'COMPLETE_search.pth')

# Create reverse lookup dictionary for French vocabulary
trg_vocab_reverse = {i: word for word, i in word_to_id_fr.items()}

# Now, translate the sentence using the model
trg_tokens = translate_sentence(sentence_to_translate, word_to_id_eng, word_to_id_fr, model,trg_vocab_reverse,device)

print("Translated Sentence:", ' '.join(trg_tokens))