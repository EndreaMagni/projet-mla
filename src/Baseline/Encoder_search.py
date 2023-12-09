import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.bi_RNN=nn.RNN(input_size=embedding_size,
                            hidden_size=hidden_size,
                            bidirectional=True)
        
    def forward(self, input):
        embedded=self.embedding(input)
        # GRU layer: forward and backward states are automatically handled by the GRU
        outputs = self.bi_gru(embedded)  # outputs shape: (batch, seq_length, 2*hidden_size)
        
        # Concatenation of forward and backward states is already done within the GRU layer       
        return outputs

#test
# Hyperparameters
batch_size = 80       # Number of sequences in a mini-batch
input_size = 30000     # Size of the input vocabulary
hidden_size = 1000     # Number of features in the hidden state
embedding_dim = 620   # Word embedding dimension


# Instantiate the model
model =Encoder(input_size, hidden_size, embedding_dim)

# Example input (batch of sequences)
sequence_length = 20
# Each integer in the sequence corresponds to a word in the vocabulary

x_batch = torch.randint(0, input_size, (batch_size, sequence_length))

# Forward pass
output = model(x_batch)

        
        
        


        
        