import torch.nn as nn
# import torch.nn.init as init

class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size):
        super(Encoder, self).__init__()

        self.hidden_size    = hidden_size

        self.embedding      = nn.Embedding(input_size, 
                                           embedding_size)

        self.gru            = nn.GRU(embedding_size, 
                                     hidden_size)

    def forward(self, input_token_sequence):
        
        embedded            = self.embedding(input_token_sequence)

        output,hidden_state = self.gru(embedded)

        return hidden_state

