import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_size):
        super(Encoder,self).__init__()

        
        self.embedding = nn.Embedding(input_size, embedding_size)
        
        self.bi_RNN=nn.GRU(input_size=embedding_size,
                            hidden_size=hidden_size,
                            bidirectional=True,
                            batch_first=True)
        
        
    def forward(self, input):
        embedded=self.embedding(input)
        # GRU layer: forward and backward states are automatically handled by the GRU
        outputs,h_n= self.bi_RNN(embedded)  # outputs shape: (batch, seq_length, 2*hidden_size)
        
        # Concatenation of forward and backward states is already done within the GRU layer       
        return outputs,h_n



        
        
        


        
        