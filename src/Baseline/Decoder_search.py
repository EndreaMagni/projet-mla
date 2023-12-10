import torch
import torch.nn.functional as F
import torch.nn as nn
from attention import Attention

class Maxout(nn.Module):
    def __init__(self, input_dim, out_dim, pool_size=1):
        super(Maxout, self).__init__()
        # the size of the maxout hidden laye : out_dim = 500    
        self.out_features = out_dim
        self.pool_size = pool_size
        self.lin = nn.Linear(input_dim, out_dim * pool_size)

    def forward(self, x):
        # Maxout operation
        output = self.linear(x)
        output = output.view(-1, self.out_features, self.pool_size)
        output = torch.max(output, 2)[0]
        return output

    
class Decoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, context_size,embedding_size,maxout_size):
        super(Decoder, self).__init__()
       
        self.input_size= hidden_size*2 
    
        # Attention model 
        self.attention= Attention(self.input_size, hidden_size, hidden_size)
        
        self.embedding = nn.Embedding(vocab_size, embedding_size)

        self.gru = nn.GRU(self.input_size + context_size, hidden_size)
    
        self.maxout = Maxout(hidden_size * 2 + context_size, maxout_size)
        
        self.fc = nn.Linear(maxout_size, vocab_size)
                 

    def forward(self, input, hidden, context):
        # Embedding du mot yi-1
        input = input.unsqueeze(0)
        embedded = self.embedding(input)
        embedded =self.dropout(embedded)
    
        context, _ = self.attn(last_hidden[-1], encoder_outputs, mask)#mask??
        
        # Concaténation avec le vecteur de contexte
        gru_input = torch.cat((embedded, context.unsqueeze(0)), 2)

        # Passage par la couche GRU
        gru_output, hidden = self.gru(gru_input, hidden.unsqueeze(0)) #hidden.unsqueeze(0) ?

        embedded = embedded.squeeze(0)
        maxout_input = gru_output.squeeze(0) 
     
        maxout_output=self.maxout(torch.cat((maxout_input, context, embedded), dim = 1))

        output_fc = self.fc(maxout_output)
        output=F.softmax(output_layer)

        return output, hidden 
        
    
