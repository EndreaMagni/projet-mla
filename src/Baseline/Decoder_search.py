import torch
import torch.nn.functional as F
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, context_size):
        super(Decoder, self).__init__()
        # Je crois que input_size c'est genre la dim de yi so 1*K 
        # Eyi-1 word embedding matrix m*K
        # Hidden_size = n in the article = 1000
        # m is the word embedding dimensions = 620 (= l)

        self.input_size=input_size
        self.hidden_size = hidden_size
        self.context_size = context_size
        self.L= 620# sequence length or  m in the article 
        self.embedding = nn.Embedding(self.L, input_size)
        self.gru = nn.GRU(input_size + context_size, hidden_size)#Pas sur du input_size + context_size
        
        #self.softmax = ...
        self.custom_weights_init()

    def custom_weights_init(self):
        for name, param in self.gru.named_parameters():
            if 'weight' in name:  
                nn.init.normal_(param, mean=0, std=0.01)  # Initialisation normale des poids

    def forward(self, input, hidden, context):
        
        # Embedding du mot yi-1
        embedded = self.embedding(input).view(1, 1, -1)
        # Concaténation avec le vecteur de contexte 
        gru_input = torch.cat((embedded, context), 2)
        # Passage par la couche GRU  
        output, hidden = self.gru(gru_input, hidden)

        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)