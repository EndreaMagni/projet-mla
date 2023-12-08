import torch
<<<<<<< HEAD
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, n, m):
        super(Decoder, self).__init__()
        #Init des matrices de poids 
        # W, U, C, Wz, Uz, Cz, Wr, Ur, Cr sont des instances de nn.Parameter
        
        self.W = nn.Parameter(torch.randn(n, m))
        self.U = nn.Parameter(torch.randn(n, n))
        self.C = nn.Parameter(torch.randn(n, 2*n))

        self.Wz = nn.Parameter(torch.randn(n, m))
        self.Uz = nn.Parameter(torch.randn(n, n))
        self.Cz = nn.Parameter(torch.randn(n, 2*n))

        self.Wr = nn.Parameter(torch.randn(n, m))
        self.Ur = nn.Parameter(torch.randn(n, n))
        self.Cr = nn.Parameter(torch.randn(n, 2*n))
        
    def calculate_context_vector(self):  
        pass

    def forward(self, Ey_minus_1, s_minus_1):
        #Calcue de ci
        ci = self.calculate_context_vector(...)

        # Calcule de ri, zi, si_tilde
        ri = torch.sigmoid(self.Wr @ Ey_minus_1 + self.Ur @ s_minus_1 + self.Cr @ ci)

        zi = torch.sigmoid(self.Wz @ Ey_minus_1 + self.Uz @ s_minus_1 + self.Cz @ ci)

        si_tilde = torch.tanh(self.W @ Ey_minus_1 + self.U @ (ri * s_minus_1) + self.C @ ci)

        # Calcule de si
        si = (1 - zi) * s_minus_1 + zi * si_tilde
        return si

# input_dim : #dimension de l'embedding du mot (Ey_minus_1).
# hidden_dim : #dimension de l'état caché (s_minus_1, si).
# context_dim :dimension du vecteur de contexte (ci).

#Again, m and n are the word embedding dimensionality and the number of hidden units, respectively
=======
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
>>>>>>> 3fe8633aca36de559b7def7ab7724a61fd7afec9
