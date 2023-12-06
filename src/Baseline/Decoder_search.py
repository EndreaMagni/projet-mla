import torch
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