import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
 
class Allignement(nn.Module):
    # Initialisation du module d'Allignement
    def __init__(self, hidden_size : int, device : torch.device )-> None:
    
        super(Allignement, self).__init__()

        # Définition des couches linéaires pour le calcul de l'Allignement
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size*2, hidden_size)
        self.va = nn.Linear(hidden_size, 1)
       

    # Forward pass du module d'Allignement
    def forward(self, si,enc_out):
        # context : si
        # encoder ouput : enc_out 
        # Calcul du scores d'énergie
        e_ij = torch.tanh(self.Wa(si) + self.Ua(enc_out).transpose(0,1))
        breakpoint()
        e_ij = self.va(e_ij).squeeze(2).transpose(0,1)
        #Calcul du poids d'attenttion 
        alpha_ij = F.softmax(e_ij, dim=1)
        # Calcul du vecteur de contexte
        context = torch.bmm(alpha_ij.unsqueeze(1), enc_out).squeeze(1)
        return context,alpha_ij 
