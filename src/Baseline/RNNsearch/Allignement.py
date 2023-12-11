import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
 
class Allignement(nn.Module):
    # Initialisation du module d'Allignement
    def __init__(self, hidden_size, attn_dim):
    
        super(Allignement, self).__init__()

        # Définition des couches linéaires pour le calcul de l'Allignement
        self.Wa = nn.Linear(hidden_size, attn_dim, bias=False)
        self.Ua = nn.Linear(hidden_size*2, attn_dim, bias=False)
        self.va = nn.Linear(attn_dim, 1, bias=False)
        #self.va = nn.Parameter(torch.rand(attn_dim, 1))

    # Forward pass du module d'Allignement
    def forward(self, enc_out, si):
        # context : si
        # ouput : enc_out 
        batch_size = si.size(0)
        enc_seq_len = si.size(1)

        # Préparation du contexte et de la sortie pour le calcul de l'énergie
        si_transformed = self.Wa(si.view(batch_size * enc_seq_len, -1)).view(batch_size, enc_seq_len, -1)
        enc_out_transformed = self.Ua(enc_out).unsqueeze(1).expand(-1, enc_seq_len, -1)

        # Calcul du scores d'énergie
        e_ij= self.va(torch.tanh(self.Wa(si) + self.Ua(enc_out)))

        #Calcul du poids d'attenttion 
        alpha_ij = F.softmax(e_ij, dim=1)

        # Calcul du vecteur de contexte
        context = torch.bmm(alpha_ij.unsqueeze(1), si).squeeze(1)
        
        return context,alpha_ij 
