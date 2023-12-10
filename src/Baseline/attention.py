import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple

class Attention(nn.Module):
    # Initialisation du module d'attention
    def __init__(self, enc_hidden_size, dec_hidden_size, attn_dim):
        super(Attention, self).__init__()

        # Définition des couches linéaires pour le calcul de l'attention
        self.Wa = nn.Linear(enc_hidden_size, attn_dim, bias=False)
        self.Ua = nn.Linear(dec_hidden_size, attn_dim, bias=False)
        self.va = nn.Parameter(torch.rand(attn_dim, 1))

    # Forward pass du module d'attention
    def forward(self, output, context, mask):
        batch_size = context.shape[0]
        enc_seq_len = context.shape[1]

        # Préparation du contexte et de la sortie pour le calcul de l'énergie
        context_transformed = self.Wa(context.view(batch_size * enc_seq_len, -1)).view(batch_size, enc_seq_len, -1)
        output_transformed = self.Ua(output)
        output_transformed = output_transformed.unsqueeze(1).expand(-1, enc_seq_len, -1)

        # Calcul des scores d'énergie
        e_ij = torch.bmm(context_transformed, self.va.repeat(batch_size, 1, 1)).squeeze(2) + \
               torch.bmm(output_transformed, self.va.repeat(batch_size, 1, 1)).squeeze(2)
        e_ij = torch.tanh(e_ij)

        # Application du masque et calcul des poids d'attention
        #e_ij.data.masked_fill_(mask, -float('inf'))
        alpha_ij = F.softmax(e_ij, dim=1)

        # Calcul du vecteur de contexte
        context = torch.bmm(alpha_ij.unsqueeze(1), context).squeeze(1)

        return context, alpha_ij
