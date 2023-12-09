import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple

class Maxout(nn.Module):
    # Maxout network implementation
    def __init__(self, in_features, out_features, pool_size):
        super(Maxout, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.pool_size = pool_size
        self.lin = nn.Linear(in_features, out_features * pool_size)

    # Forward pass of the Maxout network
    def forward(self, inputs):
        shape = inputs.size()
        shape = shape[:-1] + (self.out_features, self.pool_size)
        out = self.lin(inputs)
        m, _ = out.view(*shape).max(-1)
        return m

class Attention(nn.Module):
    # Initialisation du module d'attention
    def __init__(self, enc_hidden_size, dec_hidden_size, attn_dim, maxout_pool_size, deep_output_layers):
        super(Attention1, self).__init__()

        # Définition des couches linéaires pour le calcul de l'attention
        self.Wa = nn.Linear(enc_hidden_size, attn_dim, bias=False)
        self.Ua = nn.Linear(dec_hidden_size, attn_dim, bias=False)
        self.va = nn.Parameter(torch.rand(attn_dim, 1))

        self.maxout_pool_size = maxout_pool_size

        # Construction des couches de sortie profondes (deep output layers)
        deep_output_layers_list = [Maxout(enc_hidden_size + dec_hidden_size, dec_hidden_size, maxout_pool_size)]
        for _ in range(1, deep_output_layers):
            deep_output_layers_list.append(Maxout(dec_hidden_size, dec_hidden_size, maxout_pool_size))
        self.deep_output = nn.Sequential(*deep_output_layers_list)

    # Forward pass du module d'attention
    def forward(self, output, context, mask):
        batch_size = context.shape[0]
        enc_seq_len = context.shape[1]
        dec_seq_len = output.shape[1]

        # Préparation du contexte et de la sortie pour le calcul de l'énergie
        context_transformed = self.Wa(context.view(batch_size * enc_seq_len, -1)).view(batch_size, enc_seq_len, -1)
        output_transformed = self.Ua(output)
        output_transformed = output_transformed.unsqueeze(1).expand(-1, enc_seq_len, -1)

        # Calcul des scores d'énergie
        e_ij = torch.bmm(context_transformed, self.va.repeat(batch_size, 1, 1)).squeeze(2) + \
               torch.bmm(output_transformed, self.va.repeat(batch_size, 1, 1)).squeeze(2)
        e_ij = torch.tanh(e_ij)

        # Application du masque et calcul des poids d'attention
        e_ij.data.masked_fill_(mask, -float('inf'))
        alpha_ij = F.softmax(e_ij, dim=1)

        # Calcul du vecteur de contexte
        context = torch.bmm(alpha_ij.unsqueeze(1), context).squeeze(1)

        # Calcul de la sortie profonde (deep output)
        combined = torch.cat((context, output), dim=2)
        deep_output = self.deep_output(combined)

        return deep_output, alpha_ij
