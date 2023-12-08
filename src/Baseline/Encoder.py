import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, bidirectional=False, gaussian_sigma=0.01):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)

        # L'encodeur pour RNN encoder decoder n'est pas bidirectionnel. Mettre bidirectional = True pour RNN search
        self.gru = nn.GRU(hidden_size, hidden_size, bidirectional=bidirectional)

        # Il faut regarder comment sont initialisés les paramètres pour RNN search et mettre en commentaire la valeur.
        # Si c'est pareil que pour RNNencdec c'est super
        self.init_weights(gaussian_sigma)

        self.linear = nn.Linear(input_size, input_size)

    def init_weights(self, sigma):
        for name, param in self.gru.named_parameters():
            if 'weight' in name:
                init.normal_(param.data, mean=0.0, std=sigma)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        hidden = self.linear(hidden)
        hidden = F.tanh(hidden)

        return output, hidden