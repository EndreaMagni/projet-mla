import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()

        self.linear = nn.Linear(hidden_size, hidden_size)
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        
        # Nouvelles matrices Gl et Gr
        self.Gl = nn.Linear(hidden_size, 500)
        self.Gr = nn.Linear(500, 1000)
        
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        # Ajout de la linear layer et de l'activation tanh pour l'initialisation du hidden state
        hidden = self.linear(hidden)
        hidden = F.tanh(hidden)

        # Embedding de l'input
        embedded = self.embedding(input).view(1, 1, -1)

        # Passage à travers la GRU
        output, hidden = self.gru(embedded, hidden)

        # Application de ReLU sur la sortie
        output = F.relu(output)

        # Calcul de Gl et Gr
        Gl_result = self.Gl(output[0])
        Gr_result = self.Gr(Gl_result)

        # Calcul des logits
        logits = self.out(Gr_result)

        # Application de softmax pour obtenir les probabilités
        probabilities = F.softmax(logits, dim=1)

        return probabilities, hidden
