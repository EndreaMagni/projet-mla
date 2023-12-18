import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size):
        super(Encoder, self).__init__()

        self.hidden_size    = hidden_size

        # Couche d'embedding pour convertir les jetons d'entrée en vecteurs denses
        self.embedding      = nn.Embedding(input_size, 
                                           embedding_size)

        # Couche GRU pour traiter la séquence d'entrée incorporée
        self.gru            = nn.GRU(embedding_size, 
                                     hidden_size)

    def forward(self, input_token_sequence):
        # Embedding de la séquence de jetons d'entrée
        embedded            = self.embedding(input_token_sequence)

        # Traitement de la séquence incorporée à travers la couche GRU
        output, hidden_state = self.gru(embedded)

        return output, hidden_state
