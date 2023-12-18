import torch
import torch.nn as nn
import torch.nn.functional as F


class MaxoutLayer(nn.Module):
    def __init__(self, input_size, output_size, num_pieces):
        super(MaxoutLayer, self).__init__()

        # Couche linéaire pour le maxout
        self.fc             = nn.Linear(input_size,
                                        output_size * num_pieces)
        
        self.num_pieces     = num_pieces

    def forward(self, input_tensor):
        # Application de la couche linéaire
        output              = self.fc(input_tensor)

        # Réarrangement des dimensions pour effectuer le maxout
        output              = output.view(-1, 
                                          self.num_pieces, 
                                          output.size(1)//self.num_pieces)

        # Opération de maxout
        output, _           = torch.max(output, 
                                        dim=1)

        return output
    

class Decoder(nn.Module):
    def __init__(self, output_size, embedding_size, hidden_size, maxout_size):
        super(Decoder, self).__init__()

        # Couche d'embedding pour les jetons de sortie
        self.embedding      = nn.Embedding(output_size, 
                                           embedding_size)

        # Couche GRU pour le traitement de la séquence cachée
        self.gru            = nn.GRU(hidden_size + embedding_size, 
                                     hidden_size)
        
        # Couche maxout pour la transformation non linéaire
        self.fc1            = MaxoutLayer(embedding_size + hidden_size * 2,
                                          maxout_size, 
                                          2)

        # Couche linéaire finale pour la prédiction
        self.fc2            = nn.Linear(maxout_size, 
                                        output_size)

    def forward(self, input_token, hidden_state, context_vector):
        # Ajout d'une dimension à l'entrée du jeton
        input_token         = input_token.unsqueeze(0)
        
        # Embedding du jeton d'entrée
        embedded            = self.embedding(input_token)
              
        # Construction de l'entrée de la couche GRU
        gru_input           = torch.cat((embedded, 
                                        context_vector), dim = 2)

        # Passage à travers la couche GRU
        yi, hidden_state     = self.gru(gru_input, 
                                       hidden_state)

        # Construction du vecteur d'entrée pour la couche maxout
        output              = torch.cat((embedded.squeeze(0), 
                                         hidden_state.squeeze(0), 
                                         context_vector.squeeze(0)), 
                                         dim=1)

        # Transformation non linéaire maxout
        output              = self.fc1(output)

        # Prédiction finale
        prediction          = self.fc2(output)

        return yi, prediction, hidden_state
