import torch
import torch.nn as nn
from Baseline.configuration import config as cfg

class Seq2Seq(nn.Module):

    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, input_token_sequence):
        
        # Liste pour stocker les sorties
        outputs = []
        
        # Obtenir la sortie de l'encodeur et le vecteur de contexte
        encoder_output, context_vector = self.encoder(input_token_sequence)
        
        # Initialiser la matrice de sorties avec des zéros
        outputs = torch.zeros(cfg["batch_size"],
                              encoder_output.size(1), 
                              cfg['vocabulary_size']).to(self.device)
        
        # Initialiser l'état caché avec le vecteur de contexte
        hidden_state = context_vector
        
        # Initialiser le token d'entrée pour le décodeur avec des zéros
        input_target_token = torch.zeros(context_vector.size(1)).to(self.device).long()
        
        # Boucle pour générer la séquence de sortie
        for t in range(1, encoder_output.size(1)):

            # Appeler le décodeur pour obtenir la prédiction à l'étape t
            yi, output, hidden_state = self.decoder(input_target_token, 
                                                   hidden_state, 
                                                   context_vector)
            
            # Ajouter la prédiction à la liste des sorties
            outputs[t] = output
            
            # Mettre à jour le token d'entrée pour le prochain pas
            input_target_token = yi.argmax(2).squeeze(0)
        
        
        return outputs
