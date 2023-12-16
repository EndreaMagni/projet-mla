import torch
import torch.nn as nn
from RNNencdec.encoder import Encoder
from RNNencdec.decoder import Decoder
from configuration import config as cfg

class Seq2Seq(nn.Module):

    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder            = encoder
        self.decoder            = decoder
        self.device             = device
        
    def forward(self, input_token_sequence, target_token_sequence):
                
        target_size, batch_size = target_token_sequence.shape[:2]

        vocabulary_size         = cfg["vocabulary_size"]
        
        outputs                 = torch.zeros(target_size, 
                                              batch_size, 
                                              vocabulary_size).to(self.device)
        
        context_vector          = self.encoder(input_token_sequence)
        
        hidden_state            = context_vector
        
        input_target_token      = target_token_sequence[0,:]
        
        for t in range(1, target_size):
            
            output, hidden_state    = self.decoder(input_target_token, 
                                                   hidden_state, 
                                                   context_vector)

            outputs[t]              = output
            
            input_target_token     = output.argmax(1) 

        return outputs
