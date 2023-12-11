import torch
import torch.nn as nn
import torch.nn.functional as F
from Allignement import Allignement


class RNNsearch(nn.Module):
    def __init__(self, encoder, decoder):
        super(RNNsearch, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        

    def forward(self, src,batch_size):
        
        trg_len = src.size(2)

        trg_vocab_size = self.decoder.output_size

        outputs = torch.zeros(batch_size,trg_len, trg_vocab_size)

        # Debugging: Try-catch block for encoder
        
        encoder_outputs, hidden_enc = self.encoder(src)

        outputs, attention_weights  = self.decoder(encoder_outputs,hidden_enc)                         


        return outputs,attention_weights 
