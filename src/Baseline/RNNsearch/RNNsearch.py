import torch
import torch.nn as nn
import torch.nn.functional as F
from Allignement import Allignement
import numpy as np


class RNNsearch(nn.Module):
    def __init__(self, encoder, decoder):
        super(RNNsearch, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        

    def forward(self, src):
        src = src.squeeze(1)
        batch_size = np.shape(src)[0]
        trg_len = np.shape(src)[1]
        trg_vocab_size = self.decoder.output_size
        outputs = torch.zeros(batch_size,trg_len, trg_vocab_size)

        # Debugging: Try-catch block for encoder
        
        encoder_outputs, hidden_enc = self.encoder(src)

        outputs, attention_weights  = self.decoder(encoder_outputs,hidden_enc)                         


        return outputs,attention_weights 
