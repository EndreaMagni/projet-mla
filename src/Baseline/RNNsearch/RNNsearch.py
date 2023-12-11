import torch
import torch.nn as nn
import torch.nn.functional as F

class RNNsearch(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(RNNsearch, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, max_iterations):
        batch_size,trg_len = trg.shape

        trg_vocab_size = self.decoder.output_size

        outputs = torch.zeros(batch_size,trg_len, trg_vocab_size).to(self.device)

        # Debugging: Try-catch block for encoder
        try:
            encoder_outputs, hidden_enc = self.encoder(src)
        except Exception as e:
            print("Error in encoder:", e)

        outputs, attention_weights  = self.decoder(encoder_outputs,hidden_enc)                         


        return outputs,attention_weights 
