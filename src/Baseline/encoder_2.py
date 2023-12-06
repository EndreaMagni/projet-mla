import torch.nn as nn
import torch.nn.init as init

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, bidirectional=False, gaussian_sigma=0.01):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, bidirectional=bidirectional)

        self.init_weights(gaussian_sigma)

    def init_weights(self, sigma):
        for name, param in self.gru.named_parameters():
            if 'weight' in name:
                init.normal_(param.data, mean=0.0, std=sigma)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden