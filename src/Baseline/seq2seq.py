import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class Seq2Seq(nn.Module):
    def __init__(self, input_size, hidden_size, bidirectional=False, gaussian_sigma=0.01):
        super(Seq2Seq, self).__init__()

    def forward(self, input, hidden):
