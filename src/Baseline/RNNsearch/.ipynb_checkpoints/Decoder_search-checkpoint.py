import torch
import torch.nn.functional as F
import torch.nn as nn
from Allignement import Allignement


class Maxout(nn.Module):
    def __init__(self, input_dim: int, out_dim: int, pool_size: int):
        super(Maxout, self).__init__()
        self.output_size = out_dim
        self.pool_size = pool_size
        self.linear = nn.Linear(input_dim, out_dim * pool_size)

    def forward(self, x):
        output = self.linear(x)
        output = output.view(-1, self.output_size, self.pool_size)
        output = torch.max(output, 2)[0]
        return output


class Decoder(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, embedding_size: int, maxout_unit: int, device: torch.device):
        super(Decoder, self).__init__()

        input_size_gru = hidden_size * 2 + embedding_size
        input_size_maxout = hidden_size * 3 + embedding_size
        self.hidden_size = hidden_size
        self.output_size = vocab_size
        self.vocab_size = vocab_size

        self.Allignement = Allignement(hidden_size, device)

        self.embedding = nn.Linear(hidden_size, embedding_size).to(device)
        self.gru = nn.GRU(input_size_gru, hidden_size, batch_first=True).to(device)

        self.maxout = Maxout(input_size_maxout, maxout_unit, 2).to(device)  # maxout=500

        self.fc = nn.Linear(maxout_unit, vocab_size).to(device)

        self.Ws = nn.Linear(hidden_size, hidden_size).to(device)

    def forward(self, enc_out, hidden_enc):
        batch_size = enc_out.size(0)
        si = torch.tanh(self.Ws(hidden_enc[1, :, :])).unsqueeze(0).to(hidden_enc.device)

        attention_weights = []
        outputs = torch.zeros(hidden_enc.size(0), hidden_enc.size(1), self.maxout.output_size).to(hidden_enc.device)
        yi = torch.zeros(batch_size, self.hidden_size).to(enc_out.device)

        for i in range(enc_out.size(1)):
            context, alpha_ij = self.Allignement(si, enc_out)
            attention_weights.append(alpha_ij)

            yi_emb = self.embedding(yi)
            yi, si = self.gru(torch.cat([yi_emb, context], dim=1).unsqueeze(1), si)
            yi = yi.squeeze(1)

            maxout_output = self.maxout(torch.cat((si.view(si.shape[1], -1), context, yi_emb), dim=1))
            output_fc = self.fc(maxout_output)
            outputs[:, i, :] = output_fc


            

        return torch.stack(outputs).transpose(0, 1), torch.stack(attention_weights).transpose(0, 1)
