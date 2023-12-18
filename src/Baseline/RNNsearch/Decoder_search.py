import torch
import torch.nn.functional as F
import torch.nn as nn

# Attempt to import the Alignment module from different possible locations
try : from Allignement import Allignement
except : 
    try : from Baseline.RNNsearch.Allignement import Allignement
    except : from RNNsearch.Allignement import Allignement

# Define the Maxout layer used in the Decoder
class Maxout(nn.Module):
    def __init__(self, input_dim: int, out_dim: int, pool_size: int):
        super(Maxout, self).__init__()
        self.output_size = out_dim
        self.pool_size = pool_size
        # Linear layer that will be used before max pooling
        self.linear = nn.Linear(input_dim, out_dim * pool_size)

    def forward(self, x):
        # Apply the linear layer and reshape for max pooling
        output = self.linear(x)
        output = output.view(-1, self.output_size, self.pool_size)
        # Apply max pooling along the pool size dimension
        output = torch.max(output, 2)[0]
        return output

# Define the Decoder class
class Decoder(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, embedding_size: int, maxout_unit: int, device: torch.device):
        super(Decoder, self).__init__()
        self.device = device
        # Dimensions for the GRU input
        input_size_gru = hidden_size * 2 + embedding_size
        # Dimensions for the Maxout input
        input_size_maxout = hidden_size * 3 + embedding_size
        self.hidden_size = hidden_size
        self.output_size = vocab_size
        self.vocab_size = vocab_size

        # Initialize the Alignment, embedding, GRU, and maxout layers, and the final fully connected layer
        self.Alignment = Allignement(hidden_size, device)
        self.embedding = nn.Linear(hidden_size, embedding_size)
        self.gru = nn.GRU(input_size_gru, hidden_size, batch_first=True)
        self.maxout = Maxout(input_size_maxout, maxout_unit, 2)  # Example: maxout_unit=500
        self.fc = nn.Linear(maxout_unit, vocab_size)
        # Transformation applied to the previous hidden state
        self.Ws = nn.Linear(hidden_size, hidden_size)

    def forward(self, enc_out, hidden_enc):
        # Prepare initial states and outputs
        batch_size = enc_out.size(0)
        # Transform the initial hidden state
        si = torch.tanh(self.Ws(hidden_enc[1, :, :])).unsqueeze(0)

        attention_weights = []
        # Initialize output tensor on the correct device
        outputs = torch.zeros(hidden_enc.size(0), hidden_enc.size(1), self.maxout.output_size).to(self.device)
        # Prepare the initial input for the GRU
        yi = torch.zeros(batch_size, self.hidden_size).to(self.device)
        outputs = []
        
        # Iterate over each time step
        for i in range(enc_out.size(1)):
            # Calculate the context vector and attention weights
            context, alpha_ij = self.Alignment(si, enc_out)
            attention_weights.append(alpha_ij)

            # Embed the input and apply the GRU
            yi_emb = self.embedding(yi)
            yi, si = self.gru(torch.cat([yi_emb, context], dim=1).unsqueeze(1), si)
            yi = yi.squeeze(1)

            # Apply the maxout layer and the final fully connected layer
            maxout_output = self.maxout(torch.cat((si.view(si.shape[1], -1), context, yi_emb), dim=1))
            output_fc = self.fc(maxout_output)
            # Store the output
            outputs.append(output_fc)

        # Stack and transpose the outputs to match the expected dimensions
        return torch.stack(outputs).transpose(0, 1), torch.stack(attention_weights).transpose(0, 1)
