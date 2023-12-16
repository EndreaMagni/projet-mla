
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaxoutLayer(nn.Module):
    def __init__(self, input_size, output_size, num_pieces):
        super(MaxoutLayer, self).__init__()

        self.fc             = nn.Linear(input_size,
                                        output_size * num_pieces)
        
        self.num_pieces     = num_pieces

    def forward(self, input_tensor):

        output              = self.fc(input_tensor)

        output              = output.view(-1, 
                                          self.num_pieces, 
                                          output.size(1)//self.num_pieces)

        output, _           = torch.max(output, 
                                        dim=1)

        return output
    

class Decoder(nn.Module):
    def __init__(self, output_size, embedding_size, hidden_size, maxout_size):
        super(Decoder, self).__init__()

        self.embedding      = nn.Embedding(output_size, 
                                           embedding_size)

        self.gru            = nn.GRU(hidden_size + embedding_size, 
                                     hidden_size)
        
        self.fc1            = MaxoutLayer(embedding_size + hidden_size * 2,
                                          maxout_size, 
                                          2)

        self.fc2            = nn.Linear(maxout_size, 
                                        output_size)

    def forward(self, input_token, hidden_state, context_vector):

        input_token         = input_token.unsqueeze(0)
        
        embedded            = self.embedding(input_token)
              
        gru_input           = torch.cat((embedded, 
                                        context_vector), dim = 2)

        _, hidden_state     = self.gru(gru_input, 
                                       hidden_state)

        
        output              = torch.cat((embedded.squeeze(0), 
                                         hidden_state.squeeze(0), 
                                         context_vector.squeeze(0)), 
                                         dim=1)

        output              = self.fc1(output)

        prediction          = self.fc2(output)

    
        return prediction, hidden_state