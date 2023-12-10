import torch.nn as nn
# import torch.nn.init as init

class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, gaussian_sigma=0.01):
        super(Encoder, self).__init__()

        self.hidden_size    = hidden_size

        self.embedding      = nn.Embedding(input_size, 
                                           embedding_size)

        self.gru            = nn.GRU(embedding_size, 
                                     hidden_size)


    #     self.init_weights(gaussian_sigma)
    # def init_weights(self, sigma):
    #     for name, param in self.gru.named_parameters():
    #         if 'weight' in name:
    #             init.normal_(param.data, mean=0.0, std=sigma)

    def forward(self, input_token_sequence):
        print("dimensions de l'input token : ",input_token_sequence.size())
        embedded            = self.embedding(input_token_sequence)
        print("dimensions de l'embedded : ", embedded.size())
        print("b",embedded.size())

        
        output,hidden_state = self.gru(embedded)

        return hidden_state