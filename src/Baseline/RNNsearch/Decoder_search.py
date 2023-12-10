import torch
import torch.nn.functional as F
import torch.nn as nn
from attention import Attention
    
class Maxout(nn.Module):
    def __init__(self, input_dim, out_dim, pool_size=2):
        super(Maxout, self).__init__()
        # the size of the maxout hidden laye : out_dim = 500    
        self.out_features = out_dim
        self.pool_size = pool_size
        self.lin = nn.Linear(input_dim, out_dim * pool_size)

    def forward(self, x):
        # Maxout operation
        output = self.linear(x)
        output = output.view(-1, self.out_features, self.pool_size)
        output = torch.max(output, 2)[0]
        return output

    
class Decoder(nn.Module):
    def __init__(self, vocab_size, hidden_size,embedding_size,maxout_unit):
        super(Decoder, self).__init__()

        input_size_gru= hidden_size*3 + embedding_size
        input_size_attn= hidden_size*3
        input_size_maxout= hidden_size*3 + embedding_size
        self.hidden_size=hidden_size

        self.attention= Attention(input_size_attn, hidden_size, hidden_size)
        
        self.embedding = nn.Embedding(vocab_size, embedding_size)

        self.gru = nn.GRU(input_size_gru, hidden_size,batch_first=True)

        self.maxout = Maxout(input_size_maxout , maxout_unit) # maxout=500
        
        self.fc = nn.Linear(maxout_unit, vocab_size)
                 

    def forward(self,enc_out):# hidden peut etre en in 
        batch_size = enc_out.size(0)
        si= torch.zeros(1, batch_size ,self.hidden_size)

        # faire for i in h[1]
        attention_weights=[]
        outputs = []
        for i in range(enc_out.size(1)) :
            # Calculer le vecteur de contexte avec le model d'alignement 
            context, alpha_ij =self.attention(si , enc_out)
            attention_weights.append(alpha_ij)

            # Passage par la couche GRU
            yi, si = self.gru(context.unsqueeze(1), si) 

            yi_emb = self.embedding(yi.squeeze(1))
     
            maxout_output = self.maxout(torch.cat((si, context, yi_emb), dim = 1))
            
            output_fc = self.fc(maxout_output)

            output=F.softmax(output_fc,dim=1)

            outputs.append(output)

        return torch.stack(outputs), attention_weights 
            
        
    
