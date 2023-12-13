import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import tqdm
def make_batch(data,batch_size):   
    # creating pairs
    pairs = [[item['translation']['en'], item['translation']['fr']] for item in data]
    batches = [pairs[i:i + batch_size] for i in range(0, len(pairs), batch_size)]
    return np.array(batches)

def index_batch(batch,word_to_id_eng,word_to_id_fr):
    input_batch = []
    output_batch = []
    for pair in batch: 
        
        #input_batch.append([np.eye(vocab_size)[[word_to_id_eng.get(n,0) for n in pair[0]]]])
        #output_batch.append([np.eye(vocab_size)[[word_to_id_fr.get(n,0) for n in pair[1]]]])
        input_batch.append([word_to_id_eng.get(n,0) for n in pair[0]])
        output_batch.append([word_to_id_fr.get(n,0) for n in pair[1]])
        
    return torch.tensor(input_batch, dtype=torch.long), torch.tensor(output_batch, dtype=torch.long)

def Init_weights(model):
    for name, param in model.named_parameters():
        if 'weight_hh' in name  : init.orthogonal_(param.data) 
        elif 'weight' in name   : nn.init.normal_(param.data, mean=0, std=0.01)  
        else                    : nn.init.constant_(param.data, 0)


          
def train(model, train_data, val_data, word_to_id_eng, word_to_id_fr, batch_size, vocab_size, learning_rate, epochs, device,print_every):
    model = model.to(device)
    model.apply(Init_weights)

    optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate, rho=0.95, eps=1e-06)
    criterion = nn.NLLLoss(reduction="mean")
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model = None
     
    train_batches=make_batch(train_data, batch_size)
    val_batches=make_batch(val_data, batch_size)
    
    for epoch in range(epochs):
        attention_weights=[]
        model.train()
        total_loss = 0
        for batch in tqdm(train_batches, desc=f'Epoch {epoch + 1}/{epochs}'):
            input_batch, output_batch = index_batch(batch, len(word_to_id_eng), word_to_id_eng, word_to_id_fr)
            input_batch, output_batch = input_batch.to(device), output_batch.to(device)

            optimizer.zero_grad()
            output, attention_weights = model(input_batch)
            output = output.reshape(-1, vocab_size)
            output = F.log_softmax(output, dim=1)
            output_batch = output_batch.view(-1).long()

            loss = criterion(output, output_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_data)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_batches:
                input_batch, output_batch =index_batch(batch, len(word_to_id_eng), word_to_id_eng, word_to_id_fr)
                input_batch, output_batch = input_batch.to(device), output_batch.to(device)
                output, _ = model(input_batch)
                output = output.reshape(-1, vocab_size)
                output = F.log_softmax(output, dim=1)
                output_batch = output_batch.view(-1).long()

                val_loss = criterion(output, output_batch)
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_data)
        val_losses.append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = model.state_dict()
            best_attention_weights=attention_weights

        if epoch % print_every == 0:
            print(f'Epoch: {epoch + 1}, Training Loss: {avg_train_loss:.6f}, Validation Loss: {avg_val_loss:.6f}')

    print('Training Finished')
    np.save('train_losses.npy', np.array(train_losses))
    np.save('val_losses.npy', np.array(val_losses))
    torch.save(best_model, 'best_model.pth')
    return best_attention_weights

          

     

    

