import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from tqdm import tqdm


def Init_weights(model):
    for name, param in model.named_parameters():
        if 'weight_hh' in name  : init.orthogonal_(param.data) 
        elif 'weight' in name   : nn.init.normal_(param.data, mean=0, std=0.01)  
        else                    : nn.init.constant_(param.data, 0)


          
def train(model, train_data_loader, val_data_loader,  vocab_size, learning_rate, epochs, device,print_every):
    print('Training Started')
    model = model.to(device)
    model.apply(Init_weights)
    optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate, rho=0.95, eps=1e-06)
    #criterion = nn.NLLLoss(reduction="mean")
    criterion = nn.CrossEntropyLoss(reduction="mean")
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model = None
    best_attention_weights=[]

    for epoch in range(epochs):
        
        attention_weights=[]
        #training
        model.train()
        total_loss = 0
        for input_batch, output_batch  in tqdm(train_data_loader, desc=f'Training Epoch {epoch + 1}/{epochs}'):
            input_batch, output_batch = input_batch.to(device), output_batch.to(device)

            output_batch_onehot = F.one_hot(torch.tensor(output_batch), num_classes=vocab_size).float().to(device)
            optimizer.zero_grad()
            output, attention_weights = model(input_batch)
            output = output.reshape(-1, vocab_size)
            #output = F.log_softmax(output, dim=1)
            
            output_batch_onehot = output_batch_onehot.view(-1, vocab_size)
            
            loss = criterion(output, output_batch_onehot)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() 
            
            
             
            
        avg_train_loss = total_loss / len(train_data_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for input_batch, output_batch in tqdm(val_data_loader, desc=f'Validation Epoch {epoch + 1}/{epochs}'):
                input_batch, output_batch= input_batch.to(device), output_batch.to(device)
                output_batch_onehot= F.one_hot(torch.tensor(output_batch), num_classes=vocab_size).float().to(device)
                output, attention_weights = model(input_batch)
                output = output.reshape(-1, vocab_size)
                #output = F.log_softmax(output, dim=1)
                output_batch_onehot = output_batch_onehot.view(-1, vocab_size)
                val_loss = criterion(output, output_batch_onehot)
                total_val_loss += val_loss.item()
                
                

        avg_val_loss = total_val_loss / len(val_data_loader)
        val_losses.append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = model.state_dict()
            best_attention_weights=attention_weights
            # Inside the loop for validation, where the best validation loss is updated
        
        
        print(f'Epoch: {epoch + 1}, Training Loss: {avg_train_loss:.6f}, Validation Loss: {avg_val_loss:.6f}')

    
    np.save('train_losses_RNNsearch50.npy', np.array(train_losses))
    np.save('val_losses_RNNsearch50.npy', np.array(val_losses))
    torch.save(best_model, 'best_RNNsearch50.pt')
    print('Training Finished')
    return best_attention_weights

        
        

          

     

    

