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
    criterion = nn.NLLLoss(reduction="mean")
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model = None
     

    
    for epoch in range(epochs):
        attention_weights=[]
        model.train()
        total_loss = 0
        for input_batch, output_batch , output_batch_onthot in tqdm(train_data_loader, desc=f'Training Epoch {epoch + 1}/{epochs}'):
            input_batch, output_batch,output_batch_onthot = input_batch.to(device), output_batch.to(device),output_batch_onthot.to(device)

            optimizer.zero_grad()
            output, attention_weights = model(input_batch)
            output = output.reshape(-1, vocab_size)
            output = F.log_softmax(output, dim=1)

            output_batch_onthot = output_batch_onthot.view(-1).long()

            loss = criterion(output, output_batch_onthot)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            print(f'Epoch: {epoch + 1}, Training Loss: {loss:.6f}')

        avg_train_loss = total_loss / len(train_data_loader)
        train_losses.append(avg_train_loss)
        
        """"
        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_batches, desc=f'Validation Epoch {epoch + 1}/{epochs}'):

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
    """
    #return best_attention_weights

          

     

    

