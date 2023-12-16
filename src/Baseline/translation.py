import os
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from loadingpy import pybar

from configuration import config
from easydict import EasyDict
cfg = EasyDict(config)


# from .dataloader import create_dataloader
# from .loss import Loss
import random

class BlankStatement:
    def __init__(self):
        pass

    def __enter__(self):
        return None

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any):
        return None

class BaselineTrainer:

    def __init__(self, quiet_mode: bool, save_name: str = 'Nothing') -> None:

        self.quiet_mode         = quiet_mode
        self.create_dir(name = save_name)

    def create_dir(self, name = 'Nothing') :
        if name == 'Nothing' :
            import datetime
            now         = datetime.datetime.now()
            name        = now.strftime("%Y-%m-%d %H:%M:%S")

        self.dir = f"saves/{name}"
        os.makedirs(self.dir, exist_ok=True)
        
    def Init_weights(self, m):
        for name, param in m.named_parameters():
            if 'weight_hh' in name  : init.orthogonal_(param.data) 
            elif 'weight' in name   : nn.init.normal_(param.data, mean=0, std=0.01)  
            else                    : nn.init.constant_(param.data, 0)


    def train(self, model, train_data_loader,val_data_loader, device,
              nohup = False, mode = "RNNEncDec",seq_len = cfg.sequence_length, 
              load_model = "Nothing", chosed_optimizer = "adadelta") :


        name_code = "S" if mode == "RNNSearch" else "E"
        name_code += "50" if seq_len == 50 else "30"
        print("Let the train begin")
        
        self.device     = device
        
        model = model.to(device)
        if load_model == "Nothing" : 
            model.apply(self.Init_weights)
            
        else : 
            state_dict_complet = torch.load(load_model, map_location=torch.device(device))
            model.load_state_dict(state_dict_complet["model_state_dict"])


        len_batches = len(train_data_loader)

        if chosed_optimizer == "adadelta": 
            optimizer = torch.optim.Adadelta(model.parameters(), lr=cfg.lr, rho=0.95, eps=1e-06)
        elif chosed_optimizer == "adam":
            optimizer = torch.optim.Adam(model.parameters())

        # if load_model != "Nothing" : 
        #     optimizer.load_state_dict(state_dict_complet['optimizer_state_dict'])

        criterion       = nn.CrossEntropyLoss(reduction="mean", ignore_index = 0) 

        pbar            = pybar(range(cfg.epochs*len_batches), base_str="training")

        train_losses    = []
        val_losses      = []
        
        final_loss = val_loss = None

        best_val_loss = float('inf')

        best_model = None

        if mode == "RNNSearch "    :  best_attention_weights = []
        
        for epoch in range(cfg.epochs):
            
            if mode == "RNNSearch "    : attention_weights=[]

            model.train()

            loss_for_one_epoch = []
            n_batch = -1

            for input_batch, output_batch  in train_data_loader :
                n_batch += 1

                input_batch, output_batch = input_batch.to(device), output_batch.to(device)

                #output_batch_onehot = F.one_hot(torch.tensor(output_batch), num_classes=cfg.vocabulary_size).float().to(device)

                optimizer.zero_grad()

                if mode == "RNNEncDec" : output = model(input_batch, output_batch)                  
                else : output, attention_weights = model(input_batch)
                    
                output = output.reshape(-1, cfg.vocabulary_size)

                # output_batch_onehot = output_batch_onehot.view(-1, cfg.vocabulary_size)

                output_batch_onehot = output_batch.view(-1)

                loss                        = criterion(output, output_batch_onehot)
                loss.backward()

                optimizer.step()

                loss_for_one_epoch          += [loss.item()]
                if nohup and n_batch % (len_batches//10) == 0 : print(f"{name_code} Epoch : {epoch}, batch : {n_batch}/{len_batches}, loss : {round(loss.item(),4)}")
                if not self.quiet_mode  and not nohup : 
                    pbar.set_description(description=f"{name_code} Epoch {epoch+1}, loss: {final_loss}, val: {val_loss}, Batch {n_batch+1}/{len_batches} loss : {round(loss.item(),4)}")
                    pbar.__next__()


                
            final_loss                  = np.mean(loss_for_one_epoch)
            val_loss                    = self.evaluate(model, val_data_loader, criterion, mode = mode)

            train_losses                += [final_loss]
            val_losses                  += [val_loss] 

            final_loss                  = round(final_loss,4)
            val_loss                    = round(val_loss,4)


            np.save(f"{self.dir}/loss.npy", np.array(train_losses))
            np.save(f"{self.dir}/loss_val.npy", np.array(val_losses))


            if val_loss < best_val_loss :
                best_val_loss = val_loss
                best_model = model.state_dict()
                if mode == "RNNSearch "    : best_attention_weights=attention_weights

                torch.save({'model_state_dict': best_model,
                            'optimizer_state_dict': optimizer.state_dict()}, 
                            f"{self.dir}/best_model.pth")
                torch.save(model, 
                           f"{self.dir}/best_model_usable.pth")
                
            torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()}, 
                        f"{self.dir}/last_model.pth")


            if nohup : print(f"Epoch {epoch + 1}/{cfg.epochs} : loss = {final_loss}, val_loss = {val_loss}")
        
        try :pbar.__next__()
        except StopIteration : print("The training has ended")

    def evaluate(self, model, val_data_loader, criterion, mode = 'RNNEncDec'):
        
        model.eval()
        
        loss_for_one_epoch = []


        with torch.no_grad():
            
            for input_batch, output_batch in val_data_loader :
                
                input_batch, output_batch= input_batch.to(self.device), output_batch.to(self.device)
                
                #output_batch_onehot= F.one_hot(torch.tensor(output_batch), num_classes=cfg.vocabulary_size).float().to(self.device)

                if mode == "RNNEncDec" : output  = model(input_batch, output_batch)
                else : output, attention_weights = model(input_batch)
                    
                output = output.reshape(-1, cfg.vocabulary_size)

                
                # output_batch_onehot = output_batch_onehot.view(-1, cfg.vocabulary_size)

                output_batch_onehot = output_batch.view(-1)

                loss                        = criterion(output, output_batch_onehot)

                loss_for_one_epoch          += [loss.item()]

        return np.mean(loss_for_one_epoch)
