import os
from typing import Any

import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from loadingpy import pybar
from torchsummary import summary

from Baseline.RNNencdec.encoder import Encoder
from Baseline.RNNencdec.decoder import Decoder
from Baseline.RNNencdec.seq2seq import Seq2Seq


from Baseline.configuration import config
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

    def __init__(self, quiet_mode: bool) -> None:


        if torch.cuda.is_available():           self.device = torch.device("cuda")
        elif torch.backends.mps.is_available(): self.device = torch.device("mps")
        else:                                   self.device = torch.device("cpu")

        self.quiet_mode         = quiet_mode
        self.scope              = (torch.cuda.amp.autocast() if torch.cuda.is_available() else BlankStatement())
        self.create_dir()

    def create_dir(self, name = 'Nothing') :
        if name == 'Nothing' :
            import datetime
            now         = datetime.datetime.now()
            name        = now.strftime("%Y-%m-%d %H:%M:%S")

        self.dir = f"saves/{name}"
        os.makedirs(self.dir, exist_ok=True)


    def make_batch(self,data, is_training = True):
        pairs                   = [(item['translation']['en'], item['translation']['fr']) for item in data]
        if is_training:
            random.shuffle(pairs)
            
        batches                 = [pairs[i:i + cfg.batch_size] for i in range(0, len(pairs), cfg.batch_size)]

        return batches

    def one_hot_encode_batch(self,batch,vocab_size,word_to_id_eng,word_to_id_fr):
        input_batch = []
        output_batch = []
        for i,pair in enumerate(batch): 
            input_batch.append([np.eye(vocab_size)[[word_to_id_eng[n] for n in pair[0]]]])
            output_batch.append([np.eye(vocab_size)[[word_to_id_fr[n] for n in pair[1]]]])

        return input_batch,output_batch
    
    def one_hot_encode_batch_integer(self,batch,vocab_size,word_to_id_eng,word_to_id_fr):
        input_batch = []
        output_batch = []
        for i,pair in enumerate(batch): 

            input_batch.append([word_to_id_eng[n] for n in pair[0]])
            output_batch.append([word_to_id_fr[n] for n in pair[1]])
                
        return input_batch,output_batch 


    def one_hot_encode_batch_mix(self,batch,vocab_size,word_to_id_eng,word_to_id_fr):
        input_batch = []
        output_batch = []
        output_batch_target = []
        for i,pair in enumerate(batch): 

            input_batch.append([word_to_id_eng[n] for n in pair[0]])
            output_batch.append([word_to_id_fr[n] for n in pair[1]])
            output_batch_target.append([np.eye(vocab_size)[[word_to_id_fr[n] for n in pair[1]]]])
                
        return input_batch,output_batch, output_batch_target  
        
    def Init_weights(self, m):
        for name, param in m.named_parameters():
            if 'weight_hh' in name  : init.orthogonal_(param.data) 
            elif 'weight' in name   : nn.init.normal_(param.data, mean=0, std=0.01)  
            else                    : nn.init.constant_(param.data, 0)


    def train(self, train_data, val_data, word_to_id_eng, word_to_id_fr, nohup = False, mode = "RNNEncDec",seq_len = cfg.sequence_length) :

        print("Let the train begin")
        
        self.words_eng  = word_to_id_eng
        self.words_fr   = word_to_id_fr
        
        # batches         = self.make_batch(train_data)

        batches_eval    = self.make_batch(val_data, is_training = False)

        device          = self.device
        
        if mode == "RNNEncDec" : 
            encoder         = Encoder(cfg.input_size,
                                      cfg.embedding_size,
                                      cfg.hidden_size)
            
            decoder         = Decoder(cfg.output_size,
                                      cfg.embedding_size,
                                      cfg.hidden_size,
                                      cfg.maxout_size)
    
            model           = Seq2Seq(encoder, 
                                      decoder, 
                                      device).to(device)

        if mode == "RNNSearch" :
            
            allign_dim      =50
            encoder         = Encoder_RNNSearch(cfg.input_size,
                                                cfg.hidden_size,
                                                cfg.embedding_size)
            
            decoder         = Decoder_RNNSearch(cfg.output_size,
                                                cfg.hidden_size,
                                                cfg.embedding_size,
                                                cfg.maxout_size,
                                                allign_dim)
            
            model           = Seq2Seq_RNNSearch(encoder,
                                                decoder,
                                                device).to(device)

        model.apply(self.Init_weights)
        
        # il faut rectifier l'input size pour le summary

        """
        if not self.quiet_mode : summary(model, 
                                         [(cfg.sequence_length,cfg.batch_size), (cfg.sequence_length,cfg.batch_size)],
                                         dtypes =[torch.LongTensor, torch.LongTensor])

        """


        optimizer       = torch.optim.Adadelta(model.parameters())
        
        criterion       = nn.CrossEntropyLoss(ignore_index = 0)

        batches         = self.make_batch(train_data)
        
        len_batches     = len(batches)

        pbar            = pybar(range(cfg.epochs*len_batches), base_str="training")

        all_loss        = []
        all_val_loss    = []
        
        final_loss = val_loss = None
        
        with self.scope:

            for epoch in range(cfg.epochs):

                batches         = self.make_batch(train_data)
                
                model.train()

                loss_for_one_epoch = []
                
                for n_batch,batch in enumerate(batches):

                    input_batch,output_batch, output_batch_target = self.one_hot_encode_batch_mix(batch,cfg.vocabulary_size,self.words_eng,self.words_fr)

                    input_batch                 = torch.from_numpy(np.array(input_batch)).long().permute(1,0).to(device)
                    output_batch                = torch.from_numpy(np.array(output_batch)).long().permute(1,0).to(device)
                    output_batch_target         = torch.FloatTensor(np.array(output_batch_target)).squeeze(dim = 1).to(device)

                    optimizer.zero_grad()

                    if mode == "RNNEncDec" : 
                        
                        output                      = model(input_batch, output_batch)
                        output_dim                  = output.shape[-1]
                        output                      = output[1:].view(-1, output_dim)
                        output_batch                = output_batch[1:].reshape(-1)
                    
                    else : 
                        output,attention_weights    = model(input_batch)
                    
                    loss                        = criterion(output, output_batch)
                    loss.backward()

                    optimizer.step()

                    loss_for_one_epoch          += [loss.item()]
                    if nohup and n_batch % (len_batches//10) == 0 : print(f"Epoch : {epoch}, batch : {n_batch}/{len_batches}, loss : {round(loss.item(),4)}")
                    if not self.quiet_mode  and not nohup : 
                        pbar.set_description(description=f"Epoch {epoch+1}, loss: {final_loss}, val: {val_loss}, Batch {n_batch+1}/{len_batches} loss : {round(loss.item(),4)}")
                        pbar.__next__()


                    
                final_loss                  = np.mean(loss_for_one_epoch)
                val_loss                    = self.evaluate(model, batches_eval, criterion)

                all_loss                    += [final_loss]
                all_val_loss                += [val_loss] 

                final_loss                  = round(final_loss,4)
                val_loss                    = round(val_loss,4)
                

                if epoch + 1 % cfg.save_loss_ite == 0 : 
                    np.save(f"{self.dir}/loss.npy", np.array(all_loss))
                    np.save(f"{self.dir}/loss_val.npy", np.array(all_val_loss))


                if epoch + 1 in [3,5,10,20,40,60,80,100] :
                    torch.save({'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict()}, 
                                f"{self.dir}/model_{epoch}.pth")

                if nohup : print(f"Epoch {epoch + 1}/{cfg.epochs} : loss = {final_loss}, val_loss = {val_loss}")
        
        try :pbar.__next__()
        except StopIteration : pass

    def evaluate(self, model, batches, criterion):
        
        model.eval()
        
        loss_for_one_epoch = []


        with torch.no_grad():
            
            for batch in batches:

                input_batch,output_batch, output_batch_target = self.one_hot_encode_batch_mix(batch,cfg.vocabulary_size,self.words_eng,self.words_fr)

                input_batch                 = torch.from_numpy(np.array(input_batch)).long().permute(1,0).to(self.device)
                output_batch                = torch.from_numpy(np.array(output_batch)).long().permute(1,0).to(self.device)
                output_batch_target         = torch.FloatTensor(np.array(output_batch_target)).squeeze(dim = 1).to(self.device)
                
                output                      = model(input_batch, output_batch)

                output_dim = output.shape[-1]
    
                output = output[1:].view(-1, output_dim)

                output_batch = output_batch[1:].reshape(-1)

                loss                        = criterion(output, output_batch)

                loss_for_one_epoch          += [loss.item()]

        return np.mean(loss_for_one_epoch)
