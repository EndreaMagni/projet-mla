import os
from typing import Any

import numpy as np
import torch
import torchvision
import torch.nn as nn
from loadingpy import pybar
from torchinfo import summary

from RNNencdec.encoder import Encoder
from RNNencdec.decoder import Decoder
from RNNencdec.seq2seq import Seq2Seq
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


    def make_batch(self,data):
        pairs                   = [(item['translation']['en'], item['translation']['fr']) for item in data]
        random.shuffle(pairs)
        batches                 = [pairs[i:i + cfg.batch_size] for i in range(0, len(pairs), cfg.batch_size)]

        return batches

    def one_hot_encode_batch(self,batch,vocab_size,word_to_id_eng,word_to_id_fr):
        input_batch = []
        output_batch = []
        for pair in batch: 
            input_batch.append([np.eye(vocab_size)[[word_to_id_eng[n] for n in pair[0]]]])
            output_batch.append([np.eye(vocab_size)[[word_to_id_fr[n] for n in pair[1]]]])

            return input_batch,output_batch

    def train(self, train_data, word_to_id_eng, word_to_id_fr) :
        
        batches         = self.make_batch(train_data)

        encoder         = Encoder(cfg.input_size,
                                  cfg.embedding_size,
                                  cfg.hidden_size)
        
        decoder         = Decoder(cfg.output_size,
                                  cfg.embedding_size,
                                  cfg.hidden_size,
                                  cfg.maxout_size)

        device          = self.device

        model           = Seq2Seq(encoder, 
                                  decoder, 
                                  device).to(device)

        # il faut rectifier l'input size pour le summary
        if not self.quiet_mode : summary(model, input_size=(cfg.batch_size, 3, 224, 224))


        optimizer       = torch.optim.Adadelta(epsilon=1e-6, rho=0.95)
        
        criterion       = nn.CrossEntropyLoss()

        pbar            = pybar(range(cfg.epochs), base_str="training")

        all_loss        = []
        with self.scope:

            for epoch in range(cfg.epochs):

                loss_for_one_epoch = 0

                for batch in batches:
                    if not self.quiet_mode: pbar.__next__()
                    
                    input_batch,output_batch    = self.one_hot_encode_batch(batch,len(word_to_id_eng),word_to_id_eng,word_to_id_fr)
                    
                    input_batch                 = torch.FloatTensor(input_batch)
                    output_batch                = torch.FloatTensor(output_batch)
                    
                    optimizer.zero_grad()
                    
                    output                      = model(input_batch)

                    loss                        = criterion(output, output_batch)
                    loss.backward()
                    
                    optimizer.step()

                    loss_for_one_epoch          += loss.item()

                if not self.quiet_mode      : pbar.set_description(description=f"loss: {loss.cpu().detach().numpy():.4f}")
                
                all_loss                    += [loss_for_one_epoch]

                if epoch % cfg.save_loss_ite == 0 : 
                    np.save(f"{self.dir}/loss.npy", np.array(all_loss))
        
        try : pbar.__next__()
        except StopIteration : pass

