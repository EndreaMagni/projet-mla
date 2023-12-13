import os
from typing import Any

import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.init as init

from torchtext.data import BucketIterator
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


    def make_batch(self,data):
        pairs                   = [(item['translation']['en'], item['translation']['fr']) for item in data]
        random.shuffle(pairs)
        batches                 = [pairs[i:i + cfg.batch_size] for i in range(0, len(pairs), cfg.batch_size)]

        return batches

    def one_hot_encode_batch(self,batch,vocab_size,word_to_id_eng,word_to_id_fr):
        input_batch = []
        output_batch = []
        for i,pair in enumerate(batch): 
            input_batch.append([np.eye(vocab_size)[[word_to_id_eng[n] for n in pair[0]]]])
            output_batch.append([np.eye(vocab_size)[[word_to_id_eng[n] for n in pair[0]]]])

        return input_batch,output_batch
    


    def different_approach(self, train_data, valid_data, test_data):
        self.train_iterator, self.valid_iterator, self.test_iterator = BucketIterator.splits(
                (train_data, valid_data, test_data), 
                batch_size = cfg.batch_size, 
                device = self.device)

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

        model.apply(self.Init_weights)
        
        # il faut rectifier l'input size pour le summary
        # if not self.quiet_mode : summary(model, input_size=(cfg.batch_size, cfg.sequence_length))


        optimizer       = torch.optim.Adadelta(model.parameters())
        
        criterion       = nn.CrossEntropyLoss()

        pbar            = pybar(range(cfg.epochs), base_str="training")

        all_loss        = []
        with self.scope:

            for epoch in range(cfg.epochs):

                loss_for_one_epoch = 0
                print(np.array(batches).shape)
                
                for batch in [batches[0]]:
                    if not self.quiet_mode: pbar.__next__()

                    input_batch,output_batch    = self.one_hot_encode_batch(batch,len(word_to_id_eng),word_to_id_eng,word_to_id_fr)
                    
                    input_batch                 = np.array(input_batch)
                    input_batch                 = torch.from_numpy(input_batch)
                    input_batch                 = input_batch.long()
                    
                    output_batch                = np.array(output_batch)
                    output_batch                = torch.from_numpy(output_batch)
                    output_batch                = output_batch.long()

                    print(input_batch.size(),output_batch.size())

                    optimizer.zero_grad()
                    
                    output                      = model(input_batch, output_batch)

                    loss                        = criterion(output, output_batch)
                    loss.backward()
                    print("azd")
                    optimizer.step()

                    loss_for_one_epoch          += loss.item()
                    print("b")
                if not self.quiet_mode      : pbar.set_description(description=f"loss: {loss.cpu().detach().numpy():.4f}")
                
                all_loss                    += [loss_for_one_epoch]

                if epoch % cfg.save_loss_ite == 0 and epoch != 0 : 
                    np.save(f"{self.dir}/loss.npy", np.array(all_loss))

                if epoch % cfg.save_model_ite == 0 and epoch != 0 :
                    torch.save({'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict()}, 
                                f"{self.dir}/model_{epoch}.pth")

        
        pbar.__next__()
        # except StopIteration : pass

