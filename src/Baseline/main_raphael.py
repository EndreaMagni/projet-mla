import os
from configuration import config
from easydict import EasyDict
cfg = EasyDict(config)
from datasets import  load_from_disk

import argparse

from dataloader import *

from RNNencdec.encoder import Encoder as EncoderENCDEC
from RNNencdec.decoder import Decoder as DecoderENCDEC
from RNNencdec.seq2seq import Seq2Seq as Seq2SeqENCDEC

from RNNsearch.RNNsearch import *
from RNNsearch.Encoder_search import *
from RNNsearch.Decoder_search import *
from RNNsearch.Allignement import *

import torch
import torch.nn as nn

#hyperparameters
batch_size = 80       # Number of sequences in a mini-batch
vocab_size = 30000     # Size of the input vocabulary
hidden_size = 1000     # Number of features in the hidden state
embedding_dim = 620   # Word embedding dimension
maxout_units= 500     # Number of units in the maxout layer
allign_dim=50        # Number of features in the allignment model Tx


if __name__ == "__main__":

    if torch.cuda.is_available():           device = torch.device("cuda")
    elif torch.backends.mps.is_available(): device = torch.device("mps")
    else:                                   device = torch.device("cpu")

    print(device)

    parser = argparse.ArgumentParser(
        usage=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--evaluate",
        action="store_true",
        default=False,
        help="generate the dataset",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        default=False,
        help="generate the dataset",
    )

    parser.add_argument(
        "--minidataset",
        action="store_true",
        default=False,
        help="generate the dataset",
    )

    parser.add_argument(
        "--nohup",
        action="store_true",
        default=False,
        help="use nohup",
    )
    
    parser.add_argument(
        "--q",
        action="store_true",
        default=False,
        help="quiet verbose",
    )
    parser.add_argument(
        "--batch_size",
        nargs="?",
        type=int,
        default=0,
        help="training hyper-parameter: batch-size",
    )

    
    parser.add_argument(
        "--mode",
        nargs="?",
        type=str,
        default="RNNEncDec",
        help="choose the model to train",
    )

    parser.add_argument(
        "--save_name",
        nargs="?",
        type=str,
        default="Nothing",
        help="choose a name to save",
    )

    parser.add_argument(
        "--load_model",
        nargs="?",
        type=str,
        default="Nothing",
        help="choose a path to load the model",
    )

    parser.add_argument(
        "--optimizer",
        nargs="?",
        type=str,
        default="adadelta",
        help="choose the optimizer",
    )
    
    parser.add_argument(
        "--steps",
        nargs="?",
        type=int,
        default=0,
        help="training hyper-parameter: number of optimization steps",
    )

    parser.add_argument(
        "--sequence",
        nargs="?",
        type=int,
        default=50,
        help="training hyper-parameter: number of optimization steps",
    )
    
    parser.add_argument(
        "--lr",
        nargs="?",
        type=float,
        default=0,
        help="training hyper-parameter: initial learning rate",
    )
    

    args = parser.parse_args()

    if int(args.steps) > 0          : cfg.epochs        = int(args.steps)
    if int(args.batch_size) > 0     : cfg.batch_size    = int(args.batch_size)
    if int(args.lr) > 0             : cfg.lr            = int(args.lr)

    


    if args.train:
        from translation import BaselineTrainer

        if args.sequence == 50 : 
            train_data = load_from_disk('/home/linda/dataset_50/train')
            test_val_data = load_from_disk('/home/linda/dataset_50/test')
            cfg.sequence_length = 50
            
        elif args.sequence == 30 : 
            train_data = load_from_disk('/home/linda/dataset_30/train')
            test_val_data = load_from_disk('/home/linda/dataset_30/test')
            cfg.sequence_length = 30


        test_val_split_ratio = 0.5  # 5% for testing
        test_data, val_data = test_val_data.train_test_split(test_size=test_val_split_ratio).values()

        if args.minidataset :
            train_data   = train_data.select(range(1000))
            test_data    = test_data.select(range(1000))
            val_data     = val_data.select(range(1000))

        eng_vocab_path                          = '/home/linda/projet-mla/new30k_eng.txt'
        fr_vocab_path                           = '/home/linda/projet-mla/new30k_fr.txt'
        with open(eng_vocab_path, 'r') as file  : eng_vocab = [line.strip() for line in file]
        with open(fr_vocab_path, 'r') as file   : fr_vocab = [line.strip() for line in file]
        word_dict_eng                           = { word.strip() : i for i, word in enumerate(eng_vocab)}
        word_dict_fr                            = { word.strip() : i for i, word in enumerate(fr_vocab)}


        train_dataset   = Seq2seqData(train_data, word_dict_eng, word_dict_fr)
        val_dataset     = Seq2seqData(val_data, word_dict_eng, word_dict_fr)

        train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=device=="cuda")
        val_data_loader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=device=="cuda")

        if args.mode == "RNNEncDec" : 
            encoder         = EncoderENCDEC(cfg.input_size,
                                            cfg.embedding_size,
                                            cfg.hidden_size)
            
            decoder         = DecoderENCDEC(cfg.output_size,
                                            cfg.embedding_size,
                                            cfg.hidden_size,
                                            cfg.maxout_size)

            model           = Seq2SeqENCDEC(encoder, 
                                            decoder, 
                                            device).to(device)

        if args.mode == "RNNSearch" :
            
            encoder         = Encoder(vocab_size, 
                                      hidden_size, 
                                      embedding_dim, 
                                      device=device)
            
            decoder         = Decoder(vocab_size, 
                                      hidden_size, 
                                      embedding_dim, 
                                      maxout_units, 
                                      device=device)
            
            model           = RNNsearch(encoder, 
                                        decoder, 
                                        device=device).to(device)


        BaselineTrainer(quiet_mode=args.q,save_name = args.save_name).train( model,
                                                                             train_data_loader,
                                                                             val_data_loader,
                                                                             device,
                                                                             nohup = args.nohup,
                                                                             mode = args.mode,
                                                                             seq_len = cfg.sequence_length,
                                                                             load_model = args.load_model,
                                                                             chosed_optimizer = args.optimizer)

    if args.evaluate:
        from .Evaluation.evaluate import Evaluator



        test_val_data = load_from_disk('/home/linda/dataset_50/test')
        
        if torch.cuda.is_available():           device = torch.device("cuda")
        elif torch.backends.mps.is_available(): device = torch.device("mps")
        else:                                   device = torch.device("cpu")
        
        
        test_val_split_ratio = 0.5  # 5% for testing
        test_data, val_data = test_val_data.train_test_split(test_size=test_val_split_ratio).values()
        
        eng_vocab_path                          = '/home/linda/projet-mla/new30k_eng.txt'
        fr_vocab_path                           = '/home/linda/projet-mla/new30k_fr.txt'
        with open(eng_vocab_path, 'r') as file  : eng_vocab = [line.strip() for line in file]
        with open(fr_vocab_path, 'r') as file   : fr_vocab = [line.strip() for line in file]
        word_dict_eng                           = { word.strip() : i for i, word in enumerate(eng_vocab)}
        word_dict_fr                            = { word.strip() : i for i, word in enumerate(fr_vocab)}
        
        test_dataset        = Seq2seqData(test_data, word_dict_eng, word_dict_fr)
        
        test_dataloader     = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=True, pin_memory=device=="cuda")

        Evaluator().evaluate(
            all_students_folder=os.path.join(os.getcwd(), "submissions")
        )