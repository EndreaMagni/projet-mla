import os
from Baseline.configuration import config
from easydict import EasyDict
cfg = EasyDict(config)
from datasets import  load_from_disk
from tqdm import tqdm
import argparse

from GenerateData.dataloader import *

from Baseline.RNNencdec.encoder import Encoder as EncoderENCDEC
from Baseline.RNNencdec.decoder import Decoder as DecoderENCDEC
from Baseline.RNNencdec.seq2seq import Seq2Seq as Seq2SeqENCDEC

from Baseline.RNNsearch.RNNsearch import *
from Baseline.RNNsearch.Encoder_search import *
from Baseline.RNNsearch.Decoder_search import *
from Baseline.RNNsearch.Allignement import *

import torch
import torch.nn as nn



if __name__ == "__main__":
    path = os.getcwd().replace("projet-mla/src", "")

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
        from Baseline.translation import BaselineTrainer

        if args.sequence == 50 : 
            train_data = load_from_disk(path+'dataset_50/train')
            test_val_data = load_from_disk(path+'dataset_50/test')
            cfg.sequence_length = 50
            
        elif args.sequence == 30 : 
            train_data = load_from_disk(path+'dataset_30/train')
            test_val_data = load_from_disk(path+'dataset_30/test')
            cfg.sequence_length = 30


        test_val_split_ratio = 0.5  # 5% for testing
        test_data, val_data = test_val_data.train_test_split(test_size=test_val_split_ratio).values()

        if args.minidataset :
            train_data   = train_data.select(range(3000))
            test_data    = test_data.select(range(300))
            val_data     = val_data.select(range(300))

        eng_vocab_path                          = path+'projet-mla/new30k_eng.txt'
        fr_vocab_path                           = path+'projet-mla/new30k_fr.txt'
        with open(eng_vocab_path, 'r') as file  : eng_vocab = [line.strip() for line in file]
        with open(fr_vocab_path, 'r') as file   : fr_vocab = [line.strip() for line in file]
        word_dict_eng                           = { word.strip() : i for i, word in enumerate(eng_vocab)}
        word_dict_fr                            = { word.strip() : i for i, word in enumerate(fr_vocab)}


        train_dataset   = Seq2seqData(train_data, word_dict_eng, word_dict_fr)
        val_dataset     = Seq2seqData(val_data, word_dict_eng, word_dict_fr)

        train_data_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, pin_memory=device=="cuda")
        val_data_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, pin_memory=device=="cuda")

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
            
            encoder         = Encoder(cfg.vocabulary_size, 
                                      cfg.hidden_size, 
                                      cfg.embedding_size, 
                                      device=device)
            
            decoder         = Decoder(cfg.vocabulary_size, 
                                      cfg.hidden_size, 
                                      cfg.embedding_size, 
                                      cfg.maxout_size, 
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
                                                                             chosed_optimizer = args.optimizer,
                                                                           lr = args.lr)

    if args.evaluate:


        from Evaluation.evaluate import Evaluator

        test_data = load_from_disk('/home/linda/test_dataset_multi').select(range(10000))
        #test_val_data = load_from_disk('/home/linda/dataset_30/test')
        #test_val_split_ratio = 0.5  # 5% for testing
        #test_data, val_data = test_val_data.train_test_split(test_size=test_val_split_ratio).values()

        if torch.cuda.is_available():           device = torch.device("cuda")
        elif torch.backends.mps.is_available(): device = torch.device("mps")
        else:                                   device = torch.device("cpu")
                
        eng_vocab_path                          = path+'projet-mla/new30k_eng.txt'
        fr_vocab_path                           = path+'projet-mla/new30k_fr.txt'
        with open(eng_vocab_path, 'r') as file  : eng_vocab = [line.strip() for line in file]
        with open(fr_vocab_path, 'r') as file   : fr_vocab = [line.strip() for line in file]
        word_dict_eng                           = { word.strip() : i for i, word in enumerate(eng_vocab)}
        word_dict_fr                            = { word.strip() : i for i, word in enumerate(fr_vocab)}
        
        word_dict_eng_reverse                   = { i : word.strip() for i, word in enumerate(eng_vocab)}
        word_dict_fr_reverse                    = { i : word.strip() for i, word in enumerate(fr_vocab)}

        test_dataset        = Seq2seqDataEval(test_data, word_dict_eng, word_dict_fr)
        
        test_dataloader     = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=True, pin_memory=device=="cuda")

        # Initialisation des modèles pour RNNSearch
        
        encoder1         = Encoder(cfg.vocabulary_size, 
                                  cfg.hidden_size, 
                                  cfg.embedding_size, 
                                  device=device)
        
        decoder1         = Decoder(cfg.vocabulary_size, 
                                  cfg.hidden_size, 
                                  cfg.embedding_size, 
                                  cfg.maxout_size, 
                                  device=device)
        
        model_search_50    = RNNsearch(encoder1, 
                                    decoder1, 
                                    device=device).to(device)

        encoder2         = Encoder(cfg.vocabulary_size, 
                                  cfg.hidden_size, 
                                  cfg.embedding_size, 
                                  device=device)
        
        decoder2         = Decoder(cfg.vocabulary_size, 
                                  cfg.hidden_size, 
                                  cfg.embedding_size, 
                                  cfg.maxout_size, 
                                  device=device)
        
        model_search_30    = RNNsearch(encoder2, 
                                    decoder2, 
                                    device=device).to(device)

        # Initialisation des modèles pour RNNEncDec
        
        encoder3         = Encoder(cfg.vocabulary_size, 
                                  cfg.hidden_size, 
                                  cfg.embedding_size, 
                                  device=device)
        
        decoder3         = Decoder(cfg.vocabulary_size, 
                                  cfg.hidden_size, 
                                  cfg.embedding_size, 
                                  cfg.maxout_size, 
                                  device=device)
        
        model_encdec_30           = RNNsearch(encoder, 
                                              decoder, 
                                              device=device).to(device)

        encoder4         = Encoder(cfg.vocabulary_size, 
                                  cfg.hidden_size, 
                                  cfg.embedding_size, 
                                  device=device)
        
        decoder4         = Decoder(cfg.vocabulary_size, 
                                  cfg.hidden_size, 
                                  cfg.embedding_size, 
                                  cfg.maxout_size, 
                                  device=device)
        
        model_encdec_50           = RNNsearch(encoder, 
                                              decoder, 
                                              device=device).to(device)

        # On charge les 
        model_search_50.load_state_dict(torch.load('Models/RNN_search_50/best_model.pth', 
                                                map_location=torch.device(device))["model_state_dict"])

        model_search_30.load_state_dict(torch.load('Models/RNN_search_30/best_model.pth', 
                                                map_location=torch.device(device))["model_state_dict"])

        model_encdec_50.load_state_dict(torch.load('Models/RNN_encdec_50/best_model.pth', 
                                                map_location=torch.device(device))["model_state_dict"])

        model_encdec_30.load_state_dict(torch.load('Models/RNN_encdec_30/best_model.pth', 
                                                map_location=torch.device(device))["model_state_dict"])


        model_search_50.eval()
        model_search_30.eval()
        model_encdec_50.eval()
        model_encdec_30.eval()

        Evaluator().evaluate(model_search_30, model_search_50,model_encdec_30,model_encdec_50,
                             word_dict_eng_reverse, word_dict_fr_reverse,
                             test_dataloader, device)
