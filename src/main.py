import os
from Baseline.configuration import config
from easydict import EasyDict
cfg = EasyDict(config)
from datasets import  load_from_disk

if __name__ == "__main__":
    import argparse

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
        "--steps",
        nargs="?",
        type=int,
        default=0,
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

        train_data = load_from_disk('/Users/travail/Desktop/mini_dataset')


        eng_vocab_path                          = '/Users/travail/Documents/GitHub/projet-mla/30k_eng.txt'
        fr_vocab_path                           = '/Users/travail/Documents/GitHub/projet-mla/30k_fr.txt'
        with open(eng_vocab_path, 'r') as file  : eng_vocab = [line.strip() for line in file]
        with open(fr_vocab_path, 'r') as file   : fr_vocab = [line.strip() for line in file]
        word_dict_fr                            = {i: word.strip() for i, word in enumerate(fr_vocab)}
        word_dict_eng                           = {i: word.strip() for i, word in enumerate(eng_vocab)}

        BaselineTrainer(quiet_mode=args.q).train(train_data, 
                                                 word_dict_eng, 
                                                 word_dict_fr)

    # if args.evaluate:
    #     from .Evaluation.evaluate import Evaluator
        
    #     Evaluator().evaluate(
    #         all_students_folder=os.path.join(os.getcwd(), "submissions")
    #     )