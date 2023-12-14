import os
from Baseline.configuration import config
from easydict import EasyDict
cfg = EasyDict(config)
from datasets import  load_from_disk
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Adapter cette fonction en fonction du format de vos données
        sample = self.data[index]
        return sample


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

        if args.minidataset : 
            data = load_from_disk('projet-mla/mini_dataset')
            test_val_split_ratio = 0.05  # 5% for testing
            train_dataset, val_dataset = data.train_test_split(test_size=test_val_split_ratio).values()
            train_dataset, val_dataset = train_dataset.with_format("torch"), val_dataset.with_format("torch")  
            
            eng_vocab_path                          = 'projet-mla/30k_eng.txt'
            fr_vocab_path                           = 'projet-mla/30k_fr.txt'
            
        else :
            if args.sequence == 50 : 
                train_dataset = load_from_disk('dataset_50/train').select(range(100000))
                test_val_data = load_from_disk('dataset_50/test').select(range(2000))
                cfg.sequence_length = 50
                
            elif args.sequence == 30 : 
                train_dataset = load_from_disk('dataset_30/train').select(range(100000))
                test_val_data = load_from_disk('dataset_30/test').select(range(2000))
                cfg.sequence_length = 30
    
            test_val_split_ratio = 0.5  # 5% for testing
            test_dataset, val_dataset = test_val_data.train_test_split(test_size=test_val_split_ratio).values()
    
            """
            train_custom_dataset = CustomDataset(train_dataset)
            test_custom_dataset = CustomDataset(test_dataset)
            val_custom_dataset = CustomDataset(val_dataset)
    
            num_workers = 0  # Réglez le nombre de travailleurs en fonction de vos besoins
            batch_size = cfg.batch_size  # Réglez la taille de lot en fonction de vos besoins
            train_loader = DataLoader(train_custom_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
            test_loader = DataLoader(test_custom_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
            val_loader = DataLoader(val_custom_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
            """
            eng_vocab_path                          = 'projet-mla/30k_eng.txt'
            fr_vocab_path                           = 'projet-mla/30k_fr.txt'
        with open(eng_vocab_path, 'r') as file  : eng_vocab = [line.strip() for line in file]
        with open(fr_vocab_path, 'r') as file   : fr_vocab = [line.strip() for line in file]
        word_dict_eng                           = { word.strip() : i for i, word in enumerate(eng_vocab)}
        word_dict_fr                            = { word.strip() : i for i, word in enumerate(fr_vocab)}

        BaselineTrainer(quiet_mode=args.q).train(train_dataset,
                                                 val_dataset,
                                                 word_dict_eng, 
                                                 word_dict_fr,
                                                nohup = args.nohup,
                                                mode = args.mode,
                                                seq_len = cfg.sequence_length)

    # if args.evaluate:
    #     from .Evaluation.evaluate import Evaluator
        
    #     Evaluator().evaluate(
    #         all_students_folder=os.path.join(os.getcwd(), "submissions")
    #     )