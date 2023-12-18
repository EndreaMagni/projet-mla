# Ce fichier permettra d'évaluer notre modèle.

from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm 
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch
plt.style.use('ggplot')

def compute_bleu_score(prediction, reference) :
    # i = 0
    # ref = ref.remove("PAD"))
    return [sentence_bleu([' '.join(reference[i,:]).replace('PAD','')], ' '.join(prediction[i,:]).replace('PAD','')) for i in range(prediction.shape[0])]

class Evaluator : 
    def __init__(self) :
        pass
    
    def predict(self, model_name, model, input_batch):
        if model_name == "RNNEncDec"  : output = model(input_batch)
        else : output, _ = model(input_batch)

        output = output.detach().cpu().numpy()
        output = np.argmax(output, axis = 2)

        np.array([[self.word_dict_fr_reverse[i] for i in output.tolist()[i]] for i in range(output.shape[0])])

        return output
    

    def evaluate(self,
                 model_search_30,model_search_50,model_encdec_30,model_encdec_50,
                 word_dict_eng_reverse, word_dict_fr_reverse,
                 test_data_loader,
                 device) :
        
        self.word_dict_fr_reverse = word_dict_fr_reverse
        bleu_search_30 = []    
        bleu_search_50 = []
        bleu_encdec_30 = []
        bleu_encdec_50 = []


        sizes = []
        n_batch = 0
        with torch.no_grad():
            for input_batch, output_batch,size in tqdm(test_data_loader) :

                n_batch += 1
                input_batch, output_batch= input_batch.to(device), output_batch.to(device)

                #intput_batch=input_batch[input_batch!= 0]
                #output_batch=output_batch[output_batch!= 0]
                # On récupère les sorties des

                output_search_30 = self.predict("RNNSearch", model_search_30,input_batch)
                output_search_50 = self.predict("RNNSearch", model_search_50,input_batch)
                output_encdec_30 = self.predict("RNNSearch", output_encdec_30,input_batch)
                output_encdec_50 = self.predict("RNNSearch", output_encdec_50,input_batch)

                output_batch = np.array([[word_dict_fr_reverse[i] for i in output_batch.tolist()[i]] for i in range(output_batch.shape[0])])

                bleu_search_30 += compute_bleu_score(output_search_30,output_batch)    
                bleu_search_50 += compute_bleu_score(output_search_50,output_batch)
                bleu_encdec_30 += compute_bleu_score(output_encdec_30,output_batch)
                bleu_encdec_50 += compute_bleu_score(output_encdec_50,output_batch)

                sizes += size.detach().cpu().numpy().tolist()


        score_search_30 = {i : [] for i in range(81)}
        score_search_50 = {i : [] for i in range(81)}
        score_encdec_30 = {i : [] for i in range(81)}
        score_encdec_50 = {i : [] for i in range(81)}

        coeff = [0] *81
        for i in range(len(sizes)) :
            score_search_30[sizes[i]] += [bleu_search_30[i]]
            score_search_50[sizes[i]] += [bleu_search_50[i]]
            score_encdec_30[sizes[i]] += [bleu_encdec_30[i]]
            score_encdec_50[sizes[i]] += [bleu_encdec_50[i]]
            coeff[sizes[i]] += 1
                

                
        plt.figure()
        plt.plot([np.median(score_search_30[i])*100 for i in range(81)],label = "RNNSearch 30")
        plt.plot([np.median(score_search_50[i])*100 for i in range(81)],label = "RNNSearch 50")
        plt.plot([np.median(score_encdec_30[i])*100 for i in range(81)],label = "RNNEncDec 30")
        plt.plot([np.median(score_encdec_50[i])*100 for i in range(81)],label = "RNNEncDec 50")

        plt.xlabel("Sequence length")
        plt.ylabel("BLEU Score (%)")
        plt.legend()
        plt.xlim([0,60])
        plt.savefig('Evaluation/figures/bleu_score_median.png')
