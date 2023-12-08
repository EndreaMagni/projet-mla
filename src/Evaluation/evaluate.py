# Ce fichier permettra d'évaluer notre modèle.

from nltk.translate.bleu_score import sentence_bleu

def compute_bleu_score(prediction, reference) :
     return sentence_bleu(reference, prediction)

class Evaluator : 
     def __init__(self) :