
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
Hypothesis = namedtuple('Hypothesis', ['value', 'score'])
class seq2seq(nn.Module):
    # Initialisation du modèle seq2seq avec encodeur et décodeur
    def __init__(self,encoder,decoder):
        super(seq2seq,self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    # La méthode forward pour passer les données à travers le modèle
    def forward(self,x,x_lengths,y,y_lengths):
        context,hid = self.encoder(x,x_lengths)  # Encodage des entrées
        output,atten,hid = self.decoder(
            context,x_lengths,
            y,y_lengths,
            hid
        )  # Décodage des sorties
        return output,atten

    # Méthode beam_search pour la génération de prédictions
    def beam_search(self,x,x_lengths,y,EOS_id,topk=5,max_length=100):
        encoder_out,hid = self.encoder(x,x_lengths)  # Encodage des entrées
        BOS_id = y[0][0].item()  # ID de début de séquence
        hypotheses = [[BOS_id]]  # Initialisation des hypothèses
        hyp_scores = torch.zeros(len(hypotheses), dtype=torch.float, device=y.device)
        completed_hypotheses = []  # Hypothèses complétées
        t = 0
        while len(completed_hypotheses) < topk and t < max_length:
            t+=1
            hyp_num = len(hypotheses)

            exp_src_encodings = encoder_out.expand(hyp_num,encoder_out.shape[1],encoder_out.shape[2])
            exp_x_lengths = x_lengths.expand(hyp_num)
            exp_hid = hid.expand(hid.shape[0],hyp_num,hid.shape[2])
            output_t,atten_t,exp_hid = self.decoder(
                exp_src_encodings,exp_x_lengths,
                torch.tensor(hypotheses).long().to(y.device),torch.ones(hyp_num).long().to(y.device) * t,
                exp_hid
            )  # Décodage des sorties à chaque temps t

            live_hyp_num = topk - len(completed_hypotheses)

            # Calcul des scores pour chaque hypothèse
            contiuating_hyp_scores = (hyp_scores.unsqueeze(1).expand(hyp_num,output_t.shape[-1]) + output_t[:,-1,:].squeeze(1)).view(-1)
            top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(contiuating_hyp_scores,k=live_hyp_num)

            # Sélection des meilleures hypothèses
            prev_hyp_ids = top_cand_hyp_pos / (output_t.shape[-1])
            hyp_word_ids = top_cand_hyp_pos % (output_t.shape[-1])

            new_hypotheses = []
            live_hyp_ids = []
            new_hyp_scores = []

            for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(prev_hyp_ids, hyp_word_ids, top_cand_hyp_scores):
                prev_hyp_id = int(prev_hyp_id.item())
                hyp_word_id = int(hyp_word_id.item())
                cand_new_hyp_score = cand_new_hyp_score.item()

                new_hyp_sent = hypotheses[prev_hyp_id] + [hyp_word_id]
                if hyp_word_id == EOS_id:
                    completed_hypotheses.append(Hypothesis(value=new_hyp_sent[1:-1],
                                                           score=cand_new_hyp_score))
                else:
                    new_hypotheses.append(new_hyp_sent)
                    live_hyp_ids.append(prev_hyp_id)
                    new_hyp_scores.append(cand_new_hyp_score)

            if len(completed_hypotheses) == topk:
                break

            hypotheses = new_hypotheses
            hyp_scores = torch.tensor(new_hyp_scores, dtype=torch.float, device=y.device)

        if len(completed_hypotheses) == 0:
            completed_hypotheses.append(Hypothesis(value=hypotheses[0][1:],
                                                   score=hyp_scores[0].item()))
        completed_hypotheses.sort(key=lambda hyp: hyp.score, reverse=True)

        return completed_hypotheses  # Retourne les meilleures hypothèses
