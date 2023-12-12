import torch
import spacy

def translate_sentence_beam_search(sentence, src_field, trg_field, model, device, max_len=50, beam_size=3):

    model.eval()

    if isinstance(sentence, str):
        nlp = spacy.load('de')
        tokens = [token.text.lower() for token in nlp(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    tokens = [src_field.init_token] + tokens + [src_field.eos_token]

    src_indexes = [src_field.vocab.stoi[token] for token in tokens]

    src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)

    src_len = torch.LongTensor([len(src_indexes)])

    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor, src_len)

    mask = model.create_mask(src_tensor)

    beam_candidates = [(0, [trg_field.vocab.stoi[trg_field.init_token]], hidden, [])]
    for _ in range(max_len):
        new_candidates = []
        for cum_log_prob, trg_indexes, hidden, candidate_attentions in beam_candidates:
            trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)

            output, hidden, attention = model.decoder(trg_tensor, hidden, encoder_outputs, mask)

            log_probs = torch.log_softmax(output, dim=1)
            top_log_probs, top_indexes = log_probs.topk(beam_size)

            for log_prob, index in zip(top_log_probs.squeeze(0), top_indexes.squeeze(0)):
                new_cum_log_prob = cum_log_prob + log_prob.item()
                new_indexes = trg_indexes + [index.item()]
                new_attentions = candidate_attentions + [attention.squeeze(1)]
                new_candidates.append((new_cum_log_prob, new_indexes, hidden, new_attentions))

        beam_candidates = sorted(new_candidates, key=lambda x: x[0], reverse=True)[:beam_size]



        if len(beam_candidates) == 0:
            break

    _, trg_indexes, _, best_attentions = max(beam_candidates, key=lambda x: x[0])

    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]

    # 注意力张量的处理
    attentions_tensor = torch.cat(best_attentions, dim=0)

    return trg_tokens[1:], attentions_tensor
