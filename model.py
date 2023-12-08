import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
Hypothesis = namedtuple('Hypothesis', ['value', 'score'])
class Maxout(nn.Module):
    def __init__(self, in_features, out_features, pool_size):
        super(Maxout, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.pool_size = pool_size
        self.lin = nn.Linear(in_features, out_features * pool_size)

    def forward(self, inputs):
        shape = inputs.size()
        shape = shape[:-1] + (self.out_features, self.pool_size)
        out = self.lin(inputs)
        m, _ = out.view(*shape).max(-1)
        return m

class Attention(nn.Module):
    def __init__(self, enc_hidden_size, dec_hidden_size, attn_dim, maxout_pool_size, deep_output_layers):
        super(Attention, self).__init__()

        self.Wa = nn.Linear(enc_hidden_size, attn_dim, bias=False)
        self.Ua = nn.Linear(dec_hidden_size, attn_dim, bias=False)
        self.va = nn.Parameter(torch.rand(attn_dim, 1))

        self.maxout_pool_size = maxout_pool_size

        # Construct deep output layers
        deep_output_layers_list = [Maxout(enc_hidden_size + dec_hidden_size, dec_hidden_size, maxout_pool_size)]
        for _ in range(1, deep_output_layers):
            deep_output_layers_list.append(Maxout(dec_hidden_size, dec_hidden_size, maxout_pool_size))
        self.deep_output = nn.Sequential(*deep_output_layers_list)

        self.enc_hidden_size = enc_hidden_size
        self.dec_hidden_size = dec_hidden_size

    def forward(self, output, context, mask):
        batch_size = context.shape[0]
        enc_seq_len = context.shape[1]
        dec_seq_len = output.shape[1]

        # Prepare context and output for energy calculation
        context_transformed = self.Wa(context.view(batch_size * enc_seq_len, -1)).view(batch_size, enc_seq_len, -1)
        output_transformed = self.Ua(output)
        output_transformed = output_transformed.unsqueeze(1).expand(-1, enc_seq_len, -1)

        # Calculate energy scores
        e_ij = torch.bmm(context_transformed, self.va.repeat(batch_size, 1, 1)).squeeze(2) + \
               torch.bmm(output_transformed, self.va.repeat(batch_size, 1, 1)).squeeze(2)
        e_ij = torch.tanh(e_ij)

        # Apply mask and calculate attention weights
        e_ij.data.masked_fill_(mask, -float('inf'))
        alpha_ij = F.softmax(e_ij, dim=1)

        # Calculate context vector
        context = torch.bmm(alpha_ij.unsqueeze(1), context).squeeze(1)

        # Compute deep output
        combined = torch.cat((context, output), dim=2)
        deep_output = self.deep_output(combined)

        return deep_output, alpha_ij

class EncoderBiRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderBiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size // 2, bidirectional=True)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        # 合并前向和后向的隐藏状态
        #hidden = self.combine_hidden(hidden)
        return output, hidden

    def initHidden(self):
        # 初始化隐藏状态，为每个方向创建一个隐藏状态
        return torch.zeros(2, 1, self.hidden_size // 2)  # 2 for bidirectional

    def combine_hidden(self, hidden):
        # 假设hidden的形状是[2, batch_size, hidden_size / 2]
        forward_hidden = hidden[0:1, :, :]  # 取出前向隐藏状态
        backward_hidden = hidden[1:2, :, :]  # 取出后向隐藏状态
        combined_hidden = torch.cat((forward_hidden, backward_hidden), 2)  # 在最后一个维度上拼接
        return combined_hidden


class DecoderWithAttention(nn.Module):
    def __init__(self, hidden_size, output_size, context_size, maxout_pool_size, maxout_layers):
        super(DecoderWithAttention, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.context_size = context_size
        self.maxout_pool_size = maxout_pool_size

        # Define the attention mechanism (similar to the one in the Attention class)
        self.attn = Attention(hidden_size, hidden_size, hidden_size, maxout_pool_size, maxout_layers)

        # Define layers for the GRU, embeddings, and maxout network
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size + context_size, hidden_size)

        # Define layers for the maxout network as per the Attention class
        self.maxout = Maxout(hidden_size + context_size, hidden_size, maxout_pool_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input_step, last_hidden, encoder_outputs, mask):
        # Get embedding of current input word
        embedded = self.embedding(input_step).unsqueeze(0)  # Add extra dimension for batch

        # Calculate attention weights and apply to encoder outputs to get the context
        context, attn_weights = self.attn(last_hidden[-1], encoder_outputs, mask)

        # Combine embedded input word and context
        gru_input = torch.cat((embedded, context.unsqueeze(0)), 2)

        # Pass through GRU
        rnn_output, hidden = self.gru(gru_input, last_hidden)

        # Flatten GRU output to pass through maxout network
        maxout_input = rnn_output.squeeze(0)  # Remove extra dimension
        maxout_output = self.maxout(maxout_input)

        # Final output layer (log softmax for numerical stability)
        output = F.log_softmax(self.fc(maxout_output), dim=1)

        # Return output and final hidden state
        return output, hidden, attn_weights
class seq2seq(nn.Module):
    def __init__(self,encoder,decoder):
        super(seq2seq,self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self,x,x_lengths,y,y_lengths):
        context,hid = self.encoder(x,x_lengths)
        output,atten,hid = self.decoder(
            context,x_lengths,
            y,y_lengths,
            hid
        )
        return output,atten

    def beam_search(self,x,x_lengths,y,EOS_id,topk=5,max_length=100):
        encoder_out,hid = self.encoder(x,x_lengths)
        BOS_id = y[0][0].item()
        hypotheses = [[BOS_id]]
        hyp_scores = torch.zeros(len(hypotheses), dtype=torch.float, device=y.device)
        completed_hypotheses = []
        t = 0
        while len(completed_hypotheses) < topk and t < max_length:
            t+=1
            hyp_num = len(hypotheses)
            # 扩展成batch
            exp_src_encodings = encoder_out.expand(hyp_num,encoder_out.shape[1],encoder_out.shape[2])
            exp_x_lengths = x_lengths.expand(hyp_num)
            exp_hid = hid.expand(hid.shape[0],hyp_num,hid.shape[2])
            output_t,atten_t,exp_hid = self.decoder(
                exp_src_encodings,exp_x_lengths,
                torch.tensor(hypotheses).long().to(y.device),torch.ones(hyp_num).long().to(y.device) * t,
                exp_hid
            )
            live_hyp_num = topk - len(completed_hypotheses)

            # 这里把num * vocab 展开来方便取topk
            contiuating_hyp_scores = (hyp_scores.unsqueeze(1).expand(hyp_num,output_t.shape[-1]) + output_t[:,-1,:].squeeze(1)).view(-1)
            top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(contiuating_hyp_scores,k=live_hyp_num)

            # 标记当前概率最大的k个，其是跟在哪个单词的后面
            prev_hyp_ids = top_cand_hyp_pos / (output_t.shape[-1])
            hyp_word_ids = top_cand_hyp_pos % (output_t.shape[-1])

            new_hypotheses = []
            live_hyp_ids = []
            new_hyp_scores = []

            for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(prev_hyp_ids, hyp_word_ids, top_cand_hyp_scores):
                prev_hyp_id = int(prev_hyp_id.item())
                hyp_word_id = int(hyp_word_id.item())
                cand_new_hyp_score = cand_new_hyp_score.item()

                # 将当前最大概率的k个，拼接在正确的prev单词后面
                new_hyp_sent = hypotheses[prev_hyp_id] + [hyp_word_id]
                if hyp_word_id == EOS_id:
                    # 搜寻终止
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

        # 若搜寻了max_len后还没有一个到达EOS则取第一个
        if len(completed_hypotheses) == 0:
            completed_hypotheses.append(Hypothesis(value=hypotheses[0][1:],
                                                   score=hyp_scores[0].item()))
        completed_hypotheses.sort(key=lambda hyp: hyp.score, reverse=True)

        return completed_hypotheses

class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, inputY, target, mask):
        # inputY: batch, seq_len, vocab_size
        # target/mask: batch, seq_len
        inputY = inputY.contiguous().view(-1, inputY.shape[2])
        target = target.contiguous().view(-1, 1)
        mask = mask.contiguous().view(-1, 1)

        # 收集目标单词的对数概率
        log_probs = inputY.gather(1, target)

        # 应用掩码并计算损失
        output = -log_probs * mask

        # 计算平均损失
        return torch.sum(output) / torch.sum(mask)
