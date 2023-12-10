import torch.nn as nn
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, training_iteration, max_iterations, teacher_forcing_ratio=0.5):
        # Debugging: Print current teacher forcing ratio
        current_teacher_forcing_ratio = teacher_forcing_ratio * (1 - training_iteration / max_iterations)
        print(f"Current Teacher Forcing Ratio: {current_teacher_forcing_ratio}")

        trg_len, batch_size = trg.shape
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # Debugging: Try-catch block for encoder
        try:
            encoder_outputs, hidden = self.encoder(src)
        except Exception as e:
            print("Error in encoder:", e)

        input = trg[0, :]

        for t in range(1, trg_len):
            try:
                output, hidden = self.decoder(input, hidden, encoder_outputs)
                outputs[t] = output
            except Exception as e:
                print(f"Error at time step {t} in decoder:", e)
                break

            teacher_force = random.random() < current_teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1

        return outputs
