

from tqdm import trange,tqdm
from collections import Counter

import torch

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter
from transformers import AdamW,get_linear_schedule_with_warmup
def evaluate(args,model, data,loss_fn):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    total_num_words = total_loss = 0.
    eval_iteration = tqdm(data, desc='eval iteration')
    with torch.no_grad():
        for it, (mb_x, mb_x_len, mb_y, mb_y_len) in enumerate(eval_iteration):
            mb_x = torch.from_numpy(mb_x).to(device).long()
            mb_x_len = torch.from_numpy(mb_x_len).to(device).long()
            mb_input = torch.from_numpy(mb_y[:, :-1]).to(device).long()
            mb_output = torch.from_numpy(mb_y[:, 1:]).to(device).long()
            mb_y_len = torch.from_numpy(mb_y_len-1).to(device).long()
            mb_y_len[mb_y_len<=0] = 1

            mb_pred, attn = model(mb_x, mb_x_len, mb_input, mb_y_len)

            mb_out_mask = torch.arange(mb_y_len.max().item(), device=args.device)[None, :] < mb_y_len[:, None]
            mb_out_mask = mb_out_mask.float()

            loss = loss_fn(mb_pred, mb_output, mb_out_mask)

            num_words = torch.sum(mb_y_len).item()
            total_loss += loss.item() * num_words
            total_num_words += num_words
    print("Evaluation loss", total_loss/total_num_words)
    return total_loss/total_num_words
def train(model, data,loss_fn,eval_data,num_epoch,learnning_rate,warmup_steps):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    LOG_FILE = "translation_model.log"
    tb_writer = SummaryWriter('./runs')

    t_total = num_epoch * len(data)
    optimizer = AdamW(model.parameters(), lr=learnning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=t_total)
    global_step = 0
    total_num_words = total_loss = 0.
    logg_loss = 0.
    logg_num_words = 0.
    val_losses = []
    train_iterator = trange(num_epoch,desc='epoch')
    for epoch in train_iterator:
        model.train()
        epoch_iteration = tqdm(data, desc='iteration')
        should_continue_epoch = True
        for it, (mb_x, mb_x_len, mb_y, mb_y_len) in enumerate(epoch_iteration):

            mb_x = torch.from_numpy(mb_x).to(device).long()
            mb_x_len = torch.from_numpy(mb_x_len).to(device).long()

            mb_input = torch.from_numpy(mb_y[:, :-1]).to(device).long()
            mb_output = torch.from_numpy(mb_y[:, 1:]).to(device).long()
            mb_y_len = torch.from_numpy(mb_y_len - 1).to(device).long()

            mb_y_len[mb_y_len <= 0] = 1

            mb_pred, attn = model(mb_x, mb_x_len, mb_input, mb_y_len)
            mb_out_mask = torch.arange(mb_y_len.max().item(), device=device)[None, :] < mb_y_len[:, None]

            mb_out_mask = mb_out_mask.float()

            loss = loss_fn(mb_pred, mb_output, mb_out_mask)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

            optimizer.step()
            scheduler.step()

            global_step += 1
            num_words = torch.sum(mb_y_len).item()

            total_loss += loss.item() * num_words

            total_num_words += num_words
            if (it+1) % 100 == 0:
                current_loss = (total_loss - logg_loss) / (total_num_words-logg_num_words)
                logg_num_words = total_num_words
                logg_loss = total_loss
                if current_loss > 500 and last_loss - current_loss < 0.01:
                    should_continue_epoch = False
                    print("Loss not decreasing significantly. Stopping this epoch early.")
                    break

                last_loss = current_loss

                with open(LOG_FILE, "a") as fout:
                    fout.write("epoch: {}, iter: {}, loss: {},learn_rate: {}\n".format(epoch, it, current_loss,
                                                                                       scheduler.get_lr()[0]))
                print("epoch: {}, iter: {}, loss: {}, learning_rate: {}".format(epoch, it, current_loss,
                                                                                scheduler.get_lr()[0]))
                tb_writer.add_scalar("learning_rate", scheduler.get_lr()[0], global_step)
                tb_writer.add_scalar("loss", current_loss, global_step)

        print("Epoch", epoch, "Training loss", total_loss / total_num_words)
        eval_loss = evaluate( model, eval_data, loss_fn)
        if eval_loss < best_loss:
            best_loss = eval_loss
            patience_counter = 0
            torch.save(model.state_dict(), "translate-best.th")
        else:
            patience_counter += 1

        if patience_counter >= 5:
            print(f"Stopping training early. Best validation loss: {best_loss}")
            break

        with open(LOG_FILE, "a") as fout:
            fout.write("===========" * 20)
            fout.write("EVALUATE: epoch: {}, loss: {}\n".format(epoch, eval_loss))
        if len(val_losses) == 0 or eval_loss < min(val_losses):

            print("best model, val loss: ", eval_loss)
            torch.save(model.state_dict(), "translate-best.th")
        val_losses.append(eval_loss)
        if not should_continue_epoch:
            continue