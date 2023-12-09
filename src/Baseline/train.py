import numpy as np
import random
import torch
import torch.nn as nn

def make_batch(data,batch_size):
    # creating pairs
    pairs = [(item['translation']['en'], item['translation']['fr']) for item in data]
    random.shuffle(pairs)
    batches = [pairs[i:i + batch_size] for i in range(0, len(pairs), batch_size)]

    return batches

def one_hot_encode_batch(batch,vocab_size,word_to_id_eng,word_to_id_fr):
    input_batch = []
    output_batch = []
    for pair in batch: 
        input_batch.append([np.eye(vocab_size)[[word_to_id_eng[n] for n in pair[0]]]])
        output_batch.append([np.eye(vocab_size)[[word_to_id_fr[n] for n in pair[1]]]])

        return input_batch,output_batch

def train(model,train_data,word_to_id_eng,word_to_id_fr,learning_rate,batch_size,epochs):
    # Initialize Adadelta optimizer with the given epsilon and rho values
    optimizer = torch.optim.Adadelta(epsilon=1e-6, rho=0.95)

    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        batches = make_batch(train_data,batch_size)
        for batch in batches:
            input_batch,output_batch = one_hot_encode_batch(batch,len(word_to_id_eng),word_to_id_eng,word_to_id_fr)
            input_batch = torch.FloatTensor(input_batch)
            output_batch = torch.FloatTensor(output_batch)
            optimizer.zero_grad()
            output, hidden = model(input_batch)
            loss = criterion(output, output_batch)
            loss.backward()
            optimizer.step()
        print('Epoch: ', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))


