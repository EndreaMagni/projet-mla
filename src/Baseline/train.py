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

def train(model,train_data,word_to_id_eng,word_to_id_fr,batch_size,learning_rate,epochs,print_every=1):
    batches=make_batch(train_data,batch_size)
    # Initialize Adadelta optimizer with the given epsilon and rho values
    optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate, rho=0.95, eps=1e-06)

    criterion = nn.CrossEntropyLoss()
    batches= batches[:2]
    for epoch in range(epochs):
        
        for batch in batches:
            input_batch,output_batch = one_hot_encode_batch(batch,len(word_to_id_eng),word_to_id_eng,word_to_id_fr)
            input_batch = torch.FloatTensor(input_batch)
            print(input_batch.shape)
            output_batch = torch.FloatTensor(output_batch)
            optimizer.zero_grad()
            output,attention_weights  = model(input_batch,batch_size)
            loss = criterion(output, output_batch)
            loss.backward()
            optimizer.step()
        if epoch % print_every == 0:
            print('Epoch: ', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))
            
    print('Training Finished')
          
     

    

