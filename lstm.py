#!/usr/bin/python3

from __future__ import division
import time
import sys
import pickle
from datetime import timedelta
import numpy as np
import torch as th
import torch.autograd as ag
from torch import nn, optim
import matplotlib.pyplot as plt

from data import load_B_task_dataset, process_messages, generate_vocabulary, transform_to_indices, create_vocabulary_embeddings

if len(sys.argv) not in (3, 4): #specify input parameters
    sys.stderr.write('Usage : {} <B_train_tweets> <B_test_tweets> [state_file]'.format(sys.argv[0]))
    sys.exit(-1)

oov = "<<<<<<<OOV>>>>>>>" #unknown words

print('Loading datasets...')

train_messages, train_polarities = load_B_task_dataset(sys.argv[1]) #load training data
test_messages, test_polarities = load_B_task_dataset(sys.argv[2]) #load test data

negative_count = len(list(filter(lambda x: x == 0, train_polarities))) #counting number of negative tweets
neutral_count = len(list(filter(lambda x: x == 1, train_polarities)))#counting number of neutral tweets
positive_count = len(list(filter(lambda x: x == 2, train_polarities)))#counting number of positive tweets

p_train_messages = process_messages(train_messages) #preprocess training tweets
p_train_messages.append([oov]) #add the unknown word
vocab = generate_vocabulary(p_train_messages) #generate vocabulary from training data
p_test_messages = process_messages(test_messages) #preprocess test tweets
for msg in p_test_messages:
    for i in range(len(msg)):
        if msg[i] not in vocab:
            msg[i] = oov #replace every out of vocabulary test word by the unknown word
vocabulary_embeddings, embeddings_indices = create_vocabulary_embeddings(vocab, p_train_messages) #create embedding for each word of the vocabulary
p_train_messages.pop() #remove the unknown word from the training set

message_length = max(len(msg) for msg in p_train_messages) #get the length of the longest tweet
train_messages_indices = transform_to_indices(p_train_messages, embeddings_indices) #give an index for each word of training set
test_messages_indices = transform_to_indices(p_test_messages, embeddings_indices) #give an index for each word of test set
train_messages_np = np.empty((len(train_messages), message_length), dtype=np.int)
test_messages_np = np.empty((len(test_messages), message_length), dtype=np.int)
#replace every word in train/test by its index : short tweets are replicated
for i in range(train_messages_np.shape[0]):
    train_messages_np[i] = ((message_length // len(train_messages_indices[i]) + 1) * train_messages_indices[i])[:message_length]
for i in range(test_messages_np.shape[0]):
    test_messages_np[i] = ((message_length // len(test_messages_indices[i]) + 1) * test_messages_indices[i])[:message_length]

embeddings = nn.Embedding(*vocabulary_embeddings.shape) #construct the embeddings table
embeddings.weight = nn.Parameter(th.FloatTensor(vocabulary_embeddings)) #fill the embeddings with the values learned from word2vec on the training set

train_messages_var = ag.Variable(th.LongTensor(train_messages_np))
test_messages_var = ag.Variable(th.LongTensor(test_messages_np))

hidden = 200 #the size of the hidden state
layers = 1 #number of layers
dropout_factor = 0.2 #probability of dropping an output of a unit
print('Constructing LSTM model : layers count = {}, hidden size = {}'.format(layers, hidden))

lstm = nn.LSTM(input_size=vocabulary_embeddings.shape[1], hidden_size=hidden, num_layers=layers, batch_first=True, dropout=dropout_factor) #build an LSTM layer
linear = nn.Linear(message_length*hidden, 3) #build a fully connected layer
dropout_linear = nn.Dropout(dropout_factor) #apply a dropout
model = nn.Sequential(lstm, dropout_linear, linear) #pack the model
print(model)
optimizer = optim.Adam(model.parameters()) #Adam optimizer used
criterion = nn.CrossEntropyLoss(weight=th.FloatTensor([1 - negative_count / len(train_messages), 1 - neutral_count / len(train_messages), 1 - positive_count / len(train_messages)])) #cross entropy loss with classes balancing

train_y = ag.Variable(th.LongTensor(train_polarities))
test_y = ag.Variable(th.LongTensor(test_polarities))

epochs = 4000

figure = plt.figure('Twitter Sentiment analysis using LSTM | BELOUADAH and BOUHAHA', figsize=(20, 7), tight_layout=True) #create the figure of the plots
loss_axes = figure.add_subplot(1, 2, 1, title='Cross-entropy loss') #losses plot
accuracy_axes = figure.add_subplot(1, 2, 2, title='Accuracy') #accuracy plot
accuracy_axes.set_ylim(0, 1)
plt.ion()
plt.show()
if len(sys.argv) == 4: #if there is a saved state for the model
    print('Loading old state from {}'.format(sys.argv[3]))
    with open(sys.argv[3], 'rb') as state_file:
        state = pickle.load(state_file) #load the state
    model.load_state_dict(state['parameters']) #and use its parameters
    loss_axes.plot(state['train_losses'], color='blue', label='Train')
    loss_axes.plot(state['test_losses'], color='green', label='Test')
    accuracy_axes.plot(state['train_accuracies'], color='blue', label='Train')
    accuracy_axes.plot(state['test_accuracies'], color='green', label='Test')
else:
    print('No state loaded, initialising a new model')
    state = dict((x, []) for x in ['test_losses', 'test_accuracies', 'train_losses', 'train_accuracies']) #generate a new empty state
plt.pause(0.001)
print('Starting train...')
start_time = time.time()
try:
    for epoch in range(len(state['train_accuracies']), epochs):
        model.train(mode=False)  # Test mode : dropout disabled
        # Tesing part
        test_emb = embeddings(test_messages_var)
        test_out, hc = lstm(test_emb)
        test_out = linear(test_out.contiguous().view(-1, message_length * hidden))
        state['test_losses'].append(criterion(test_out, test_y).data.numpy()[0])
        state['test_accuracies'].append(np.mean((th.max(test_out, 1)[1] == test_y).data.numpy()))
        if len(state['test_accuracies']) >= 2:
            if state['test_accuracies'][-1] > state['test_accuracies'][-2]:
                state['parameters'] = model.state_dict()  # Save parameters if the accuracy became better

        model.train(mode=True)  # Train mode : dropout enabled
        # Training part
        model.zero_grad()
        train_emb = embeddings(train_messages_var)
        train_out, hc = lstm(train_emb)
        train_out = linear(dropout_linear(train_out.contiguous().view(-1, message_length * hidden)))
        loss = criterion(train_out, train_y)
        state['train_losses'].append(loss.data.numpy()[0])
        state['train_accuracies'].append(np.mean((th.max(train_out, 1)[1] == train_y).data.numpy()))
        loss.backward()
        optimizer.step()

        loss_axes.plot(state['train_losses'], color='blue', label='Train')
        loss_axes.plot(state['test_losses'], color='green', label='Test')
        accuracy_axes.plot(state['train_accuracies'], color='blue', label='Train')
        accuracy_axes.plot(state['test_accuracies'], color='green', label='Test')
        plt.pause(0.001)
        print('{}/{} {} | Train loss = {:.4f} acc = {:.2%} | Test loss = {:.4f} acc = {:.2%}'
              .format(epoch + 1, epochs, timedelta(seconds=round(time.time() - start_time)), state['train_losses'][-1],
                      state['train_accuracies'][-1], state['test_losses'][-1], state['test_accuracies'][-1]))
except KeyboardInterrupt: #if the user press Ctrl+C
    pass
finally:
    plt.ioff()
    print('Elapsed training time : {}'.format(timedelta(seconds=round(time.time() - start_time))))
    import os.path as p
    with open(p.join('states', 'LSTM-{}x{}.pkl'.format(layers, hidden)), 'wb') as state_file:
        pickle.dump(state, state_file) #save the state to a file
