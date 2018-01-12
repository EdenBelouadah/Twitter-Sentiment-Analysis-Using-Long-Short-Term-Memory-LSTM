from __future__ import division
import time
import sys
import numpy as np
import torch as th
import torch.autograd as ag
from torch import nn, optim
import matplotlib.pyplot as plt

from data import load_B_task_dataset, process_messages, generate_vocabulary, transform_to_indices, create_vocabulary_embeddings

messages, polarities = load_B_task_dataset(sys.argv[1])

negative_count = len(list(filter(lambda x: x == 0, polarities)))
neutral_count = len(list(filter(lambda x: x == 1, polarities)))
positive_count = len(list(filter(lambda x: x == 2, polarities)))
print(negative_count, neutral_count, positive_count)

p_messages = process_messages(messages)
vocab = generate_vocabulary(p_messages)

vocabulary_embeddings,  embeddings_indices = create_vocabulary_embeddings(vocab, p_messages)

message_length = max(len(msg) for msg in p_messages)
messages_indices = transform_to_indices(p_messages, embeddings_indices)
messages_np = np.empty((len(messages), message_length), dtype=np.int)
for i in range(messages_np.shape[0]):
    messages_np[i] = ((message_length//len(messages_indices[i])+1)*messages_indices[i])[:message_length]

embeddings = nn.Embedding(*vocabulary_embeddings.shape)
embeddings.weight = nn.Parameter(th.FloatTensor(vocabulary_embeddings))

messages_var = ag.Variable(th.LongTensor(messages_np))

hidden = 100
layers = 1

lstm = nn.LSTM(input_size=vocabulary_embeddings.shape[1], hidden_size=hidden, num_layers=layers, batch_first=True)
linear = nn.Linear(hidden, 3)
model = nn.Sequential(lstm, linear)
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(weight=th.FloatTensor([1-negative_count/len(messages), 1-neutral_count/len(messages), 1-positive_count/len(messages)]))

y = ag.Variable(th.LongTensor(polarities))

epochs = 500

figure = plt.figure('Learning from structured data', figsize=(20, 7), tight_layout=True)
loss_axes = figure.add_subplot(1, 2, 1, title='Cross-entropy loss')
accuracy_axes = figure.add_subplot(1, 2, 2, title='Accuracy')
accuracy_axes.set_ylim(0, 1)
plt.ion()
plt.show()
losses = []
accuracies = []
start_time = time.time()
try:
    for epoch in range(epochs):
        model.zero_grad()
        emb = embeddings(messages_var)
        out, state = lstm(emb)  # Forward
        out = linear(out[:, -1, :])
        loss = criterion(out, y)
        accuracy = np.mean((th.max(out, 1)[1] == y).data.numpy())
        print('{}/{} {:.2f}s | loss = {:.4f} | accuracy = {:.2%}'.format(epoch+1, epochs, time.time()-start_time, loss.data.numpy()[0], accuracy))
        losses.append(loss.data.numpy()[0])
        accuracies.append(accuracy)
        loss_axes.plot(losses, color='blue')
        accuracy_axes.plot(accuracies, color='blue')
        plt.pause(0.001)
        loss.backward()
        optimizer.step()
except KeyboardInterrupt:
    pass
finally:
    plt.ioff()
    print('Elapsed training time : {:.2f}s'.format(time.time() - start_time))
