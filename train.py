from __future__ import division
import sys
import numpy as np
import torch as th
import torch.autograd as ag
from torch import nn, optim


from data import load_B_task_dataset, process_messages, generate_vocabulary, transform_to_indices, create_vocabulary_embeddings

messages, polarities = load_B_task_dataset(sys.argv[1])
p_messages = process_messages(messages)
vocab = generate_vocabulary(p_messages)
vocab.add("<unk>")

vocabulary_embeddings,  embeddings_indices = create_vocabulary_embeddings(vocab, p_messages)

message_length = max(len(msg) for msg in p_messages)
messages_indices = transform_to_indices(p_messages, embeddings_indices)
messages_np = np.empty((len(messages), message_length), dtype=np.int)
for i in range(messages_np.shape[0]):
    messages_np[i] = ((message_length//len(messages_indices[i])+1)*messages_indices[i])[:message_length]

embeddings = nn.Embedding(*vocabulary_embeddings.shape)
embeddings.weight = nn.Parameter(th.FloatTensor(vocabulary_embeddings))

messages_var = ag.Variable(th.LongTensor(messages_np))
# print(messages_var.max(), messages_var.min())

lstm = nn.LSTM(input_size=vocabulary_embeddings.shape[1], hidden_size=15, num_layers=1, batch_first=True)
linear = nn.Linear(15, 3)
model = nn.Sequential(lstm, linear)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=1)
criterion = nn.CrossEntropyLoss()

y = ag.Variable(th.LongTensor(polarities))

epochs = 50

for epoch in range(epochs):
    model.zero_grad()
    emb = embeddings(messages_var)
    out, state = lstm(emb)  # Forward
    out = linear(out[:, -1, :])
    loss = criterion(out, y)
    accuracy = np.mean((th.max(out, 1)[1] == y).data.numpy())
    print('{}/{} loss = {:.4f} | accuracy = {:.2f}'.format(epoch+1, epochs, loss.data.numpy()[0], accuracy))
    loss.backward()
    optimizer.step()





