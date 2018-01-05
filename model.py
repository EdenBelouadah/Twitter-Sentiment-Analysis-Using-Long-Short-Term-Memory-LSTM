from torch import nn
from torch import autograd as ag
from torch import optim


class EmbeddingNeuralNetwork(nn.Module):

    def __init__(self, vocabulary_size, embedding_dimension, hidden_layers_dimensions, output_dim, activation=nn.ReLU):
        super(EmbeddingNeuralNetwork, self).__init__()


    def forward(self, input_):
        pass

    def train_step(self, x, y):
        pass
