"""
Module containing classes pytorch architectures of policy and value neural networks
"""

from abc import ABC, abstractmethod

import torch
from adabelief_pytorch import AdaBelief
from pytorch_lightning import LightningModule
from torch.nn import Linear, Module, Dropout, ModuleList
from torch.nn.functional import relu
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn.conv import GCNConv
from torch_geometric.nn.pool import global_add_pool


class GraphEmbedding(Module):
    """
    Needed to convert molecule atom vectors to the single vector using graph convolution
    """

    def __init__(self, vector_dim: int = 512, dropout: float = 0.4, num_conv_layers: int = 5):
        """
        It initializes a graph convolutional module. Needed to convert molecule atom vectors to the single vector
        using graph convolution.

        :param vector_dim: The dimensionality of the hidden layers and output layer of graph convolution module.
        :type vector_dim: int
        :param dropout: Dropout is a regularization technique used in neural networks to prevent overfitting.
        It randomly sets a fraction of input units to 0 at each update during training time.
        :type dropout: float
        :param num_conv_layers: The number of convolutional layers in a graph convolutional module.
        :type num_conv_layers: int
        """

        super(GraphEmbedding, self).__init__()
        self.expansion = Linear(11, vector_dim)
        self.dropout = Dropout(dropout)
        self.gcn_convs = ModuleList([GCNConv(vector_dim, vector_dim, improved=True) for _ in range(num_conv_layers)])

    def forward(self, graph):
        """
        The forward function takes a graph as input and performs graph convolution on it.

        :param graph: The molecular graph, where each atom is represented by the atom/bond vector
        """
        atoms, connections = graph.x.float(), graph.edge_index.long()
        atoms = torch.log(atoms + 1)
        atoms = self.expansion(atoms)
        for gcn_conv in self.gcn_convs:
            atoms = atoms + self.dropout(relu(gcn_conv(atoms, connections)))

        return global_add_pool(atoms, graph.batch)


class MCTSNetwork(LightningModule, ABC):
    """
    Basic class for policy and value networks
    """

    def __init__(self, vector_dim, batch_size, dropout=0.4, num_conv_layers=5, learning_rate=0.001):
        """
        The basic class for MCTS graph convolutional neural networks (policy and value network).

        :param vector_dim: The dimensionality of the hidden layers and output layer of graph convolution module.
        :type vector_dim: int
        :param dropout: Dropout is a regularization technique used in neural networks to prevent overfitting.
        It randomly sets a fraction of input units to 0 at each update during training time.
        :type dropout: float
        :param num_conv_layers: The number of convolutional layers in a graph convolutional module.
        :type num_conv_layers: int
        :param learning_rate: The learning rate determines how quickly the model learns from the training data.
        :type learning_rate: float
        """
        super(MCTSNetwork, self).__init__()
        self.embedder = GraphEmbedding(vector_dim, dropout, num_conv_layers)
        self.batch_size = batch_size
        self.lr = learning_rate

    @abstractmethod
    def forward(self, batch):
        """
        The forward function takes a batch of input data and performs forward propagation through the neural network.

        :param batch: The batch parameter is a collection of input data that is processed together in a single forward
        pass through the neural network.
        """
        ...

    @abstractmethod
    def _get_loss(self, batch):
        """
        This function is used to calculate the loss for a given batch of data.

        :param batch: The batch parameter is a batch of input data that is used to compute the loss.
        """
        ...

    def training_step(self, batch, batch_idx):
        """
        Calculates the loss for a given training batch and logs the loss value.

        :param batch: The batch of data that is used for training.
        :param batch_idx: The index of the batch.
        :return: the value of the training loss.
        """
        metrics = self._get_loss(batch)
        for name, value in metrics.items():
            self.log('train_' + name, value, prog_bar=True, on_step=True, on_epoch=True, batch_size=self.batch_size)
        return metrics['loss']

    def validation_step(self, batch, batch_idx):
        """
        Calculates the loss for a given validation batch and logs the loss value.

        :param batch: The batch of data that is used for validation.
        :param batch_idx: The index of the batch.
        """
        metrics = self._get_loss(batch)
        for name, value in metrics.items():
            self.log('val_' + name, value, on_epoch=True, batch_size=self.batch_size)

    def test_step(self, batch, batch_idx):
        """
        Calculates the loss for a given test batch and logs the loss value.

        :param batch: The batch of data that is used for testing.
        :param batch_idx: The index of the batch.
        """
        metrics = self._get_loss(batch)
        for name, value in metrics.items():
            self.log('test_' + name, value, on_epoch=True, batch_size=self.batch_size)

    def configure_optimizers(self):
        """
        Returns an optimizer and a learning rate scheduler for training a model using the AdaBelief optimizer
        and ReduceLROnPlateau scheduler.
        :return:  The optimizer and a scheduler.
        """
        optimizer = AdaBelief(self.parameters(), lr=self.lr, eps=1e-16, betas=(0.9, 0.999), weight_decouple=True,
                              rectify=True, weight_decay=0.01, print_change_log=False)

        lr_scheduler = ReduceLROnPlateau(optimizer, patience=3, factor=0.8, min_lr=5e-5, verbose=True)
        scheduler = {'scheduler': lr_scheduler, 'reduce_on_plateau': True, 'monitor': 'val_loss'}
        return [optimizer], [scheduler]
