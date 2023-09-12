"""
Module containing classes pytorch architectures of policy and value neural networks
"""

import torch

from abc import ABC, abstractmethod
from pytorch_lightning import LightningModule
from adabelief_pytorch import AdaBelief
from torch.nn import Linear, Module, Dropout, ModuleList
from torch.nn.functional import relu, binary_cross_entropy_with_logits
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn.pool import global_add_pool
from torch_geometric.nn.conv import GCNConv
from torchmetrics.functional.classification import multilabel_f1_score, multilabel_recall, multilabel_specificity
from torchmetrics.functional.classification import binary_f1_score, binary_recall, binary_specificity


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
        optimizer = AdaBelief(self.parameters(), lr=self.lr, eps=1e-16, betas=(0.9, 0.999),
                              weight_decouple=True, rectify=True, weight_decay=0.01,
                              print_change_log=False)

        lr_scheduler = ReduceLROnPlateau(optimizer, patience=3, factor=0.8, min_lr=5e-5, verbose=True)
        scheduler = {
            'scheduler': lr_scheduler,
            'reduce_on_plateau': True,
            'monitor': 'val_loss'
        }
        return [optimizer], [scheduler]


# class PolicyNetwork(MCTSNetwork, LightningModule, ABC):
#     def __init__(self, n_rules, vector_dim, *args, **kwargs):
#         super(PolicyNetwork, self).__init__(vector_dim, *args, **kwargs)
#         self.save_hyperparameters()
#         self.n_rules = n_rules
#         self.rules_predictor = Linear(vector_dim, n_rules)
#         self.priority_predictor = Linear(vector_dim, n_rules)
#
#     def forward(self, batch):
#         x = self.embedder(batch)
#         y = torch.sigmoid(self.rules_predictor(x))
#         priority = torch.sigmoid(self.priority_predictor(x))
#         return y, priority
#
#     def _get_loss(self, batch):
#         true_rules = batch.y_rules.float()
#         true_priority = batch.y_priority.float()
#
#         x = self.embedder(batch)
#
#         pred_rules = self.rules_predictor(x)
#         pred_priority = self.priority_predictor(x)
#
#         loss_rules = binary_cross_entropy_with_logits(pred_rules, true_rules)
#         loss_priority = binary_cross_entropy_with_logits(pred_priority, true_priority)
#         loss = loss_rules + loss_priority
#
#         true_rules = true_rules.long()
#         true_priority = true_priority.long()
#
#         ba_rules = (multilabel_recall(pred_rules, true_rules, num_labels=self.n_rules) +
#                     multilabel_specificity(pred_rules, true_rules, num_labels=self.n_rules)) / 2
#         f1_rules = multilabel_f1_score(pred_rules, true_rules, num_labels=self.n_rules)
#
#         ba_priority = (multilabel_recall(pred_priority, true_priority, num_labels=self.n_rules) +
#                        multilabel_specificity(pred_priority, true_priority, num_labels=self.n_rules)) / 2
#         f1_priority = multilabel_f1_score(pred_priority, true_priority, num_labels=self.n_rules)
#
#         metrics = {
#             'loss': loss,
#             'balanced_accuracy_rules': ba_rules, 'f1_score_rules': f1_rules,
#             'balanced_accuracy_priority': ba_priority, 'f1_score_priority': f1_priority,
#         }
#         return metrics


class PolicyNetwork(MCTSNetwork, LightningModule, ABC):
    """
    Policy value network
    """
    def __init__(self, n_rules, vector_dim, *args, **kwargs):
        """
        Initializes a policy network with the given number of reaction rules (output dimension) and vector graph
        embedding dimension, and creates linear layers for predicting the regular and priority reaction rules.

        :param n_rules: The number of reaction rules in the policy network.
        :param vector_dim: The dimensionality of the input vectors.
        """
        super(PolicyNetwork, self).__init__(vector_dim, *args, **kwargs)
        self.save_hyperparameters()
        self.n_rules = n_rules
        self.y_predictor = Linear(vector_dim, n_rules)
        self.priority_predictor = Linear(vector_dim, n_rules)

    def forward(self, batch):
        """
        The forward function takes a molecular graph, applies a graph convolution and sigmoid layers to predict
        regular and priority reaction rules.

        :param batch: The input batch of molecular graphs.
        :return: Returns the vector of probabilities (given by sigmoid) of successful application of regular and
        priority reaction rules.
        """
        x = self.embedder(batch)
        y = torch.sigmoid(self.y_predictor(x))
        priority = torch.sigmoid(self.priority_predictor(x))
        return y, priority

    def _get_loss(self, batch):
        """
        Calculates the loss and various classification metrics for a given batch for reaction rules prediction.

        :param batch: The batch of molecular graphs.
        :return: a dictionary with loss value and balanced accuracy of reaction rules prediction.
        """
        true_y = batch.y.float()
        true_priority = batch.priority.float()

        x = self.embedder(batch)
        pred_y = self.y_predictor(x)
        pred_priority = self.priority_predictor(x)
        loss_y = binary_cross_entropy_with_logits(pred_y, true_y)
        loss_priority = binary_cross_entropy_with_logits(pred_priority, true_priority)
        loss = loss_y + loss_priority

        true_y = true_y.long()
        true_priority = true_priority.long()

        ba_y = (multilabel_recall(pred_y, true_y, num_labels=self.n_rules) +
                multilabel_specificity(pred_y, true_y, num_labels=self.n_rules)) / 2
        f1_y = multilabel_f1_score(pred_y, true_y, num_labels=self.n_rules)

        ba_priority = (multilabel_recall(pred_priority, true_priority, num_labels=self.n_rules) +
                       multilabel_specificity(pred_priority, true_priority, num_labels=self.n_rules)) / 2
        f1_priority = multilabel_f1_score(pred_priority, true_priority, num_labels=self.n_rules)

        metrics = {
            'loss': loss,
            'balanced_accuracy_y': ba_y, 'f1_score_y': f1_y,
            'balanced_accuracy_priority': ba_priority, 'f1_score_priority': f1_priority,
        }
        return metrics


class ValueGraphNetwork(MCTSNetwork, LightningModule, ABC):
    """
    Value value network
    """
    def __init__(self, vector_dim, *args, **kwargs):
        """
        Initializes a value network, and creates linear layer for predicting the synthesisability of given retron
        represented by molecular graph.

        :param vector_dim: The dimensionality of the output linear layer.
        """
        super(ValueGraphNetwork, self).__init__(vector_dim, *args, **kwargs)
        self.save_hyperparameters()
        self.predictor = Linear(vector_dim, 1)

    def forward(self, batch) -> torch.Tensor:
        """
        The forward function takes a batch of molecular graphs, applies a graph convolution returns the synthesisability
        (probability given by sigmoid function) of a given retron represented by molecular graph precessed by
        graph convolution.

        :param batch: The batch of molecular graphs.
        :return: a predicted synthesisability (between 0 and 1).
        """
        x = self.embedder(batch)
        x = torch.sigmoid(self.predictor(x))
        return x

    def _get_loss(self, batch):
        """
        Calculates the loss and various classification metrics for a given batch for retron synthesysability prediction.

        :param batch: The batch of molecular graphs.
        :return: a dictionary with loss value and balanced accuracy of retron synthesysability prediction.
        """
        true_y = batch.y.float()
        true_y = torch.unsqueeze(true_y, -1)
        x = self.embedder(batch)
        pred_y = self.predictor(x)
        loss = binary_cross_entropy_with_logits(pred_y, true_y)
        true_y = true_y.long()
        ba = (binary_recall(pred_y, true_y) +
              binary_specificity(pred_y, true_y)) / 2
        f1 = binary_f1_score(pred_y, true_y)
        metrics = {'loss': loss, 'balanced_accuracy': ba, 'f1_score': f1}
        return metrics
