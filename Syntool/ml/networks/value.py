from abc import ABC
from dataclasses import dataclass
from typing import Dict, Any

import yaml
import torch
from pytorch_lightning import LightningModule
from torch.nn import Linear
from torch.nn.functional import binary_cross_entropy_with_logits
from torchmetrics.functional.classification import binary_recall, binary_specificity, binary_f1_score

from Syntool.ml.networks.modules import MCTSNetwork
from Syntool.utils.config import ConfigABC


class SynthesabilityValueNetwork(MCTSNetwork, LightningModule, ABC):
    """
    Value value network
    """

    def __init__(self, vector_dim, *args, **kwargs):
        """
        Initializes a value network, and creates linear layer for predicting the synthesisability of given retron
        represented by molecular graph.

        :param vector_dim: The dimensionality of the output linear layer.
        """
        super(SynthesabilityValueNetwork, self).__init__(vector_dim, *args, **kwargs)
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
        x = self.embedder(batch, self.batch_size)
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
        x = self.embedder(batch, self.batch_size)
        pred_y = self.predictor(x)
        loss = binary_cross_entropy_with_logits(pred_y, true_y)
        true_y = true_y.long()
        ba = (binary_recall(pred_y, true_y) + binary_specificity(pred_y, true_y)) / 2
        f1 = binary_f1_score(pred_y, true_y)
        metrics = {'loss': loss, 'balanced_accuracy': ba, 'f1_score': f1}
        return metrics
