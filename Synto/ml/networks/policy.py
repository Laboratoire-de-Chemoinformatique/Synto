from abc import ABC

import torch
from pytorch_lightning import LightningModule
from torch.nn import Linear
from torch.nn.functional import binary_cross_entropy_with_logits
from torchmetrics.functional.classification import multilabel_recall, multilabel_specificity, multilabel_f1_score

from Synto.ml.networks.modules import MCTSNetwork


class PolicyNetwork(MCTSNetwork, LightningModule, ABC):
    """
    Policy value network
    """
    def __init__(self, n_rules, vector_dim, mode="filtering", *args, **kwargs):
        """
        Initializes a policy network with the given number of reaction rules (output dimension) and vector graph
        embedding dimension, and creates linear layers for predicting the regular and priority reaction rules.

        :param n_rules: The number of reaction rules in the policy network.
        :param vector_dim: The dimensionality of the input vectors.
        """
        super(PolicyNetwork, self).__init__(vector_dim, *args, **kwargs)
        self.save_hyperparameters()
        self.mode = mode
        self.n_rules = n_rules
        self.y_predictor = Linear(vector_dim, n_rules)
        if self.mode == "filtering":
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
        y = self.y_predictor(x)
        if self.mode == "filtering":
            y = torch.sigmoid(y)
            priority = torch.sigmoid(self.priority_predictor(x))
            return y, priority
        elif self.mode == "ranking":
            y = torch.softmax(y, dim=-1)
            return y

    def _get_loss(self, batch):
        """
        Calculates the loss and various classification metrics for a given batch for reaction rules prediction.

        :param batch: The batch of molecular graphs.
        :return: a dictionary with loss value and balanced accuracy of reaction rules prediction.
        """
        true_y = batch.y_rules.float()
        x = self.embedder(batch)
        pred_y = self.y_predictor(x)

        loss_y = binary_cross_entropy_with_logits(pred_y, true_y)
        loss = loss_y
        true_y = true_y.long()
        ba_y = (multilabel_recall(pred_y, true_y, num_labels=self.n_rules) +
                multilabel_specificity(pred_y, true_y, num_labels=self.n_rules)) / 2
        f1_y = multilabel_f1_score(pred_y, true_y, num_labels=self.n_rules)

        metrics = {
            'balanced_accuracy_y': ba_y,
            'f1_score_y': f1_y
        }
        
        if self.mode == "filtering":
            true_priority = batch.y_priority.float()
            pred_priority = self.priority_predictor(x)

            loss_priority = binary_cross_entropy_with_logits(pred_priority, true_priority)
            loss = loss + loss_priority

            true_priority = true_priority.long()

            ba_priority = (multilabel_recall(pred_priority, true_priority, num_labels=self.n_rules) +
                           multilabel_specificity(pred_priority, true_priority, num_labels=self.n_rules)) / 2
            f1_priority = multilabel_f1_score(pred_priority, true_priority, num_labels=self.n_rules)

            metrics['balanced_accuracy_priority'] = ba_priority
            metrics['f1_score_priority'] = f1_priority

        metrics["loss"] = loss

        return metrics
