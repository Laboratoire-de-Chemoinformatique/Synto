from abc import ABC
from dataclasses import dataclass
from typing import Dict, Any

import yaml
import torch
from pytorch_lightning import LightningModule
from torch.nn import Linear
from torch.nn.functional import binary_cross_entropy_with_logits, cross_entropy, one_hot
from torchmetrics.functional.classification import recall, specificity, f1_score

from Syntool.ml.networks.modules import MCTSNetwork
from Syntool.utils.config import ConfigABC


@dataclass
class PolicyNetworkConfig(ConfigABC):
    """
    Configuration class for the policy network, inheriting from ConfigABC.

    :ivar vector_dim: Dimension of the input vectors.
    :ivar batch_size: Number of samples per batch.
    :ivar dropout: Dropout rate for regularization.
    :ivar learning_rate: Learning rate for the optimizer.
    :ivar num_conv_layers: Number of convolutional layers in the network.
    :ivar num_epoch: Number of training epochs.
    :ivar policy_type: Mode of operation, either 'filtering' or 'ranking'.
    """
    policy_type: str = "ranking"
    vector_dim: int = 256
    batch_size: int = 500
    dropout: float = 0.4
    learning_rate: float = 0.008
    num_conv_layers: int = 5
    num_epoch: int = 100
    weights_path: str = None
    threshold: float = 0.0
    
    # for filtering policy
    priority_rules_fraction: float = 0.5
    rule_prob_threshold: float = 0.0
    top_rules: int = 50

    @staticmethod
    def from_dict(config_dict: Dict[str, Any]) -> 'PolicyNetworkConfig':
        """
        Creates a PolicyNetworkConfig instance from a dictionary of configuration parameters.

        :param config_dict: A dictionary containing configuration parameters.
        :return: An instance of PolicyNetworkConfig.
        """
        return PolicyNetworkConfig(**config_dict)

    @staticmethod
    def from_yaml(file_path: str) -> 'PolicyNetworkConfig':
        """
        Deserializes a YAML file into a PolicyNetworkConfig object.

        :param file_path: Path to the YAML file containing configuration parameters.
        :return: An instance of PolicyNetworkConfig.
        """
        with open(file_path, 'r') as file:
            config_dict = yaml.safe_load(file)
        return PolicyNetworkConfig.from_dict(config_dict)

    def _validate_params(self, params: Dict[str, Any]):

        if params['policy_type'] not in ["filtering", "ranking"]:
            raise ValueError("policy_type must be either 'filtering' or 'ranking'.")

        if not isinstance(params['vector_dim'], int) or params['vector_dim'] <= 0:
            raise ValueError("vector_dim must be a positive integer.")

        if not isinstance(params['batch_size'], int) or params['batch_size'] <= 0:
            raise ValueError("batch_size must be a positive integer.")

        if not isinstance(params['num_conv_layers'], int) or params['num_conv_layers'] <= 0:
            raise ValueError("num_conv_layers must be a positive integer.")

        if not isinstance(params['num_epoch'], int) or params['num_epoch'] <= 0:
            raise ValueError("num_epoch must be a positive integer.")

        if not isinstance(params['dropout'], float) or not (0.0 <= params['dropout'] <= 1.0):
            raise ValueError("dropout must be a float between 0.0 and 1.0.")

        if not isinstance(params['learning_rate'], float) or params['learning_rate'] <= 0.0:
            raise ValueError("learning_rate must be a positive float.")

        if not isinstance(params['priority_rules_fraction'], float) or params['priority_rules_fraction'] < 0.0:
            raise ValueError("priority_rules_fraction must be a non-negative positive float.")

        if not isinstance(params['rule_prob_threshold'], float) or params['rule_prob_threshold'] < 0.0:
            raise ValueError("rule_prob_threshold must be a non-negative float.")

        if not isinstance(params['top_rules'], int) or params['top_rules'] <= 0:
            raise ValueError("top_rules must be a positive integer.")


class PolicyNetwork(MCTSNetwork, LightningModule, ABC):
    """
    Policy value network
    """

    def __init__(self, n_rules, vector_dim, policy_type="filtering", *args, **kwargs):
        """
        Initializes a policy network with the given number of reaction rules (output dimension) and vector graph
        embedding dimension, and creates linear layers for predicting the regular and priority reaction rules.

        :param n_rules: The number of reaction rules in the policy network.
        :param vector_dim: The dimensionality of the input vectors.
        """
        super(PolicyNetwork, self).__init__(vector_dim, *args, **kwargs)
        self.save_hyperparameters()
        self.policy_type = policy_type
        self.n_rules = n_rules
        self.y_predictor = Linear(vector_dim, n_rules)
        if self.policy_type == "filtering":
            self.priority_predictor = Linear(vector_dim, n_rules)

    def forward(self, batch):
        """
        The forward function takes a molecular graph, applies a graph convolution and sigmoid layers to predict
        regular and priority reaction rules.

        :param batch: The input batch of molecular graphs.
        :return: Returns the vector of probabilities (given by sigmoid) of successful application of regular and
        priority reaction rules.
        """
        x = self.embedder(batch, self.batch_size)
        y = self.y_predictor(x)
        if self.policy_type == "filtering":
            y = torch.sigmoid(y)
            priority = torch.sigmoid(self.priority_predictor(x))
            return y, priority
        elif self.policy_type == "ranking":
            y = torch.softmax(y, dim=-1)
            return y

    def _get_loss(self, batch):
        """
        Calculates the loss and various classification metrics for a given batch for reaction rules prediction.

        :param batch: The batch of molecular graphs.
        :return: a dictionary with loss value and balanced accuracy of reaction rules prediction.
        """
        true_y = batch.y_rules.long()
        x = self.embedder(batch, self.batch_size)
        pred_y = self.y_predictor(x)

        if self.policy_type == "ranking":
            true_one_hot = one_hot(true_y, num_classes=self.n_rules)
            loss = cross_entropy(pred_y, true_one_hot.float())
            ba_y = (
                           recall(pred_y, true_y, task="multiclass", num_classes=self.n_rules) +
                           specificity(pred_y, true_y, task="multiclass", num_classes=self.n_rules)
                   ) / 2
            f1_y = f1_score(pred_y, true_y, task="multiclass", num_classes=self.n_rules)
            metrics = {
                'loss': loss,
                'balanced_accuracy_y': ba_y,
                'f1_score_y': f1_y
            }
        elif self.policy_type == "filtering":
            loss_y = binary_cross_entropy_with_logits(pred_y, true_y.float())
            ba_y = (
                           recall(pred_y, true_y, task="multilabel", num_labels=self.n_rules) +
                           specificity(pred_y, true_y, task="multilabel", num_labels=self.n_rules)
                   ) / 2
            f1_y = f1_score(pred_y, true_y, task="multilabel", num_labels=self.n_rules)

            true_priority = batch.y_priority.float()
            pred_priority = self.priority_predictor(x)

            loss_priority = binary_cross_entropy_with_logits(pred_priority, true_priority)
            loss = loss_y + loss_priority

            true_priority = true_priority.long()

            ba_priority = (
                                  recall(pred_priority, true_priority, task="multilabel", num_labels=self.n_rules) +
                                  specificity(pred_priority, true_priority, task="multilabel", num_labels=self.n_rules)
                          ) / 2
            f1_priority = f1_score(pred_priority, true_priority, task="multilabel", num_labels=self.n_rules)
            metrics = {
                'loss': loss,
                'balanced_accuracy_y': ba_y,
                'f1_score_y': f1_y,
                'balanced_accuracy_priority': ba_priority,
                'f1_score_priority': f1_priority
            }
        else:
            raise ValueError(f"Invalid mode: {self.policy_type}")

        return metrics
