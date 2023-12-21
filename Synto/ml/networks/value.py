from abc import ABC
from dataclasses import dataclass
from typing import Dict, Any

import yaml
import torch
from pytorch_lightning import LightningModule
from torch.nn import Linear
from torch.nn.functional import binary_cross_entropy_with_logits
from torchmetrics.functional.classification import binary_recall, binary_specificity, binary_f1_score

from Synto.ml.networks.modules import MCTSNetwork
from Synto.utils.config import ConfigABC


@dataclass
class ValueNetworkConfig(ConfigABC):
    """
    Configuration class for the value network, inheriting from ConfigABC.

    :ivar vector_dim: Dimension of the input vectors.
    :ivar batch_size: Number of samples per batch.
    :ivar dropout: Dropout rate for regularization.
    :ivar learning_rate: Learning rate for the optimizer.
    :ivar num_conv_layers: Number of convolutional layers in the network.
    :ivar num_epoch: Number of training epochs.
    """

    vector_dim: int = 256
    batch_size: int = 500
    dropout: float = 0.4
    learning_rate: float = 0.008
    num_conv_layers: int = 5
    num_epoch: int = 100

    @staticmethod
    def from_dict(config_dict: Dict[str, Any]) -> 'ValueNetworkConfig':
        """
        Creates a ValueNetworkConfig instance from a dictionary of configuration parameters.

        :ivar config_dict: A dictionary containing configuration parameters.
        :return: An instance of ValueNetworkConfig.
        """
        return ValueNetworkConfig(**config_dict)

    @staticmethod
    def from_yaml(file_path: str) -> 'ValueNetworkConfig':
        """
        Deserializes a YAML file into a ValueNetworkConfig object.

        :ivar file_path: Path to the YAML file containing configuration parameters.
        :return: An instance of ValueNetworkConfig.
        """
        with open(file_path, 'r') as file:
            config_dict = yaml.safe_load(file)
        return ValueNetworkConfig.from_dict(config_dict)

    def to_yaml(self, file_path: str):
        """
        Serializes the configuration to a YAML file.

        :ivar file_path: Path to the YAML file for serialization.
        """
        with open(file_path, 'w') as file:
            yaml.dump(self.to_dict(), file)

    def _validate_params(self, params: Dict[str, Any]):
        """
        Validates the configuration parameters.

        :ivar params: A dictionary of parameters to validate.
        :raises ValueError: If any parameter is invalid.
        """
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
