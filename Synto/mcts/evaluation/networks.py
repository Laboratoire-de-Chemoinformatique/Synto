"""
Module containing a class that represents a value function for prediction of synthesisablity
of new nodes in the search tree
"""

import logging

import torch

from Synto.networks.networks import ValueGraphNetwork
from Synto.training.loading import load_value_net
from Synto.training.preprocessing import compose_retrons
from Synto.training.preprocessing import mol_to_pyg


class ValueFunction:
    """
    Value function based on value neural network for node evaluation (synthesisability prediction) in MCTS
    """

    def __init__(self, config):
        """
        The value function predicts the probability to synthesize the target molecule with available building blocks
        starting from a given retron.

        :param config: The `config` parameter is a dictionary that contains configuration settings for the
        policy function
        :type config: dict
        """

        value_net = load_value_net(ValueGraphNetwork, config)
        self.value_network = value_net.eval()

    def predict_value(self, retrons: list):
        """
        The function predicts a value based on the given retrons. For prediction, retrons must be composed into a single
        molecule (product)

        :param retrons: The list of retrons
        :type retrons: list
        """

        molecule = compose_retrons(retrons=retrons, exclude_small=True)
        pyg_graph = mol_to_pyg(molecule)
        if pyg_graph:
            with torch.no_grad():
                value_pred = self.value_network.forward(pyg_graph)[0].item()
        else:
            try:
                logging.debug(f"Molecule {str(molecule)} was not preprocessed. Giving value equal to -1e6.")
            except:
                logging.debug(f"There is a molecule for which SMILES cannot be generated")

            value_pred = -1e6

        return value_pred
