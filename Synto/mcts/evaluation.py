"""
Module containing a class that represents a value function for prediction of synthesisablity
of new nodes in the search tree
"""

import logging

import torch

from Synto.ml.networks.value import SynthesabilityValueNetwork
from Synto.ml.training import mol_to_pyg
from Synto.chem.retron import compose_retrons


class ValueFunction:
    """
    Value function based on value neural network for node evaluation (synthesisability prediction) in MCTS
    """

    def __init__(self, weights_path):
        """
        The value function predicts the probability to synthesize the target molecule with available building blocks
        starting from a given retron.

        """

        value_net = SynthesabilityValueNetwork.load_from_checkpoint(
            weights_path,
            map_location=torch.device("cpu")
        )

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
