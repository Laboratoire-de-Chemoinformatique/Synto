"""
Module containing a class that represents a policy function for node expansion in the search tree
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import torch
import torch_geometric
import yaml

from Synto.chem.retron import Retron
from Synto.ml.networks.policy import PolicyNetwork
from Synto.ml.training import mol_to_pyg
from Synto.utils import path_type
from Synto.utils.config import ConfigABC


@dataclass
class PolicyConfig(ConfigABC):
    """
    Configuration class for the policy, inheriting from ConfigABC.

    :ivar weights_path: Path to the weights file.
    :ivar top_rules: Number of top rules to be considered. Defaults to 50.
    :ivar threshold: Threshold for rule selection. Defaults to 0.0.
    :ivar priority_rules_fraction: Fraction of priority rules. Defaults to 0.5.
    """

    weights_path: path_type
    top_rules: int = 50
    threshold: float = 0.0
    priority_rules_fraction: float = 0.5

    def __post_init__(self):
        super().__post_init__()
        self.weights_path = Path(self.weights_path).resolve(strict=True)
        params = self.to_dict()
        self._validate_params(params)

    @staticmethod
    def from_dict(config_dict: Dict[str, Any]):
        """
        Creates a PolicyConfig instance from a dictionary of configuration parameters.

        :param config_dict: A dictionary containing configuration parameters.
        :return: An instance of PolicyConfig.
        """
        return PolicyConfig(**config_dict)

    @staticmethod
    def from_yaml(file_path: str):
        """
        Deserializes a YAML file into a PolicyConfig object.

        :param file_path: Path to the YAML file containing configuration parameters.
        :return: An instance of PolicyConfig.
        """
        with open(file_path, "r") as file:
            config_dict = yaml.safe_load(file)
        return PolicyConfig.from_dict(config_dict)

    def _validate_params(self, params: Dict[str, Any]):
        """
        Validates the configuration parameters.

        :param params: A dictionary of parameters to validate.
        :raises ValueError: If any parameter is invalid.
        """
        if not isinstance(params["top_rules"], int) or params["top_rules"] < 0:
            raise ValueError("top_rules must be a non-negative integer.")
        if not isinstance(params["threshold"], float) or not (
            0.0 <= params["threshold"] <= 1.0
        ):
            raise ValueError("threshold must be a float between 0.0 and 1.0.")
        if not isinstance(params["priority_rules_fraction"], float) or not (
            0.0 <= params["priority_rules_fraction"] <= 1.0
        ):
            raise ValueError(
                "priority_rules_fraction must be a float between 0.0 and 1.0."
            )


class PolicyFunction:
    """
    Policy function based on policy neural network for node expansion in MCTS
    """

    def __init__(self, policy_config: PolicyConfig, compile=False):
        """
        Initializes the expansion function (filter policy network).

        :param policy_config: A dictionary containing configuration settings for the expansion policy
        :type policy_config: dict
        """

        self.config = policy_config

        policy_net = PolicyNetwork.load_from_checkpoint(
            self.config.weights_path,
            map_location=torch.device("cpu"),
            batch_size=1,
            dropout=0,
        )
        policy_net = policy_net.eval()
        if compile:
            self.policy_net = torch_geometric.compile(policy_net, dynamic=True)
        else:
            self.policy_net = policy_net

    def predict_reaction_rules(self, retron: Retron, reaction_rules: list):
        """
        The policy function predicts reaction rules based on a given retron and return a list of predicted reaction rules.

        :param retron: The current retron for which the reaction rules are predicted
        :type retron: Retron
        :param reaction_rules: The list of reaction rules from which applicable reaction rules are predicted
        and selected.
        :type reaction_rules: list
        """

        pyg_graph = mol_to_pyg(retron.molecule, canonicalize=False)
        if pyg_graph:
            with torch.no_grad():
                if self.policy_net.mode == "filtering":
                    probs, priority = self.policy_net.forward(pyg_graph)
                else:
                    probs = self.policy_net.forward(pyg_graph)
            del pyg_graph
        else:
            return []

        probs = probs[0].double()
        if self.policy_net.mode == "filtering":
            priority = priority[0].double()
            priority_coef = self.config.priority_rules_fraction
            probs = (1 - priority_coef) * probs + priority_coef * priority

        sorted_probs, sorted_rules = torch.sort(probs, descending=True)
        sorted_probs, sorted_rules = (
            sorted_probs[: self.config.top_rules],
            sorted_rules[: self.config.top_rules],
        )

        if self.policy_net.mode == "filtering":
            sorted_probs = torch.softmax(sorted_probs, -1)

        sorted_probs, sorted_rules = sorted_probs.tolist(), sorted_rules.tolist()

        for prob, rule_id in zip(sorted_probs, sorted_rules):
            if prob > self.config.threshold:
                yield prob, reaction_rules[rule_id], rule_id
