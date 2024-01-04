"""
Module containing a class that represents a policy function for node expansion in the search tree
"""

import torch
import torch_geometric
from Syntool.chem.retron import Retron
from Syntool.ml.networks.policy import PolicyNetwork
from Syntool.ml.training import mol_to_pyg
from Syntool.utils.config import PolicyConfig



class PolicyFunction:
    """
    Policy function based on policy neural network for node expansion in MCTS
    """

    def __init__(self, policy_config: PolicyConfig, compile: bool = False):
        """
        Initializes the expansion function (ranking or filter policy network).

        :param policy_config: A configuration object settings for the expansion policy
        :type policy_config: PolicyConfig
        :param compile: XX # TODO what is compile
        :type compile: bool
        """

        self.config = policy_config

        # policy_net = PolicyNetwork.load_from_checkpoint(   # TODO remove these block ?
        #     self.config.weights_path,
        #     map_location=torch.device("cpu"),
        #     batch_size=1,
        #     dropout=0,
        # )

        policy_net = PolicyNetwork.load_from_checkpoint(
            self.config.weights_path,
            map_location=torch.device("cpu"),
            batch_size=1,
            dropout=0
        )

        policy_net = policy_net.eval()
        if compile:
            self.policy_net = torch_geometric.compile(policy_net, dynamic=True)
        else:
            self.policy_net = policy_net

    def predict_reaction_rules(self, retron: Retron, reaction_rules: list):  # TODO what is output - finish annotation
        """
        The policy function predicts the list of reaction rules given a retron.

        :param retron: The current retron for which the reaction rules are predicted
        :type retron: Retron
        :param reaction_rules: The list of reaction rules from which applicable reaction rules are predicted and selected.
        :type reaction_rules: list
        """

        pyg_graph = mol_to_pyg(retron.molecule, canonicalize=False)
        if pyg_graph:
            with torch.no_grad():
                if self.policy_net.policy_type == "filtering":
                    probs, priority = self.policy_net.forward(pyg_graph)
                if self.policy_net.policy_type == "ranking":
                    probs = self.policy_net.forward(pyg_graph)
            del pyg_graph
        else:
            return []

        probs = probs[0].double()
        if self.policy_net.policy_type == "filtering":
            priority = priority[0].double()
            priority_coef = self.config.priority_rules_fraction
            probs = (1 - priority_coef) * probs + priority_coef * priority

        sorted_probs, sorted_rules = torch.sort(probs, descending=True)
        sorted_probs, sorted_rules = (
            sorted_probs[: self.config.top_rules],
            sorted_rules[: self.config.top_rules],
        )

        if self.policy_net.policy_type == "filtering":
            sorted_probs = torch.softmax(sorted_probs, -1)

        sorted_probs, sorted_rules = sorted_probs.tolist(), sorted_rules.tolist()

        for prob, rule_id in zip(sorted_probs, sorted_rules):
            if prob > self.config.threshold:
                yield prob, reaction_rules[rule_id], rule_id
