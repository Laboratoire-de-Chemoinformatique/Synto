"""
Module containing a class that represents a policy function for node expansion in the search tree
"""

import torch
import torch_geometric
from Synto.chem.retron import Retron
from Synto.ml.networks.policy import PolicyNetwork
from Synto.ml.training import mol_to_pyg
from Synto.utils.config import PolicyConfig



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

        # policy_net = PolicyNetwork.load_from_checkpoint(
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
