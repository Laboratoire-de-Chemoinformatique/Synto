"""
Module containing a class Tree that used for tree search of retrosynthetic paths
"""

import logging
from collections import deque, defaultdict
from math import sqrt
from random import choice, uniform
from time import time
from typing import Dict, Set, List, Tuple

from numpy.random import uniform

from Synto.chem.reaction import Reaction
from Synto.chem.reaction import apply_reaction_rule
from Synto.chem.retron import Retron
from Synto.interfaces.visualisation import tqdm
from Synto.mcts.evaluation.networks import ValueFunction
from Synto.mcts.expansion.filter_policy import PolicyFunction
from Synto.mcts.node import Node
from Synto.training.preprocessing import safe_canonicalization
from Synto.utils.loading import load_building_blocks, load_reaction_rules


class Tree:
    """
    Tree class with attributes and methods for the Monte-Carlo tree search
    """

    def __init__(self, target: object = None, config: dict = None):
        """
        The function initializes a tree object with optional parameters for tree search for target molecule.

        :param target: a target molecule for retrosynthesis paths search
        :type target: object
        :param config: The `config` parameter is a dictionary that contains configuration settings for the tree search
        :type config: dict
        """

        if not target:
            assert 'Target is not defined'
        target_retron = Retron(target)
        target_retron.prev_retrons.append(Retron(target))
        target_node = Node(retrons_to_expand=(target_retron,), new_retrons=(target_retron,))

        # config parameters
        self.max_iterations = int(config['Tree']['max_iterations'])
        self.max_tree_size = int(config['Tree']['max_tree_size'])
        self.max_time = config['Tree']['max_time']
        self.max_depth = config['Tree']['max_depth']
        self.ucb_type = config['Tree']['ucb_type']
        self.backprop_type = config['Tree']['backprop_type']
        self.c_ucb = config['Tree']['c_usb']
        # self.epsilon = config['Tree']['epsilon']
        # self.exclude_small = config['ValueNetwork']['exclude_small']
        self.epsilon = 0.0
        self.exclude_small = True
        self.evaluation_agg = config['Tree']['evaluation_agg']
        self.evaluation_mode = config['Tree']['evaluation_mode']
        self.init_new_node_value = None
        self.silent = not config['Tree']['verbose']

        # tree structure init
        self.nodes: Dict[int, Node] = {1: target_node}
        self.parents: Dict[int, int] = {1: 0}
        self.children: Dict[int, Set[int]] = {1: set()}
        self.winning_nodes: Dict[int, int] = {}
        self.visited_nodes: Set[int] = set()
        self.expanded_nodes: Set[int] = set()
        self.nodes_visit: Dict[int, int] = {1: 0}
        self.nodes_depth: Dict[int, int] = {1: 0}
        self.nodes_prob: Dict[int, float] = {1: 0.0}
        self.nodes_init_value: Dict[int, float] = {1: 0.0}
        self.nodes_total_value: Dict[int, float] = {1: 0.0}

        # tree building limits
        self.curr_iteration: int = 0
        self.curr_tree_size: int = 2
        self.curr_time: float = 2

        # utils
        self._tqdm = None

        # networks loading
        self.policy_function = PolicyFunction(config)
        self.value_function = ValueFunction(config)

        # building blocks and reaction reaction_rules
        self.reaction_rules = load_reaction_rules(config['ReactionRules']['reaction_rules_path'])
        self.building_blocks = load_building_blocks(config['General']['building_blocks_path'])

    def __len__(self) -> int:
        """
        Returns the current size (number of nodes) of a Tree.
        """

        return self.curr_tree_size - 1

    def __iter__(self) -> "Tree":
        """
        The function is defining an iterator for a Tree object. Also needed for the bar progress display.
        """

        if not self._tqdm:
            self._start_time = time()
            self._tqdm = tqdm(total=self.max_iterations, disable=self.silent)
        return self

    def __repr__(self) -> str:
        """
        Returns a string representation of a Tree object (target smiles, tree size, and the number of found paths).
        """
        return self.report()

    def __next__(self):
        """
        The __next__ function is used to do one iteration of the tree building.
        """
        # check if target is building_block
        if self.nodes[1].curr_retron.is_building_block(self.building_blocks):
            raise StopIteration("Target is building block \n")

        if self.curr_iteration >= self.max_iterations:
            self._tqdm.close()  # TODO correct later
            raise StopIteration("Iterations limit exceeded. \n")
        elif self.curr_tree_size >= self.max_tree_size:
            self._tqdm.close()
            raise StopIteration("Max tree size exceeded or all possible paths found")
        elif self.curr_time >= self.max_time:
            self._tqdm.close()
            raise StopIteration("Time limit exceeded. \n")
        else:
            pass

        # start new iteration

        self.curr_iteration += 1
        self.curr_time = time() - self._start_time
        self._tqdm.update()

        curr_depth, node_id = 0, 1  # start from the root node_id

        explore_path = True
        while explore_path:

            self.visited_nodes.add(node_id)

            if self.nodes_visit[node_id]:  # already visited
                if not self.children[node_id]:  # dead node
                    self._update_visits(node_id)
                    explore_path = False
                else:
                    node_id = self._select_node(node_id)  # select the child node
                    curr_depth += 1
            else:
                if self.nodes[node_id].is_solved():  # found path!
                    self._update_visits(node_id)  # this prevents expanding of bb node_id
                    self.winning_nodes[node_id] = self.curr_iteration
                    return True, [node_id]

                elif curr_depth < self.max_depth:  # expand node if depth limit is not reached

                    self._expand_node(node_id)
                    if not self.children[node_id]:  # node was not expanded
                        return False, [node_id]
                    self.expanded_nodes.add(node_id)

                    # recalculate node value based on children synthesisability and backpropagation
                    child_values = [self.nodes_init_value[child_id] for child_id in self.children[node_id]]

                    if self.evaluation_agg == "max":
                        value_to_backprop = max(child_values)

                    elif self.evaluation_agg == "avg":
                        value_to_backprop = sum(child_values) / len(self.children[node_id])

                    else:
                        raise ValueError(f"I don't know this evaluation aggregation mode: {self.evaluation_agg}")

                    # backpropagation
                    self._backpropagate(node_id, value_to_backprop)
                    self._update_visits(node_id)
                    explore_path = False

                    # found after expansion
                    found_after_expansion = set()
                    for child_id in iter(self.children[node_id]):
                        if self.nodes[child_id].is_solved():
                            found_after_expansion.add(child_id)
                            self.winning_nodes[child_id] = self.curr_iteration

                    if found_after_expansion:
                        return True, list(found_after_expansion)

                else:
                    self._backpropagate(node_id, self.nodes_total_value[node_id])
                    self._update_visits(node_id)
                    explore_path = False

        return False, [node_id]

    def _ucb(self, node_id: int) -> float:
        """
        The function calculates the Upper Confidence Bound (UCB) for a given node.

        :param node_id: The `node_id` parameter is an integer that represents the ID of a node in a tree
        :type node_id: int
        """

        prob = self.nodes_prob[node_id]  # Predicted by policy network score
        visit = self.nodes_visit[node_id]

        if self.ucb_type == "puct":
            u = (self.c_ucb * prob * sqrt(self.nodes_visit[self.parents[node_id]])) / (visit + 1)
            return self.nodes_total_value[node_id] + u
        elif self.ucb_type == "uct":
            u = self.c_ucb * sqrt(self.nodes_visit[self.parents[node_id]]) / (visit + 1)
            return self.nodes_total_value[node_id] + u
        elif self.ucb_type == "value":
            return self.nodes_init_value[node_id] / (visit + 1)

        else:
            raise ValueError(f"I don't know this UCB type: {self.ucb_type}")

    def _select_node(self, node_id: int) -> int:
        """
        This function selects a node based on its UCB value and returns the ID of the node with the highest value of
        the UCB function.

        :param node_id: The `node_id` parameter is an integer that represents the ID of a node
        :type node_id: int
        """

        if self.epsilon > 0:
            n = uniform(0, 1)
            if n < self.epsilon:
                return choice(list(self.children[node_id]))
        return max(self.children[node_id], key=self._ucb)

    def _expand_node(self, node_id: int) -> None:
        """
        The function expands a given node by generating new retrons with policy (expansion) policy.

        :param node_id: The `node_id` parameter is an integer that represents the ID of the current node
        :type node_id: int
        """

        curr_node = self.nodes[node_id]
        prev_retrons = curr_node.curr_retron.prev_retrons

        tmp_retrons = []
        for prob, rule, rule_id in self.policy_function.predict_reaction_rules(curr_node.curr_retron,
                                                                               self.reaction_rules):
            for reaction in apply_reaction_rule(curr_node.curr_retron.molecule, rule):

                # check repeated products
                # TODO: change it to 6 and check it
                products = tuple(mol for mol in (~reaction).decompose()[1].split() if len(mol) > 0)
                if products in tmp_retrons:
                    continue
                tmp_retrons.append(products)
                #
                for reactant in products:
                    reactant.meta['reactor_id'] = rule_id
                #
                new_retrons = tuple(Retron(mol) for mol in products)
                scaled_prob = prob * len(list(filter(lambda x: len(x) > 6, products)))
                #
                if set(prev_retrons).isdisjoint(new_retrons):

                    retrons_to_expand = (*curr_node.next_retrons,
                                         *(x for x in new_retrons if not x.is_building_block(self.building_blocks)))

                    child_node = Node(retrons_to_expand=retrons_to_expand, new_retrons=new_retrons)
                    #
                    for new_retron in new_retrons:
                        new_retron.prev_retrons = [new_retron, *prev_retrons]

                    self._add_node(node_id, child_node, scaled_prob)

    def _rollout_node(self, retron: Retron, curr_depth: int = None):  # TODO it still depends on CGRTools
        """
        The function `_rollout_node` performs a rollout simulation from a given node in a tree.
        Given the current retron, find the first successful reaction and return the new retrons.

        If the retron is a building_block, return 1.0, else check the first successful reaction;

        If the reaction is not successful, return -1.0;

        If the reaction is successful, but the generated retrons are not the building_blocks and the retrons
        cannot be generated without exceeding curr_depth threshold, return -0.5;

        If the reaction is successful, but the retrons are not the building_blocks and the retrons
        cannot be generated, return -1.0;

        :param retron: A Retron object
        :type retron: Retron
        :param curr_depth: The current depth of the the tree
        :type curr_depth: int
        """

        max_depth = self.max_depth - curr_depth

        # retron checking
        if retron.is_building_block(self.building_blocks):
            reward = 1.0
            return reward

        if max_depth == 0:
            if retron.is_building_block(self.building_blocks):
                reward = 1.0
            else:
                reward = -1.0
            return reward

        # retron simulating
        occurred_retrons = set()
        retrons_to_expand = deque([retron.molecule])
        history = defaultdict(dict)
        while retrons_to_expand:

            # Iterate through reactors and pick first successful reaction.
            # Check products of the reaction if you can find them in in-building_blocks data
            # If not, then add missed products to retrons_to_expand and try to decompose them
            if len(history) >= max_depth:
                reward = 0.0  # changed from -0.5
                return reward

            current_mol = retrons_to_expand.popleft()
            history[curr_depth]["target"] = str(current_mol)
            occurred_retrons.add(current_mol)

            # Pick first successful reaction while iterating through reactors
            # Predict top-10 reactors for every molecule in simulation (time-consuming)

            reaction_rules = [(prob, rule, rule_id) for prob, rule, rule_id in
                              self.policy_function.predict_reaction_rules(Retron(current_mol), self.reaction_rules)][
                             :10]
            #
            reaction_rule_applied = False
            for prob, rule, rule_id in reaction_rules:
                for reaction in apply_reaction_rule(current_mol, rule):
                    if reaction:
                        products = [safe_canonicalization(prod) for prod in reaction.products]
                        reaction_rule_applied = True
                        break

                if reaction_rule_applied:
                    history[curr_depth]["rule_index"] = rule_id
                    break

            if not reaction_rule_applied:
                logging.debug(f"Max curr_depth limited: %s", history)
                reward = -1.0
                return reward

            history[curr_depth]["products"] = [str(res) for res in products]

            # check loops
            if any(x in occurred_retrons for x in products) and products:
                # Sometimes hardcoded_rules can create a loop, when
                logging.debug('Rollout got in the loop: %s', history)
                # print('occurred_retrons')
                reward = -1.0
                return reward

            if occurred_retrons.isdisjoint(products):
                # Added number of atoms check
                retrons_to_expand.extend(
                    [x for x in products if not Retron(x).is_building_block(self.building_blocks) and len(x) > 6])
                curr_depth += 1
        reward = 1.0
        return reward

    def _add_node(self, node_id: int, new_node: Node, policy_prob: float = None) -> None:
        """
        This function adds a new node to a tree with its policy probability.

        :param node_id: ID of the parent node
        :type node_id: int
        :param new_node: The `new_node` is an instance of the`Node` class
        :type new_node: Node
        :param policy_prob: The `policy_prob` a float value that represents the probability associated with a new node.
        :type policy_prob: float
        """

        new_node_id = self.curr_tree_size
        #
        self.nodes[new_node_id] = new_node
        self.parents[new_node_id] = node_id
        self.children[node_id].add(new_node_id)
        self.children[new_node_id] = set()
        self.nodes_visit[new_node_id] = 0
        self.nodes_prob[new_node_id] = policy_prob

        if self.evaluation_mode == 'random':
            new_node_synth = uniform()

        elif self.evaluation_mode == "rollout":
            curr_depth = self.nodes_depth[node_id]
            new_node_synth = min((self._rollout_node(retron, curr_depth=curr_depth)
                                  for retron in new_node.retrons_to_expand), default=1.0)

        elif self.evaluation_mode == 'gcn':
            new_node_synth = self.value_function.predict_value(new_node.new_retrons)

        elif self.init_new_node_value:
            new_node_synth = self.init_new_node_value
        else:
            raise ValueError(f"I don't know this evaluation mode: {self.evaluation_mode}")

        self.nodes_depth[new_node_id] = self.nodes_depth[node_id] + 1
        self.nodes_init_value[new_node_id] = new_node_synth
        self.nodes_total_value[new_node_id] = new_node_synth

        self.curr_tree_size += 1

    def _update_visits(self, node_id: int) -> None:
        """
        The function updates the number of visits from a given node to a root node.

        :param node_id: The ID of a current node
        :type node_id: int
        """

        while node_id:
            self.nodes_visit[node_id] += 1
            node_id = self.parents[node_id]

    def _backpropagate(self, node_id: int, value: float = None) -> None:
        """
        The function backpropagates a value through a tree of a given node specified by node_id.

        :param node_id: The ID of a given node from which to backpropagate value
        :type node_id: int
        :param value: The value to backpropagate
        :type value: float
        """

        while node_id:
            if self.backprop_type == "muzero":
                self.nodes_total_value[node_id] = (self.nodes_total_value[node_id] * self.nodes_visit[
                    node_id] + value) / (self.nodes_visit[node_id] + 1)
            elif self.backprop_type == "cumulative":
                self.nodes_total_value[node_id] += value
            else:
                raise ValueError(f"I don't know this backpropagation type: {self.backprop_type}")
            node_id = self.parents[node_id]

    def report(self) -> str:
        """
        Returns the string representation of the tree.
        """

        return (
            f"Tree for: {str(self.nodes[1].retrons_to_expand[0])}\n"
            f"Size: {len(self)}\nNumber of visited nodes: {len(self.visited_nodes)}\n"
            f"Found paths: {len(self.winning_nodes)}\nTime: {round(self.curr_time, 1)} seconds"
        )

    def path_score(self, node_id) -> float:
        """
        The function calculates the score of a given path from the node with node_id to the root node.

        :param node_id: The ID of a given node
        """

        cumulated_nodes_value, path_length = 0, 0
        while node_id:
            path_length += 1

            cumulated_nodes_value += self.nodes_init_value[node_id]
            node_id = self.parents[node_id]

        return cumulated_nodes_value / (path_length ** 2)

    def path_to_node(self, node_id: int) -> List:
        """
        The function returns the path (list of IDs of nodes) to from a node specified by node_id to the root node.

        :param node_id: The ID of a given node
        :type node_id: int
        """

        nodes = []
        while node_id:
            nodes.append(node_id)
            node_id = self.parents[node_id]
        return [self.nodes[node_id] for node_id in reversed(nodes)]

    # def get_subtree(self, molecule, graph) -> Dict:  # TODO this function for what ?
    #     nodes = []
    #     try:
    #         graph[molecule]
    #     except KeyError:
    #         return []
    #     for retron in graph[molecule]:
    #         temp_obj = {
    #             "smiles": str(retron),
    #             "type": "mol",
    #             "in_stock": retron.is_building_block(self.building_blocks),
    #         }
    #         node = self.get_subtree(retron, graph)
    #         if node:
    #             temp_obj["children"] = [node]
    #         nodes.append(temp_obj)
    #     return {"type": "reaction", "children": nodes}

    def synthesis_path(self, node_id: int) -> Tuple[Reaction, ...]:
        """
        Given a node_id, return a tuple of Reactions that represent the synthesis path from the
        node specified with node_id to the root node

        :param node_id: The ID of a given node
        """

        nodes = self.path_to_node(node_id)

        tmp = [Reaction([x.molecule for x in after.new_retrons], [before.curr_retron.molecule], ) for before, after in
               zip(nodes, nodes[1:])]

        for r in tmp:
            r.clean2d()
        return tuple(reversed(tmp))

    def newickify(self, visits_threshold=0, root_node_id=1):
        """
        Adopted from https://stackoverflow.com/questions/50003007/how-to-convert-python-dictionary-to-newick-form-format
        :param visits_threshold: int
        :param root_node_id: The ID of a root node
        """
        visited_nodes = set()

        def newick_render_node(current_node_id) -> str:
            """
            Recursively generates a Newick string representation of a tree

            :param current_node_id: The identifier of the current node in the tree
            :return: A string representation of a node in a Newick format
            """
            assert current_node_id not in visited_nodes, "Error: The tree may not be circular!"
            node_visit = self.nodes_visit[current_node_id]

            visited_nodes.add(current_node_id)
            if self.children[current_node_id]:
                # Nodes
                children = [child for child in list(self.children[current_node_id]) if
                            self.nodes_visit[child] >= visits_threshold]
                children_strings = [newick_render_node(child) for child in children]
                children_strings = ",".join(children_strings)
                if children_strings:
                    return f"({children_strings}){current_node_id}:{node_visit}"
                else:
                    # Leafs within threshold
                    return f"{current_node_id}:{node_visit}"
            else:
                # Leafs
                return f"{current_node_id}:{node_visit}"

        newick_string = newick_render_node(root_node_id) + ";"

        meta = {}
        for node_id in iter(visited_nodes):
            node_value = round(self.nodes_total_value[node_id], 3)

            node_synthesisability = round(self.nodes_init_value[node_id])

            visit_in_node = self.nodes_visit[node_id]
            meta[node_id] = (node_value, node_synthesisability, visit_in_node)

        return newick_string, meta
