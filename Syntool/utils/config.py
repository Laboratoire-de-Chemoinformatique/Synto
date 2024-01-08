"""
Module containing training and planning configuration dictionaries
"""

import yaml
from abc import ABC, abstractmethod
from typing import List, Union, Tuple, IO, Dict, Set, Iterable, Any
import yaml
from CGRtools.containers import MoleculeContainer, QueryContainer, ReactionContainer
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict
from Syntool.utils import path_type


@dataclass
class ConfigABC(ABC):
    """
    Abstract base class for configuration classes.
    """

    @staticmethod
    @abstractmethod
    def from_dict(config_dict: Dict[str, Any]):
        """
        Create an instance of the configuration from a dictionary.
        """
        pass

    @staticmethod
    @abstractmethod
    def from_yaml(file_path: str):
        """
        Deserialize a YAML file into a configuration object.
        """
        pass

    @abstractmethod
    def _validate_params(self, params: Dict[str, Any]):
        """
        Validate configuration parameters.
        """
        pass

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the configuration into a dictionary.

        Returns:
            A dictionary representation of the ConfigABC instance.
        """
        return {k: str(v) if isinstance(v, Path) else v for k, v in self.__dict__.items()}

    def to_yaml(self, file_path: str):
        """
        Serialize the configuration to a YAML file.

        Args:
            file_path: Path where the YAML file will be saved.
        """
        with open(file_path, "w") as file:
            yaml.dump(self.to_dict(), file)

    def __post_init__(self):
        # Call _validate_params method after initialization
        params = self.to_dict()  # Convert the current instance to a dictionary
        self._validate_params(params)


@dataclass
class ExtractRuleConfig(ConfigABC):
    """
    Configuration class for extracting reaction rules, inheriting from ConfigABC.

    :ivar multicenter_rules: If True, extracts a single rule encompassing all centers.
                             If False, extracts separate reaction rules for each reaction center in a multicenter reaction.
    :ivar as_query_container: If True, the extracted rules are generated as QueryContainer objects,
                              analogous to SMARTS objects for pattern matching in chemical structures.
    :ivar reverse_rule: If True, reverses the direction of the reaction for rule extraction.
    :ivar reactor_validation: If True, validates each generated rule in a chemical reactor to ensure correct
                              generation of products from reactants.
    :ivar include_func_groups: If True, includes specific functional groups in the reaction rule in addition
                               to the reaction center and its environment.
    :ivar func_groups_list: A list of functional groups to be considered when include_func_groups is True.
    :ivar include_rings: If True, includes ring structures in the reaction rules.
    :ivar keep_leaving_groups: If True, retains leaving groups in the extracted reaction rule.
    :ivar keep_incoming_groups: If True, retains incoming groups in the extracted reaction rule.
    :ivar keep_reagents: If True, includes reagents in the extracted reaction rule.
    :ivar environment_atom_count: Defines the size of the environment around the reaction center to be included
                                  in the rule (0 for only the reaction center, 1 for the first environment, etc.).
    :ivar min_popularity: Minimum number of times a rule must be applied to be considered for further analysis.
    :ivar keep_metadata: If True, retains metadata associated with the reaction in the extracted rule.
    :ivar single_reactant_only: If True, includes only reaction rules with a single reactant molecule.
    :ivar atom_info_retention: Controls the amount of information about each atom to retain ('none',
                                'reaction_center', or 'all').
    """

    multicenter_rules: bool = True
    as_query_container: bool = True
    reverse_rule: bool = True
    reactor_validation: bool = True
    include_func_groups: bool = False
    func_groups_list: List[Union[MoleculeContainer, QueryContainer]] = field(default_factory=list)
    include_rings: bool = False
    keep_leaving_groups: bool = True
    keep_incoming_groups: bool = False
    keep_reagents: bool = False
    environment_atom_count: int = 1
    min_popularity: int = 3
    keep_metadata: bool = False
    single_reactant_only: bool = True
    atom_info_retention: Dict[str, Dict[str, bool]] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        self._validate_params(self.to_dict())
        self._initialize_default_atom_info_retention()

    def _initialize_default_atom_info_retention(self):
        default_atom_info = {
            "reaction_center": {
                "neighbors": True,
                "hybridization": True,
                "implicit_hydrogens": True,
                "ring_sizes": True,
            },
            "environment": {
                "neighbors": True,
                "hybridization": True,
                "implicit_hydrogens": True,
                "ring_sizes": True,
            },
        }

        if not self.atom_info_retention:
            self.atom_info_retention = default_atom_info
        else:
            for key in default_atom_info:
                self.atom_info_retention[key].update(
                    self.atom_info_retention.get(key, {})
                )

    @staticmethod
    def from_dict(config_dict: Dict[str, Any]):
        """
        Creates an ExtractRuleConfig instance from a dictionary of configuration parameters.

        :ivar config_dict: A dictionary containing configuration parameters.
        :return: An instance of ExtractRuleConfig.
        """
        return ExtractRuleConfig(**config_dict)

    @staticmethod
    def from_yaml(file_path: str):
        """
        Deserializes a YAML file into an ExtractRuleConfig object.

        :ivar file_path: Path to the YAML file containing configuration parameters.
        :return: An instance of ExtractRuleConfig.
        """
        with open(file_path, "r") as file:
            config_dict = yaml.safe_load(file)
        return ExtractRuleConfig.from_dict(config_dict)

    def _validate_params(self, params: Dict[str, Any]):
        """
        Validate the parameters of the configuration.
        """
        if not isinstance(params["multicenter_rules"], bool):
            raise ValueError("multicenter_rules must be a boolean.")

        if not isinstance(params["as_query_container"], bool):
            raise ValueError("as_query_container must be a boolean.")

        if not isinstance(params["reverse_rule"], bool):
            raise ValueError("reverse_rule must be a boolean.")

        if not isinstance(params["reactor_validation"], bool):
            raise ValueError("reactor_validation must be a boolean.")

        if not isinstance(params["include_func_groups"], bool):
            raise ValueError("include_func_groups must be a boolean.")

        if params["func_groups_list"] is not None and not all(
            isinstance(group, (MoleculeContainer, QueryContainer))
            for group in params["func_groups_list"]
        ):
            raise ValueError(
                "func_groups_list must be a list of MoleculeContainer or QueryContainer objects."
            )

        if not isinstance(params["include_rings"], bool):
            raise ValueError("include_rings must be a boolean.")

        if not isinstance(params["keep_leaving_groups"], bool):
            raise ValueError("keep_leaving_groups must be a boolean.")

        if not isinstance(params["keep_incoming_groups"], bool):
            raise ValueError("keep_incoming_groups must be a boolean.")

        if not isinstance(params["keep_reagents"], bool):
            raise ValueError("keep_reagents must be a boolean.")

        if not isinstance(params["environment_atom_count"], int):
            raise ValueError("environment_atom_count must be an integer.")

        if not isinstance(params["min_popularity"], int):
            raise ValueError("min_popularity must be an integer.")

        if not isinstance(params["keep_metadata"], bool):
            raise ValueError("keep_metadata must be a boolean.")

        if not isinstance(params["single_reactant_only"], bool):
            raise ValueError("single_reactant_only must be a boolean.")

        if params["atom_info_retention"] is not None:
            if not isinstance(params["atom_info_retention"], dict):
                raise ValueError("atom_info_retention must be a dictionary.")

            required_keys = {"reaction_center", "environment"}
            if not required_keys.issubset(params["atom_info_retention"]):
                missing_keys = required_keys - set(params["atom_info_retention"].keys())
                raise ValueError(
                    f"atom_info_retention missing required keys: {missing_keys}"
                )

            for key, value in params["atom_info_retention"].items():
                if key not in required_keys:
                    raise ValueError(f"Unexpected key in atom_info_retention: {key}")

                expected_subkeys = {
                    "neighbors",
                    "hybridization",
                    "implicit_hydrogens",
                    "ring_sizes",
                }
                if not isinstance(value, dict) or not expected_subkeys.issubset(value):
                    missing_subkeys = expected_subkeys - set(value.keys())
                    raise ValueError(
                        f"Invalid structure for {key} in atom_info_retention. Missing subkeys: {missing_subkeys}"
                    )

                for subkey, subvalue in value.items():
                    if not isinstance(subvalue, bool):
                        raise ValueError(
                            f"Value for {subkey} in {key} of atom_info_retention must be boolean."
                        )


@dataclass
class TreeConfig(ConfigABC):
    """
    Configuration class for the tree-based search algorithm, inheriting from ConfigABC.

    :ivar max_iterations: The number of iterations to run the algorithm for, defaults to 100.
    :ivar max_tree_size: The maximum number of nodes in the tree, defaults to 10000.
    :ivar max_time: The time limit (in seconds) for the algorithm to run, defaults to 600.
    :ivar max_depth: The maximum depth of the tree, defaults to 6.
    :ivar ucb_type: Type of UCB used in the search algorithm. Options are "puct", "uct", "value", defaults to "uct".
    :ivar c_ucb: The exploration-exploitation balance coefficient used in Upper Confidence Bound (UCB), defaults to 0.1.
    :ivar backprop_type: Type of backpropagation algorithm. Options are "muzero", "cumulative", defaults to "muzero".
    :ivar search_strategy: The strategy used for tree search. Options are "expansion_first", "evaluation_first", defaults to "expansion_first".
    :ivar exclude_small: Whether to exclude small molecules during the search, defaults to True.
    :ivar evaluation_agg: Method for aggregating evaluation scores. Options are "max", "average", defaults to "max".
    :ivar evaluation_mode: The method used for evaluating nodes. Options are "random", "rollout", "gcn", defaults to "gcn".
    :ivar init_node_value: Initial value for a new node, defaults to 0.0.
    :ivar epsilon: A parameter in the epsilon-greedy search strategy representing the chance of random selection
    of reaction rules during the selection stage in Monte Carlo Tree Search,
    specifically during Upper Confidence Bound estimation.
    It balances between exploration and exploitation, defaults to 0.0.
    :ivar min_mol_size: Defines the minimum size of a molecule that is have to be synthesized.
    Molecules with 6 or fewer heavy atoms are assumed to be building blocks by definition,
    thus setting the threshold for considering larger molecules in the search, defaults to 6.
    :ivar silent: Whether to suppress progress output, defaults to False.
    """

    max_iterations: int = 100
    max_tree_size: int = 10000
    max_time: float = 600
    max_depth: int = 6
    ucb_type: str = "uct"
    c_ucb: float = 0.1
    backprop_type: str = "muzero"
    search_strategy: str = "expansion_first"
    exclude_small: bool = True
    evaluation_agg: str = "max"
    evaluation_mode: str = "gcn"
    init_node_value: float = 0.0
    epsilon: float = 0.0
    min_mol_size: int = 6
    silent: bool = False

    @staticmethod
    def from_dict(config_dict: Dict[str, Any]):
        """
        Creates a TreeConfig instance from a dictionary of configuration parameters.

        Args:
            config_dict: A dictionary containing configuration parameters.

        Returns:
            An instance of TreeConfig.
        """
        return TreeConfig(**config_dict)

    @staticmethod
    def from_yaml(file_path: str):
        """
        Deserializes a YAML file into a TreeConfig object.

        Args:
            file_path: Path to the YAML file containing configuration parameters.

        Returns:
            An instance of TreeConfig.
        """
        with open(file_path, "r") as file:
            config_dict = yaml.safe_load(file)
        return TreeConfig.from_dict(config_dict)

    def _validate_params(self, params):
        if params["ucb_type"] not in ["puct", "uct", "value"]:
            raise ValueError(
                "Invalid ucb_type. Allowed values are 'puct', 'uct', 'value'."
            )
        if params["backprop_type"] not in ["muzero", "cumulative"]:
            raise ValueError(
                "Invalid backprop_type. Allowed values are 'muzero', 'cumulative'."
            )
        if params["evaluation_mode"] not in ["random", "rollout", "gcn"]:
            raise ValueError(
                "Invalid evaluation_mode. Allowed values are 'random', 'rollout', 'gcn'."
            )
        if params["evaluation_agg"] not in ["max", "average"]:
            raise ValueError(
                "Invalid evaluation_agg. Allowed values are 'max', 'average'."
            )
        if not isinstance(params["c_ucb"], float):
            raise TypeError("c_ucb must be a float.")
        if not isinstance(params["max_depth"], int) or params["max_depth"] < 1:
            raise ValueError("max_depth must be a positive integer.")
        if not isinstance(params["max_tree_size"], int) or params["max_tree_size"] < 1:
            raise ValueError("max_tree_size must be a positive integer.")
        if (
            not isinstance(params["max_iterations"], int)
            or params["max_iterations"] < 1
        ):
            raise ValueError("max_iterations must be a positive integer.")
        if not isinstance(params["max_time"], int) or params["max_time"] < 1:
            raise ValueError("max_time must be a positive integer.")
        if not isinstance(params["silent"], bool):
            raise TypeError("silent must be a boolean.")
        if not isinstance(params["init_node_value"], float):
            raise TypeError("init_node_value must be a float if provided.")
        if params["search_strategy"] not in ["expansion_first", "evaluation_first"]:
            raise ValueError(
                f"Invalid search_strategy: {params['search_strategy']}: "
                f"Allowed values are 'expansion_first', 'evaluation_first'"
            )

@dataclass
class PolicyNetworkConfig(ConfigABC):
    """
    Configuration class for the policy network, inheriting from ConfigABC.

    :ivar vector_dim: simension of the input vectors.
    :ivar batch_size: number of samples per batch.
    :ivar dropout: dropout rate for regularization.
    :ivar learning_rate: learning rate for the optimizer.
    :ivar num_conv_layers: number of convolutional layers in the network.
    :ivar num_epoch: number of training epochs.
    :ivar policy_type: mode of operation, either 'filtering' or 'ranking'.
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


def convert_config_to_dict(config_attr, config_type):
    """
    Converts a configuration attribute to a dictionary if it's either a dictionary
    or an instance of a specified configuration type.

    :param config_attr: The configuration attribute to be converted.
    :param config_type: The type to check against for conversion.
    :return: The configuration attribute as a dictionary, or None if it's not an instance of the given type or dict.
    """
    if isinstance(config_attr, dict):
        return config_attr
    elif isinstance(config_attr, config_type):
        return config_attr.to_dict()
    return None


def read_planning_config(config_path):
    """
    Reads planning configuration file and checks if the setting parameters are correct.

    :param config_path: the path to the file with configuration settings for retrosynthetic planning
    :return: the validated configuration dictionary
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def read_training_config(config_path):
    """
    Reads training configuration file and checks if the setting parameters are correct

    :param config_path: the path to the file with configuration settings for training policy and value networks
    :return: the validated configuration dictionary.
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config
