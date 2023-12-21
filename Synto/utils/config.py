"""
Module containing training and planning configuration dictionaries
"""

from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict
from dataclasses import dataclass

import yaml


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


planning_config = {
    'General': {
        'num_cpus': 5,
        'num_gpus': 1,
        'targets_path': 'targets.smi',
        'results_root': 'synto_planning_results'},
    'InputData': {
        'reaction_rules_path': 'synto_planning_data/reaction_rules.pickle',
        'building_blocks_path': 'synto_planning_data/building_blocks.txt',
        'standardize_building_blocks': True},
    'PolicyNetwork': {
        'weights_path': 'synto_planning_data/policy_network.ckpt',
        'priority_rules_fraction': 0.5,
        'top_rules': 50,
        'rule_prob_threshold': 0.0,
        'mode': 'filtering'
    },
    'ValueNetwork': {
        'weights_path': 'synto_planning_data/value_network.ckpt'},
    'Tree': {
        'ucb_type': 'uct',
        'c_usb': 0.1,
        'max_depth': 6,
        'max_iterations': 50,
        'max_time': 120,
        'max_tree_size': 100000,
        'verbose': True,
        'evaluation_mode': 'gcn',
        'evaluation_agg': 'max',
        'backprop_type': 'muzero',
        'init_new_node_value': None}}

training_config = {
    'General': {
        'num_cpus': 5,
        'num_gpus': 1,
        'results_root': 'synto_training_data_small'},
    'InputData': {
        'building_blocks_path': 'synto_training_data_small/building_blocks/building_blocks.smi',
        'policy_data_path': 'synto_training_data_small/policy_molecules/policy_molecules.smi',
        'reaction_data_path': 'synto_training_data_small/reaction_data/reaction_data.rdf',
        'value_data_path': 'synto_training_data_small/value_molecules/value_molecules.smi'},
    'DataCleaning': {
        'clean_reactions': True,
        'standardize_building_blocks': True},
    'ReactionRules': {
        'min_popularity': 5},
    'Tree': {
        'backprop_type': 'muzero',
        'c_usb': 0.1,
        'evaluation_agg': 'max',
        'evaluation_mode': 'gcn',
        'init_new_node_value': None,
        'max_depth': 6,
        'max_iterations': 15,
        'max_time': 600,
        'max_tree_size': 1000000,
        'ucb_type': 'uct',
        'verbose': False},
    'SelfTuning': {
        'batch_size': 5,
        'num_simulations': 1},
    'PolicyNetwork': {
        'batch_size': 500,
        'dropout': 0.4,
        'learning_rate': 0.0005,
        'num_conv_layers': 5,
        'num_epoch': 100,
        'priority_rules_fraction': 0.5,
        'rule_prob_threshold': 0.0,
        'top_rules': 50,
        'vector_dim': 512,
        'mode': 'filtering'
    },
    'ValueNetwork': {
        'batch_size': 500,
        'dropout': 0.4,
        'learning_rate': 0.0005,
        'num_conv_layers': 5,
        'num_epoch': 100,
        'vector_dim': 512}}


def check_planning_config(loaded_config):
    """
    Takes planning configuration dictionary and checks if the setting parameters are correct

    :param loaded_config: The dictionary that contains the configuration settings for retrosynthetic planning
    :return: The validated configuration dictionary
    """
    updated_config = deepcopy(planning_config)
    for i, v in loaded_config.items():
        for ii, vv in v.items():
            updated_config[i][ii] = vv
    #
    assert Path(updated_config["InputData"]["reaction_rules_path"]).exists(), "Path for reaction rules does not exists"
    assert Path(updated_config["InputData"]["building_blocks_path"]).exists(), "Path for building blocks does not exists"
    assert Path(updated_config["PolicyNetwork"]["weights_path"]).exists(), "Path for Policy Network does not exists"
    assert Path(updated_config["ValueNetwork"]["weights_path"]).exists(), "Path for Value Network does not exists"

    return updated_config


def check_training_config(loaded_config):  # TODO complete assert checking
    """
    Takes training configuration dictionary and checks if the setting parameters are correct

    :param loaded_config: The dictionary that contains the configuration settings for training policy and value networks
    :return: The validated configuration dictionary
    """
    updated_config = deepcopy(training_config)
    for i, v in loaded_config.items():
        for ii, vv in v.items():
            updated_config[i][ii] = vv

    return updated_config


def read_planning_config(config_path):
    """
    Reads planning configuration file and checks if the setting parameters are correct

    :param config_path: The path to the file with configuration settings for retrosynthetic planning
    :return: The validated configuration dictionary
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    config = check_planning_config(config)
    return config


def read_training_config(config_path):
    """
    Reads training configuration file and checks if the setting parameters are correct

    :param config_path: The path to the file with configuration settings for training policy and value networks
    :return: The validated configuration dictionary
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    config = check_training_config(config)
    return config
