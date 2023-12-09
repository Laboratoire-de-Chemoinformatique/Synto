"""
Module containing training and planning configuration dictionaries
"""

import yaml

from copy import deepcopy
from pathlib import Path
from os import getcwd, path

planning_config = {
    "General": {
        "reaction_rules_path": None,
        "building_blocks_path": None,
    },
    "Tree": {
        "ucb_type": "uct",
        "c_usb": 0.1,
        "max_depth": 6,
        "max_iterations": 100,
        "max_time": 600,
        "max_tree_size": 1e6,
        "verbose": False,
        "evaluation_mode": "gcn",
        "evaluation_agg": "max",
        "backprop_type": "muzero",
        "init_new_node_value": None,
        "epsilon": 0.0
    },
    "PolicyNetwork": {
        "weights_path": None,
        "priority_rules_fraction": 0.5,
        "top_rules": 50,
        "rule_prob_threshold": 0.0
    },
    "ValueNetwork": {
        "weights_path": None
    },
}

cdw = getcwd()
if path.exists(cdw + '/synto_planning_data/reaction_rules.pickle'):
    planning_config["General"]["reaction_rules_path"] = cdw + '/synto_planning_data/reaction_rules.pickle'
if path.exists(cdw + '/synto_planning_data/building_blocks.pickle'):
    planning_config["General"]["building_blocks_path"] = cdw + '/synto_planning_data/building_blocks.txt'
if path.exists(cdw + '/synto_planning_data/policy_network.ckpt'):
    planning_config["PolicyNetwork"]["weights_path"] = cdw + '/synto_planning_data/policy_network.ckpt'
if path.exists(cdw + '/synto_planning_data/value_network.ckpt'):
    planning_config["ValueNetwork"]["weights_path"] = cdw + '/synto_planning_data/value_network.ckpt'


training_config = {
    'General': {
        'results_root': None,
        'building_blocks_path': None,
        'reaction_rules_path': None,
        'num_cpus': 10,
        'num_gpus': 1},
    'Tree': {
        'ucb_type': 'uct',
        'c_usb': 0.1,
        'max_depth': 9,
        'max_iterations': 5,
        'max_time': 120,
        'max_tree_size': 1e6,
        'verbose': False,
        'evaluation_mode': 'gcn',
        'evaluation_agg': 'max',
        'backprop_type': 'muzero',
        'init_new_node_value': None,
        'epsilon': 0.0},
    'DataCleaning': {
        'standardize_reactions': True,
        'reaction_data_path': None,
        'standardized_reactions_path': True},
    'ReactionRules': {
        'results_root': None,
        'reaction_data_path': None,
        'min_popularity': 10},
    'PolicyNetwork': {
        'results_root': None,
        'dataset_path': None,
        'datamodule_path': None,
        'weights_path': None,
        'priority_rules_fraction': 0.5,
        "top_rules": 50,
        "rule_prob_threshold": 0.0,
        'num_conv_layers': 5,
        'vector_dim': 512,
        'dropout': 0.4,
        'learning_rate': 0.0005,
        'num_epoch': 100,
        'batch_size': 500},
    'SelfTuning': {
        'results_root': None,
        'dataset_path': None,
        'num_simulations': 1,
        'batch_size': 5},
    'ValueNetwork': {
        'weights_path': None,
        'num_conv_layers': 5,
        'vector_dim': 512,
        'dropout': 0.4,
        'learning_rate': 0.0005,
        'num_epoch': 100,
        'batch_size': 500}
}


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
    assert Path(updated_config["General"]["reaction_rules_path"]).exists(), "Path for reaction rules does not exists"
    assert Path(updated_config["General"]["building_blocks_path"]).exists(), "Path for building blocks does not exists"
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
