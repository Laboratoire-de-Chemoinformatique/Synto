"""
Module containing training and planning configuration dictionaries
"""

import os
import yaml
from pathlib import Path
from copy import deepcopy


planning_config = {
    "General": {
        "reaction_rules_path": os.path.abspath('gslretro_training/reaction_rules/reaction_rules.pickle'),
        "building_blocks_path": os.path.abspath('gslretro_training/building_blocks/building_blocks.pickle'),
    },
    "Tree": {
        "ucb_type": "UCT",
        "c_usb": 0.1,
        "max_depth": 6,
        "max_iterations": 10,
        "max_time": 120,
        "max_tree_size": 1e6,
        "verbose": False,
        "evaluation_mode": "gcn",
        "evaluation_agg": "max",
        "backprop_type": "muzero",
        "init_new_node_value": None,
        "epsilon": 0.0
    },
    "PolicyNetwork": {
        "weights_path": os.path.abspath('gslretro_training/policy_network/weights/policy_network.ckpt'),
        "priority_rules_fraction": 0.5,
        "top_rules": 50,
        "rule_prob_threshold": 0.0
    },
    "ValueNetwork": {
        "weights_path": os.path.abspath('data/value_net.ckpt')
    },
}


# training_config = {
#     "General": {
#         "results_folder": 'gslretro_training',
#         "reaction_data_path": 'gslretro_training/reaction_data/reaction_data.rdf',
#         "policy_molecules_path": 'gslretro_training/policy_molecules/policy_molecules.sdf',
#         "building_blocks_path": 'gslretro_training/building_blocks/building_blocks.pickle',
#         "reaction_rules_path": 'gslretro_training/reaction_rules/reaction_rules.pickle',
#         "num_cpus": 20,
#         "gpu": True
#     },
#     "Tree": {
#         "ucb_type": "UCT",
#         "c_usb": 0.1,
#         "max_depth": 6,
#         "max_iterations": 10,
#         "max_time": 120,
#         "max_tree_size": 1e6,
#         "verbose": False,
#         "evaluation_mode": "gcn",
#         "evaluation_agg": "max",
#         "backprop_type": "muzero",
#         "init_new_node_value": None,
#         "epsilon": 0.0
#     },
#     'PolicyNetwork': {
#         "results_path": 'gslretro_training/policy_network',
#         'weights_path': None,
#         "policy_dataset_path": 'gslretro_training/policy_network/policy_dataset.pt',
#         "priority_rules_fraction": 0.5,
#         "top_rules": 50,
#         "rule_prob_threshold": 0.0,
#         "vector_dim": 512,
#         "num_conv_layers": 5,
#         "dropout": 0.4,
#         "learning_rate": 0.0005,
#         "n_epoch": 100,
#         "batch_size": 500
#     },
#     "ValueNetwork": {
#         "results_path": 'gslretro_training/value_network',
#         'weights_path': None,
#         'vector_dim': 512,
#         "num_conv_layers": 5,
#         "dropout": 0.4,
#         "learning_rate": 0.0005,
#         "n_epoch": 100,
#         "batch_size": 500
#     }
# }


training_config = {
    'General': {
        'results_root': 'gslretro_training',
        'building_blocks_path': 'gslretro_training/building_blocks/building_blocks.pickle',
        'num_cpus': 20,
        'num_gpus': 1},

        'Tree': {
           'ucb_type': 'UCT',
           'c_usb': 0.1,
           'max_depth': 6,
           'max_iterations': 50,
           'max_time': 120,
           'max_tree_size': 1000000,
           'verbose': False,
           'evaluation_mode': 'gcn',
           'evaluation_agg': 'max',
           'backprop_type': 'muzero',
           'init_new_node_value': None,
           'epsilon': 0.0},

    'ReactionRules': {
        'reaction_data_path': 'gslretro_training/reaction_data/reaction_data.rdf',
        'reaction_rules_path': 'gslretro_training/reaction_rules/reaction_rules.pickle'},

    'SelfLearning': {
        'results_root': 'gslretro_training/value_network',
        'dataset_path': 'gslretro_training/value_molecules/value_molecules.sdf',
        'num_simulations': 1,
        'batch_size': 5,
        'balance_positive': False},

    'PolicyNetwork': {
      'results_root': 'gslretro_training/policy_network',
      'dataset_path': 'gslretro_training/policy_molecules/policy_molecules.sdf',
      'datamodule_path': 'gslretro_training/policy_network/policy_dataset.pt',
      'weights_path': 'data/policy_net.ckpt',
      'priority_rules_fraction': 0.5,
      'num_conv_layers': 5,
      'vector_dim': 512,
      'dropout': 0.4,
      'learning_rate': 0.0005,
      'num_epoch': 100,
      'batch_size': 500},

    'ValueNetwork': {
      'results_root': 'gslretro_training/value_network',
      'weights_path': 'gslretro_training/value_network/value_network.ckpt',
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
    assert Path(updated_config["PolicyNetwork"]["weights_path"]).exists(), "Path for Policy Network does not exists"
    assert Path(updated_config["ValueNetwork"]["weights_path"]).exists(), "Path for Value Network does not exists"
    assert Path(updated_config["General"]["reaction_rules_path"]).exists(), "Path for reaction rules does not exists"
    assert Path(updated_config["General"]["building_blocks_path"]).exists(), "Path for building blocks does not exists"

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
