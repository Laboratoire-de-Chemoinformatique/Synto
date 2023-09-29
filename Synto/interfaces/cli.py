"""
Module containing commands line scripts for training and planning mode
"""

import warnings
warnings.filterwarnings("ignore")

import os
import shutil
from pathlib import Path

import click
import gdown
import yaml

from Synto.chem.reaction_rules.rule_extraction import extract_reaction_rules
from Synto.ml.training import create_policy_training_set, run_policy_training
from Synto.ml.training.self_learning import run_self_learning
from Synto.utils.config import planning_config, training_config
from Synto.utils.config import read_planning_config, read_training_config
from Synto.utils.search import tree_search


main = click.Group()


@main.command(name='download_data')
def download_data_cli():
    """
    Downloads a file from Google Drive using its remote ID, saves it as a zip file, extracts the contents,
    and then deletes the zip file
    """
    remote_id = '1omYgc95sCf4Hj9tiwnPiDk0I248NiYiZ'
    output = 'synto_data.zip'
    #
    gdown.download(output=output, id=remote_id, quiet=True)
    shutil.unpack_archive(output, './')
    #
    os.remove(output)


@main.command(name='planning_config')
def planning_config_cli():
    """
    Writes the planning configuration dictionary to a YAML file named "planning_config.yaml".
    """
    with open("planning_config.yaml", "w") as file:
        yaml.dump(planning_config, file, sort_keys=False)


@main.command(name='training_config')
def training_config_cli():
    """
    Writes the training configuration dictionary to a YAML file named "training_config.yaml".
    """
    with open("training_config.yaml", "w") as file:
        yaml.dump(training_config, file, sort_keys=False)

    return None


@main.command(name='tree_search')
@click.option(
    "--targets",
    "targets_file",
    help="Path to targets SDF molecules_path. The name of molecules_path will be used to save report.",
    type=click.Path(exists=True),
)
@click.option("--config", "config_path",
              required=True,
              help="Path to the config YAML molecules_path. To generate default config, use command Synto_default_config",
              type=click.Path(exists=True, path_type=Path),
              )
@click.option(
    "--results_root",
    help="Path to the folder where to save all statistics and results",
    required=True,
    type=click.Path(path_type=Path),
)
def synto_planning_cli(targets_file, config_path, results_root):
    """
    Launches tree search for the given target molecules and stores search statistics and found retrosynthetic paths

    :param targets_file: The path to a file that contains the list of targets for tree search
    :param config_path: The path to the configuration file that contains the settings and parameters for the tree search
    :param results_root: The root directory where the search results will be saved
    """
    config = read_planning_config(config_path)
    config['Tree']['verbose'] = False
    tree_search(results_root, targets_file, config)


@main.command(name='extract_rules')
@click.option(
    "--config", "config",
    required=True,
    help="Path to the config YAML molecules_path. To generate default config, use command Synto_default_config",
    type=click.Path(exists=True, path_type=Path),
)
def extract_reaction_cli(config):
    """
    Extracts reaction rules from a reaction data file and saves the results in a specified directory

    :param config: The configuration file that contains settings for the reaction rule extraction
    """
    config = read_training_config(config)
    extract_reaction_rules(reaction_file=config['ReactionRules']['reaction_data_path'],
                           results_root=config['ReactionRules']['results_root'])


@main.command(name='policy_training')
@click.option(
    "--config",
    "config",
    required=True,
    help="Path to the policy training config YAML molecules_path.",
    type=click.Path(exists=True, path_type=Path),
)
def policy_training_cli(config):
    """
    The function for preparation of the training set abd training a policy network

    :param config: The configuration file that contains settings for the policy training.
    specific requirements
    """
    config = read_training_config(config)

    datamodule = create_policy_training_set(config)
    run_policy_training(datamodule, config)


@main.command(name='self_learning')
@click.option(
    "--config",
    "config",
    required=True,
    help="Path to the config YAML file.",
    type=click.Path(exists=True, path_type=Path),
)
def self_learning_cli(config):
    """
    Runs a self-learning process for training value network

    :param config: The configuration file with settings for running the self-learning process
    """
    config = read_training_config(config)
    run_self_learning(config)


@main.command(name='synto_training')
@click.option(
    "--config",
    "config",
    required=True,
    help="Path to the config YAML file.",
    type=click.Path(exists=True, path_type=Path)
)
def synto_training_cli(config):

    # read training config
    config = read_training_config(config)

    # reaction rules extraction
    extract_reaction_rules(reaction_file=config['ReactionRules']['reaction_data_path'],
                           results_root=config['ReactionRules']['results_root'])

    # train policy network
    datamodule = create_policy_training_set(config)
    run_policy_training(datamodule, config)

    # self-learning value network training
    run_self_learning(config)


if __name__ == '__main__':
    main()
