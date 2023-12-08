"""
Module containing commands line scripts for training and planning mode
"""

import warnings
import os
import shutil
from pathlib import Path

import click
import gdown

from Synto.chem.reaction_rules.extraction import extract_rules_from_reactions
from Synto.ml.training import create_policy_training_set, run_policy_training
from Synto.ml.training.reinforcement import run_self_tuning
from Synto.utils.loading import canonicalize_building_blocks
from Synto.utils.config import read_planning_config, read_training_config
from Synto.utils.search import tree_search

main = click.Group()
warnings.filterwarnings("ignore")


@main.command(name='planning_data')
def planning_data_cli():
    """
    Downloads a file from Google Drive using its remote ID, saves it as a zip file, extracts the contents,
    and then deletes the zip file
    """
    remote_id = '1c5YJDT-rP1ZvFA-ELmPNTUj0b8an4yFf'
    output = 'synto_planning_data.zip'
    #
    gdown.download(output=output, id=remote_id, quiet=True)
    shutil.unpack_archive(output, './')
    #
    os.remove(output)


@main.command(name='training_data')
def training_data_cli():
    """
    Downloads a file from Google Drive using its remote ID, saves it as a zip file, extracts the contents,
    and then deletes the zip file
    """
    remote_id = '1r4I7OskGvzg-zxYNJ7WVYpVR2HSYW10N'
    output = 'synto_training_data.zip'
    #
    gdown.download(output=output, id=remote_id, quiet=True)
    shutil.unpack_archive(output, './')
    #
    os.remove(output)


@main.command(name='building_blocks')
@click.option("--input", "input_file", required=True, help="Path to the file with original building blocks",
    type=click.Path(exists=True), )
@click.option("--output", "output_file", required=True, help="Path to the file with processed building blocks",
    type=click.Path(exists=True), )
def building_blocks_cli(input_file, output_file):
    """
    Canonicalizes custom building blocks
    """
    canonicalize_building_blocks(input_file, output_file)


@main.command(name='tree_search')
@click.option("--targets", "targets_file",
    help="Path to targets SDF molecules_path. The name of molecules_path will be used to save report.",
    type=click.Path(exists=True), )
@click.option("--config", "config_path", required=True,
              help="Path to the config YAML molecules_path. To generate default config, use command Synto_default_config",
              type=click.Path(exists=True, path_type=Path), )
@click.option("--results_root", help="Path to the folder where to save all statistics and results", required=True,
    type=click.Path(path_type=Path), )
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
@click.option("--config", "config", required=True,
    help="Path to the config YAML molecules_path. To generate default config, use command Synto_default_config",
    type=click.Path(exists=True, path_type=Path), )
def extract_rules_cli(config):
    """
    Extracts reaction rules from a reaction data file and saves the results in a specified directory

    :param config: The configuration file that contains settings for the reaction rule extraction
    """
    config = read_training_config(config)
    extract_rules_from_reactions(reaction_file=config['ReactionRules']['reaction_data_path'],
                           results_root=config['ReactionRules']['results_root'],
                           min_popularity=config['ReactionRules']['min_popularity'])


@main.command(name='policy_training')
@click.option("--config", "config", required=True, help="Path to the policy training config YAML molecules_path.",
    type=click.Path(exists=True, path_type=Path), )
def policy_training_cli(config):
    """
    The function for preparation of the training set abd training a policy network

    :param config: The configuration file that contains settings for the policy training.
    specific requirements
    """
    config = read_training_config(config)

    datamodule = create_policy_training_set(config)
    run_policy_training(datamodule, config)


@main.command(name='self_tuning')
@click.option("--config", "config", required=True, help="Path to the config YAML file.",
    type=click.Path(exists=True, path_type=Path), )
def self_tuning_cli(config):
    """
    Runs a self-tuning process for training value network

    :param config: The configuration file with settings for running the self-tuning process
    """
    config = read_training_config(config)
    run_self_tuning(config)


@main.command(name='synto_training')
@click.option("--config", "config", required=True, help="Path to the config YAML file.",
    type=click.Path(exists=True, path_type=Path))
def synto_training_cli(config):
    # read training config
    print('READ CONFIG ...')
    config = read_training_config(config)
    print('Config is read')

    # reaction rules extraction
    print('\nEXTRACT REACTION RULES ...')
    extract_rules_from_reactions(reaction_file=config['ReactionRules']['reaction_data_path'],
                           results_root=config['ReactionRules']['results_root'],
                           min_popularity=config['ReactionRules']['min_popularity'])

    # create policy network dataset
    print('\nCREATE POLICY NETWORK DATASET ...')
    datamodule = create_policy_training_set(config)

    # train policy network
    print('\nTRAIN POLICY NETWORK ...')
    run_policy_training(datamodule, config)

    # self-tuning value network training
    print('\nTRAIN VALUE NETWORK ...')
    run_self_tuning(config)


if __name__ == '__main__':
    main()
