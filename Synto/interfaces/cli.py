"""
Module containing commands line scripts for training and planning mode
"""

import os
import shutil
from pathlib import Path

import click
import gdown
import yaml

from Synto.chem.reaction_rules.rule_extraction import extract_reaction_rules
from Synto.training.policy_training import create_policy_training_set, run_policy_training
from Synto.training.self_learning import run_self_learning
from Synto.utils.config import planning_config, training_config
from Synto.utils.config import read_training_config
from Synto.utils.search import tree_search

# from Synto.training.self_learning import run_self_learning, run_micro_self_learning, start_simulation, tune_value_net


# training
# python Synto/interfaces/cli.py download_data
# python Synto/interfaces/cli.py planning_config
# python Synto/interfaces/cli.py training_config
# python Synto/interfaces/cli.py extract_rules --config training_config.yaml
# python Synto/interfaces/cli.py policy_training --config training_config.yaml
# python Synto/interfaces/cli.py self_learning --config training_config.yaml
# python Synto/interfaces/cli.py Synto_training --config training_config.yaml


# planning
# python Synto/interfaces/cli.py tree_search --targets="targets.sdf" --config="planning_config.yaml" --results_root="Synto_results"


main = click.Group()


# ==================================================================================================================== #


@main.command(name='download_data')
def download_data_cli():
    """
    Downloads a file from Google Drive using its remote ID, saves it as a zip file, extracts the contents,
    and then deletes the zip file
    """
    remote_id = '1H3A28VAJ0jlur6de2CwI4vWxDGtsEx4b'
    output = 'Synto_data.zip'
    #
    gdown.download(output=output, id=remote_id, quiet=True)
    shutil.unpack_archive(output, './')
    #
    os.remove(output)


# ==================================================================================================================== #


@main.command(name='planning_config')
def planning_config_cli():
    """
    Writes the planning configuration dictionary to a YAML file named "planning_config.yaml".
    """
    with open("planning_config.yaml", "w") as file:
        yaml.dump(planning_config, file, sort_keys=False)


# ==================================================================================================================== #


@main.command(name='training_config')
def training_config_cli():
    """
    Writes the training configuration dictionary to a YAML file named "training_config.yaml".
    """
    with open("training_config.yaml", "w") as file:
        yaml.dump(training_config, file, sort_keys=False)

    return None


# ==================================================================================================================== #


@main.command(name='tree_search')
@click.option(
    "--targets",
    "targets_file",
    help="Path to targets SDF molecules_path. The name of molecules_path will be used to save report.",
    type=click.Path(exists=True),
)
@click.option("--config", "config",
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
def tree_search_cli(targets_file, config_path, results_root):
    """
    Launches tree search for the given target molecules and stores search statistics and found retrosynthetic paths

    :param targets_file: The path to a file that contains the list of targets for tree search
    :param config_path: The path to the configuration file that contains the settings and parameters for the tree search
    :param results_root: The root directory where the search results will be saved
    """
    tree_search(results_root, targets_file, config_path)  # TODO set config not path


# ==================================================================================================================== #


@main.command(name='extract_rules')
@click.option(
    "--config", "config",
    required=True,
    help="Path to the config YAML molecules_path. To generate default config, use command Synto_default_config",
    type=click.Path(exists=True, path_type=Path),
)
def extract_reaction_cli(config):  # TODO reaction_rules_file name correct in GGRtools
    """
    Extracts reaction rules from a reaction data file and saves the results in a specified directory

    :param config: The configuration file that contains settings for the reaction rule extraction
    """
    config = read_training_config(config)
    extract_reaction_rules(reaction_file=config['ReactionRules']['reaction_data_path'],
                           results_root=config['ReactionRules']['results_root'])


# ==================================================================================================================== #


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


# ==================================================================================================================== #


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


# ==================================================================================================================== #


@main.command(name='Synto_training')
@click.option(
    "--config",
    "config",
    required=True,
    help="Path to the config YAML file.",
    type=click.Path(exists=True, path_type=Path)
)
# def Synto_training(config):
#
#     # read training config
#     config = read_training_config(config)
#
#     # reaction rules extraction
#     extract_reaction_rules(reaction_file=config['ReactionRules']['reaction_data_path'],
#                            results_root=config['ReactionRules']['results_root'])
#
#     # train policy network
#     datamodule = create_policy_training_set(config)
#     run_policy_training(datamodule, config)
#
#     # self-learning value network training
#     run_self_learning(config)
#
def Synto_training(config):
    """
    Performs all end-to-end training steps for the preparation of policy and value networks

    :param config: The configuration file that contains settings for the training process
    """
    import time

    with open('Synto_tmp/training_log.txt', 'w') as fw:
        fw.write('Training started\n\n')
        fw.flush()

        # read training config
        config = read_training_config(config)

        # # reaction rules extraction
        # start = time.time()
        #
        # extract_reaction_rules(reaction_file=config['ReactionRules']['reaction_data_path'],
        #                        results_root=config['ReactionRules']['results_root'])
        #
        # end = time.time()
        # fw.write(f'Reaction rules extraction finished ({round((end - start) / 3600, 2)} h)\n\n')
        # fw.flush()

        # create policy training set
        start = time.time()

        datamodule = create_policy_training_set(config)

        end = time.time()
        fw.write(f'Policy training set creating finished ({round((end - start) / 3600, 2)} h)\n\n')
        fw.flush()

        # train policy network
        start = time.time()

        run_policy_training(datamodule, config)

        end = time.time()
        fw.write(f'Policy training finished ({round((end - start) / 3600, 2)} h)\n\n')
        fw.flush()

        # self-learning value network training
        start = time.time()

        run_self_learning(config)

        end = time.time()
        fw.write(f'Self-learning finished ({round((end - start) / 3600, 2)} h)\n')
        fw.flush()


# ==================================================================================================================== #


if __name__ == '__main__':
    main()
