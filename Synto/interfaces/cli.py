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
from Synto.chem.data.cleaning import reactions_cleaner
from Synto.ml.training import create_policy_dataset, run_policy_training
from Synto.ml.training.reinforcement import run_self_tuning
from Synto.chem.loading import standardize_building_blocks
from Synto.utils.config import read_planning_config, read_training_config
from Synto.mcts.search import tree_search

from Synto.ml.networks.policy import PolicyNetworkConfig
from Synto.utils.config import TreeConfig

warnings.filterwarnings("ignore")
main = click.Group()


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


@main.command(name='synto_planning')
@click.option("--config",
              "config_path",
              required=True,
              help="Path to the config YAML molecules_path. To generate default config, use command Synto_default_config",
              type=click.Path(exists=True, path_type=Path),
              )
def synto_planning_cli(config_path):
    """
    Launches tree search for the given target molecules and stores search statistics and found retrosynthetic paths

    :param config_path: The path to the configuration file that contains the settings and parameters for the tree search
    """
    config = read_planning_config(config_path)
    config['Tree']['silent'] = True

    # standardize building blocks
    if config['InputData']['standardize_building_blocks']:
        print('STANDARDIZE BUILDING BLOCKS ...')
        standardize_building_blocks(config['InputData']['building_blocks_path'],
                                    config['InputData']['building_blocks_path'])
    # run planning
    print('\nRUN PLANNING ...')
    tree_config = TreeConfig.from_dict(config['Tree'])
    tree_search(targets=config['General']['targets_path'],
                tree_config=tree_config,
                reaction_rules_path=config['InputData']['reaction_rules_path'],
                building_blocks_path=config['InputData']['building_blocks_path'],
                policy_weights_path=config['PolicyNetwork']['weights_path'],
                value_weights_paths=config['ValueNetwork']['weights_path'],
                results_root=config['General']['results_root'])


@main.command(name='synto_training')
@click.option(
    "--config",
    "config_path",
    required=True,
    help="Path to the config YAML file.",
    type=click.Path(exists=True, path_type=Path)
             )
def synto_training_cli(config_path):

    # read training config
    print('READ CONFIG ...')
    config = read_training_config(config_path)
    print('Config is read')

    # reaction mapping
    pass

    # reaction data cleaning
    if config['DataCleaning']['clean_reactions']:
        print('\nCLEAN REACTION DATA ...')

        data_output_folder = os.path.join(config['General']['results_root'], 'reaction_data')
        Path(data_output_folder).mkdir(parents=True, exist_ok=True)
        cleaned_data_file = os.path.join(data_output_folder, 'reaction_data_cleaned.rdf')

        reactions_cleaner(input_file=config['InputData']['reaction_data_path'],
                          output_file=cleaned_data_file,
                          num_cpus=config['General']['num_cpus'])

    # standardize building blocks
    if config['DataCleaning']['standardize_building_blocks']:
        print('\nSTANDARDIZE BUILDING BLOCKS ...')

        standardize_building_blocks(config['InputData']['building_blocks_path'],
                                    config['InputData']['building_blocks_path'])

    # reaction rules extraction
    print('\nEXTRACT REACTION RULES ...')
    if config['DataCleaning']['clean_reactions']:
        reaction_file = cleaned_data_file
    else:
        reaction_file = config['InputData']['reaction_data_path']

    rules_output_folder = os.path.join(config['General']['results_root'], 'reaction_rules')
    Path(rules_output_folder).mkdir(parents=True, exist_ok=True)
    reaction_rules_path = os.path.join(rules_output_folder, 'reaction_rules_filtered.pickle')
    config['InputData']['reaction_rules_path'] = reaction_rules_path

    extract_rules_from_reactions(config=config,
                                 reaction_file=reaction_file,
                                 results_root=rules_output_folder,
                                 num_cpus=config['General']['num_cpus'])

    # create policy network dataset
    print('\nCREATE POLICY NETWORK DATASET ...')
    policy_output_folder = os.path.join(config['General']['results_root'], 'policy_network')
    Path(policy_output_folder).mkdir(parents=True, exist_ok=True)
    policy_data_file = os.path.join(policy_output_folder, 'policy_dataset.pt')

    if config['PolicyNetwork']['policy_type'] == 'ranking':
        molecules_or_reactions_path = reaction_file
    elif config['PolicyNetwork']['policy_type'] == 'filtering':
        molecules_or_reactions_path = config['InputData']['policy_data_path']
    else:
        raise ValueError(
            "Invalid policy_type. Allowed values are 'ranking', 'filtering'."
        )
    datamodule = create_policy_dataset(reaction_rules_path=reaction_rules_path,
                                       molecules_or_reactions_path=molecules_or_reactions_path,
                                       output_path=policy_data_file,
                                       dataset_type=config['PolicyNetwork']['policy_type'],
                                       batch_size=config['PolicyNetwork']['batch_size'],
                                       num_cpus=config['General']['num_cpus'])

    # train policy network
    print('\nTRAIN POLICY NETWORK ...')
    policy_config = PolicyNetworkConfig.from_dict(config['PolicyNetwork'])
    run_policy_training(datamodule, config=policy_config, results_path=policy_output_folder)
    config['PolicyNetwork']['weights_path'] = os.path.join(policy_output_folder, 'weights', 'policy_network.ckpt')

    # self-tuning value network training
    print('\nTRAIN VALUE NETWORK ...')
    value_output_folder = os.path.join(config['General']['results_root'], 'value_network')
    Path(value_output_folder).mkdir(parents=True, exist_ok=True)
    config['ValueNetwork']['weights_path'] = os.path.join(value_output_folder, 'weights', 'value_network.ckpt')

    run_self_tuning(config, results_root=value_output_folder)


if __name__ == '__main__':
    main()
