"""
Module containing commands line scripts for training and planning mode
"""

import os
import shutil
import warnings
from pathlib import Path

import click
import gdown

from Synto.chem.data.cleaning import reactions_cleaner
from Synto.chem.data.filtering import filter_reactions, ReactionCheckConfig
from Synto.chem.loading import load_reaction_rules
from Synto.chem.loading import standardize_building_blocks
from Synto.chem.reaction_rules.extraction import extract_rules_from_reactions
from Synto.mcts.search import tree_search
from Synto.ml.training.reinforcement import run_self_tuning
from Synto.ml.training.supervised import create_policy_dataset, run_policy_training
from Synto.utils.config import read_training_config

from Synto.ml.networks.policy import PolicyNetworkConfig
from Synto.mcts.tree import TreeConfig

warnings.filterwarnings("ignore")


@click.group(name="syntool")
def syntool():
    pass


@syntool.command()
def download_planning_data():
    """
    Downloads a file from Google Drive using its remote ID, saves it as a zip file, extracts the contents,
    and then deletes the zip file
    """
    remote_id = "1c5YJDT-rP1ZvFA-ELmPNTUj0b8an4yFf"
    output = "synto_planning_data.zip"
    #
    gdown.download(output=output, id=remote_id, quiet=True)
    shutil.unpack_archive(output, "./")
    #
    os.remove(output)


@syntool.command()
def download_training_data():
    """
    Downloads a file from Google Drive using its remote ID, saves it as a zip file, extracts the contents,
    and then deletes the zip file
    """
    remote_id = "1r4I7OskGvzg-zxYNJ7WVYpVR2HSYW10N"
    output = "synto_training_data.zip"
    #
    gdown.download(output=output, id=remote_id, quiet=True)
    shutil.unpack_archive(output, "./")
    #
    os.remove(output)


@syntool.command()
@click.option(
    "-c",
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to the configuration file. This file contains settings for filtering reactions.",
)
@click.option(
    "--reaction_data",
    required=True,
    type=click.Path(exists=True),
    help="Path to the reaction database file that will be processed.",
)
@click.option(
    "--results_dir",
    default=Path("./"),
    type=click.Path(exists=True),
    help="Directory where the results will be stored. Defaults to the current directory.",
)
@click.option(
    "--clean_file_name",
    default="clean_reactions",
    type=str,
    help="Filename for the file containing processed reactions. Defaults to 'clean_reactions'.",
)
@click.option(
    "--removed_file_name",
    default="removed_reactions",
    type=str,
    help="Filename for the file containing reactions that were filtered out. Defaults to 'removed_reactions'.",
)
@click.option(
    "--files_format",
    default="rdf",
    type=str,
    help="Format of the output files. Supported formats include 'rdf'. Defaults to 'rdf'.",
)
@click.option(
    "--append_results",
    is_flag=True,
    default=False,
    help="If set, results will be appended to existing files. By default, new files are created.",
)
@click.option(
    "--num_cpus",
    default=1,
    type=int,
    help="Number of CPUs to use for processing. Defaults to 1.",
)
@click.option(
    "--batch_size",
    default=10,
    type=int,
    help="Size of the batch for processing reactions. Defaults to 10.",
)
def filtering_cli(
    config_path,
    reaction_data,
    results_dir,
    clean_file_name,
    removed_file_name,
    files_format,
    append_results,
    num_cpus,
    batch_size,
):
    """
    Processes a database of chemical reactions, applying checks based on the provided configuration,
    and writes the results to specified files. All configurations are provided by the ReactionCheckConfig object.
    """
    reaction_check_config = ReactionCheckConfig().from_yaml(config_path)
    filter_reactions(
        reaction_check_config,
        reaction_data,
        results_dir,
        clean_file_name,
        removed_file_name,
        files_format,
        append_results,
        num_cpus,
        batch_size,
    )


@syntool.command(name="rules_extraction")
def rules_extraction_cli():
    pass


@syntool.command(name="policy_training")
def policy_training_cli():
    pass


@syntool.command(name="planning")
@click.option(
    "--tree_config",
    required=True,
    type=click.Path(exists=True),
    help="Path to the YAML file containing the tree search configuration."
)
@click.option(
    "--reaction_rules",
    required=True,
    type=click.Path(exists=True),
    help="Path to the file containing reaction rules."
)
@click.option(
    "--building_blocks",
    required=True,
    type=click.Path(exists=True),
    help="Path to the file containing building blocks."
)
@click.option(
    "--policy_weights",
    required=True,
    type=click.Path(exists=True),
    help="Path to the file containing policy weights."
)
@click.option(
    "--targets",
    required=True,
    type=click.Path(exists=True),
    help="Path to the file containing target molecules (in SDF or SMILES format)."
)
@click.option(
    "--value_weights",
    type=click.Path(exists=True),
    help="Path to the file containing value weights. Optional."
)
@click.option(
    "--results_root",
    default="search_results/",
    type=click.Path(),
    help="Directory to store the results of the tree search. Defaults to 'search_results/'."
)
@click.option(
    "--stats_name",
    default="tree_search_stats.csv",
    type=str,
    help="Name of the file to save statistics of the tree search. Defaults to 'tree_search_stats.csv'."
)
@click.option(
    "--retropaths_files_name",
    default="retropath",
    type=str,
    help="Base name for the files storing retro paths. Defaults to 'retropath'."
)
@click.option(
    "--logging_file_name",
    default="tree_search.log",
    type=str,
    help="Name of the log file for the tree search process. Defaults to 'tree_search.log'."
)
@click.option(
    "--log_level",
    default=10,
    type=int,
    help="Logging level for recording messages. Defaults to 10."
)
def planning_cli(
    tree_config,
    reaction_rules,
    building_blocks,
    policy_weigths,
    targets,
    value_weights,
    results_root,
    stats_name,
    retropaths_files_name,
    logging_file_name,
    log_level,
):
    """
    Executes a tree search for synthetic pathways of target molecules using specified configuration, reaction rules,
    and building blocks.
    """
    tree_search(
        tree_config,
        reaction_rules,
        building_blocks,
        policy_weigths,
        targets,
        value_weights,
        results_root,
        stats_name,
        retropaths_files_name,
        logging_file_name,
        log_level,
    )


@syntool.command(name="magic_button")
@click.option(
    "--config",
    "config_path",
    required=True,
    help="Path to the config YAML file.",
    type=click.Path(exists=True, path_type=Path),
)

def synto_training_cli(config_path):

    # read training config
    print("READ CONFIG ...")
    config = read_training_config(config_path)
    print("Config is read")

    # reaction mapping
    pass

    # reaction data cleaning
    if config["DataCleaning"]["clean_reactions"]:
        print("\nCLEAN REACTION DATA ...")

        output_folder = os.path.join(config["General"]["results_root"], "reaction_data")
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        cleaned_data_file = os.path.join(output_folder, "reaction_data_cleaned.rdf")

        reactions_cleaner(
            input_file=config["InputData"]["reaction_data_path"],
            output_file=cleaned_data_file,
            num_cpus=config["General"]["num_cpus"],
        )

    # standardize building blocks
    print("\nSTANDARDIZE BUILDING BLOCKS ...")
    if config["DataCleaning"]["standardize_building_blocks"]:
        standardize_building_blocks(
            config["InputData"]["building_blocks_path"],
            config["InputData"]["building_blocks_path"],
        )

    # reaction rules extraction
    print("\nEXTRACT REACTION RULES ...")
    if config["DataCleaning"]["clean_reactions"]:
        reaction_file = cleaned_data_file
    else:
        reaction_file = config["InputData"]["reaction_data_path"]

    reaction_rules_folder = os.path.join(
        config["General"]["results_root"], "reaction_rules"
    )
    # TODO remove redundant variables - instead write to the config['General']['results_root'] = path
    reaction_rules_folder = os.path.join(config['General']['results_root'], 'reaction_rules')
    Path(reaction_rules_folder).mkdir(parents=True, exist_ok=True)

    extract_rules_from_reactions(
        reaction_file=reaction_file,
        results_root=reaction_rules_folder,
        min_popularity=config["ReactionRules"]["min_popularity"],
        num_cpus=config["General"]["num_cpus"],
    )
    extract_rules_from_reactions(reaction_file=reaction_file,
                                 results_root=reaction_rules_folder,
                                 num_cpus=config['General']['num_cpus'])

    # create policy network dataset
    print("\nCREATE POLICY NETWORK DATASET ...")

    reaction_rules_path = os.path.join(
        reaction_rules_folder, "reaction_rules_filtered.pickle"
    )
    config["InputData"]["reaction_rules_path"] = reaction_rules_path

    policy_output_folder = os.path.join(
        config["General"]["results_root"], "policy_network"
    )
    Path(policy_output_folder).mkdir(parents=True, exist_ok=True)
    policy_data_file = os.path.join(policy_output_folder, "policy_dataset.pt")

    if config['PolicyNetwork']['policy_type'] == 'ranking':
        molecules_or_reactions_path = reaction_file
    else:

        molecules_or_reactions_path = config['InputData']['policy_data_path']

    datamodule = create_policy_dataset(reaction_rules_path=reaction_rules_path,
                                       molecules_or_reactions_path=molecules_or_reactions_path,
                                       output_path=policy_data_file,
                                       dataset_type=config['PolicyNetwork']['policy_type'],
                                       batch_size=config['PolicyNetwork']['batch_size'],
                                       num_cpus=config['General']['num_cpus'])

    # train policy network
    print("\nTRAIN POLICY NETWORK ...")
    n_rules = len(load_reaction_rules(reaction_rules_path))
    run_policy_training(
        datamodule, config, n_rules=n_rules, results_path=policy_output_folder
    )
    config["PolicyNetwork"]["weights_path"] = os.path.join(
        policy_output_folder, "policy_network.ckpt"
    )
    print('\nTRAIN POLICY NETWORK ...')

    policy_config = PolicyNetworkConfig.from_dict(config['PolicyNetwork'])

    run_policy_training(datamodule, config=policy_config, results_path=policy_output_folder)
    config['PolicyNetwork']['weights_path'] = os.path.join(policy_output_folder, 'weights', 'policy_network.ckpt')

    # self-tuning value network training
    print("\nTRAIN VALUE NETWORK ...")
    value_output_folder = os.path.join(
        config["General"]["results_root"], "value_network"
    )
    Path(value_output_folder).mkdir(parents=True, exist_ok=True)

    config["ValueNetwork"]["weights_path"] = os.path.join(
        value_output_folder, "value_network.ckpt"
    )
    config['ValueNetwork']['weights_path'] = os.path.join(value_output_folder, 'weights', 'value_network.ckpt')
    run_self_tuning(config, results_root=value_output_folder)
