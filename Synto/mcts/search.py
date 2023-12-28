"""
Module containing functions for running tree search for the set of target molecules
"""

import csv
import logging
from pathlib import Path

from CGRtools import smiles, MoleculeContainer
from tqdm import tqdm

from Synto.chem.utils import safe_canonicalization
from Synto.interfaces.visualisation import to_table
from Synto.mcts.tree import Tree, TreeConfig
from Synto.mcts.evaluation import ValueFunction
from Synto.mcts.expansion import PolicyConfig, PolicyFunction
from Synto.utils import path_type
from Synto.utils.files import MoleculeReader


def extract_tree_stats(tree, target):
    """
    Collects various statistics from a tree and returns them in a dictionary format

    :param tree: The retro tree.
    :param target: The target molecule or compound that you want to search for in the tree. It is
    expected to be a string representing the SMILES notation of the target molecule
    :return: A dictionary with the calculated statistics
    """
    newick_tree, newick_meta = tree.newickify(visits_threshold=0)
    newick_meta_line = ";".join(
        [f"{nid},{v[0]},{v[1]},{v[2]}" for nid, v in newick_meta.items()]
    )
    return {
        "target_smiles": str(target),
        "tree_size": len(tree),
        "search_time": round(tree.curr_time, 1),
        "found_paths": len(tree.winning_nodes),
        "newick_tree": newick_tree,
        "newick_meta": newick_meta_line,
    }


def tree_search(
    targets: path_type,
    tree_config: TreeConfig,
    reaction_rules_path: path_type,
    building_blocks_path: path_type,
    policy_weights_path: path_type,
    value_weights_paths: path_type = None,
    results_root: path_type = "search_results/",
    stats_name: str = "tree_search_stats.csv",
    retropaths_files_name: str = "retropath",
    logging_file_name: str = "tree_search.log",
    log_level: int = 10,
):
    """
    Performs a tree search on a set of target molecules using specified configuration and rules,
    logging the results and statistics.

    :param tree_config: The path to the YAML file containing the configuration for the tree search.
    :param reaction_rules_path: The path to the file containing reaction rules.
    :param building_blocks_path: The path to the file containing building blocks.
    :param policy_weights_path: The path to the file containing policy weights.
    :param targets: The path to the file containing the target molecules (in SDF or SMILES format).
    :param value_weights_paths: The path to the file containing value weights (optional).
    :param results_root: The path to the directory where the results of the tree search will be saved. Defaults to 'search_results/'.
    :param stats_name: The name of the file where the statistics of the tree search will be saved. Defaults to 'tree_search_stats.csv'.
    :param retropaths_files_name: The base name for the files that will be generated to store the retro paths. Defaults to 'retropath'.
    :param logging_file_name: The name of the log file for recording the tree search process. Defaults to 'tree_search.log'.
    :param log_level: The level of logging for recording messages. Defaults to 10.

    This function configures and executes a tree search algorithm, leveraging reaction rules and building blocks
    to find synthetic pathways for given target molecules. The results, including paths and statistics, are
    saved in the specified directory. Logging is used to record the process and any issues encountered.
    """

    policy_config = PolicyConfig(weights_path=policy_weights_path)
    policy_function = PolicyFunction(policy_config=policy_config)

    value_function = None
    if tree_config.evaluation_mode == 'gcn':
        value_function = ValueFunction(weights_path=value_weights_paths)

    # results folder
    results_root = Path(results_root)
    if not results_root.exists():
        results_root.mkdir()
        print(f"Created results directory at {results_root}")

    # logging molecules_path
    logging_file = results_root.joinpath(logging_file_name)
    logging.basicConfig(
        filename=logging_file,
        encoding="utf-8",
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%d/%m/%Y %I:%M:%S %p",
    )

    # targets molecules_path
    targets_file = Path(targets)
    assert targets_file.exists(), f"Target file at path {targets_file} does not exist"
    assert targets_file.suffix == ".smi", "Only SMI files are accepted"

    # stats molecules_path
    if stats_name:
        if ".csv" not in stats_name:
            stats_name += ".csv"
    else:
        stats_name = targets_file.stem + ".csv"

    stats_header = [
        "target_smiles",
        "tree_size",
        "search_time",
        "found_paths",
        "newick_tree",
        "newick_meta",
    ]

    stats_file = results_root.joinpath(stats_name)

    logging.info(f"Stats file will be saved at {stats_file}")

    # run search
    solved_trees = 0
    if retropaths_files_name is not None:
        retropaths_folder = results_root.joinpath("retropaths")
        retropaths_folder.mkdir(exist_ok=True)
    try:
        with MoleculeReader(targets_file) as inp, open(stats_file, "w", newline="\n") as csvfile:
            statswriter = csv.DictWriter(csvfile, delimiter=",", fieldnames=stats_header)
            statswriter.writeheader()

            targets_list = [m for m in inp.read()]

            for ti, target in tqdm(enumerate(targets_list), total=len(targets_list), position=0):
                target = safe_canonicalization(target)
                try:
                    tree = Tree(
                        target=target,
                        tree_config=tree_config,
                        reaction_rules_path=reaction_rules_path,
                        building_blocks_path=building_blocks_path,
                        policy_function=policy_function,
                        value_function=value_function,
                    )
                    for solved, _ in tree:
                        if solved:
                            solved_trees += 1
                            break

                    if retropaths_files_name is not None:
                        retropaths_file = retropaths_folder.joinpath(f"{retropaths_files_name}_target_{ti}.html")
                        to_table(tree, retropaths_file, extended=True)

                    statistics = extract_tree_stats(tree, target)
                    statswriter.writerow(statistics)
                    csvfile.flush()
                except AssertionError:
                    pass

        print(f"Solved number of target molecules: {solved_trees}")

    except KeyboardInterrupt:
        logging.info(f"So far solved: {solved_trees}")
