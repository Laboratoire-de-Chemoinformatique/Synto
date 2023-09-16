"""
Module containing functions for running tree search for the set of target molecules
"""

import csv
import logging
from pathlib import Path

from tqdm import tqdm

from CGRtools import smiles
from CGRtools.files import SDFRead

from Synto.interfaces.visualisation import to_table
from Synto.mcts import Tree
from Synto.utils.config import read_planning_config



def collect_stats(tree, target):
    """
    Collects various statistics from a tree and returns them in a dictionary format

    :param tree: The retro tree.
    :param target: The target molecule or compound that you want to search for in the tree. It is
    expected to be a string representing the SMILES notation of the target molecule
    :return: A dictionary with the calculated statistics
    """
    newick_tree, newick_meta = tree.newickify(visits_threshold=0)
    newick_meta_line = ";".join([f"{nid},{v[0]},{v[1]},{v[2]}" for nid, v in newick_meta.items()])
    return {
        "target_smiles": str(target),
        "tree_size": len(tree),
        "search_time": round(tree.curr_time, 1),
        "found_paths": len(tree.winning_nodes),
        "newick_tree": newick_tree,
        "newick_meta": newick_meta_line,
    }


def tree_search(
        results_root,
        targets_file,
        config,
        stats_name: str = 'tree_search_stats.csv',
        retropaths_files_name: str = 'retropath',
        logging_file_name: str = 'tree_search.log',
        log_level: int = 10
):
    """
    The function performs a tree search on a set of target molecules stored in SDF file using a specified search
    configuration, logging the results and statistics.

    :param results_root: The path to the directory where the results of the tree search will be saved.
    :param targets_file: The path to the file containing the target molecules. It should be in SDF format.
    :param config: The path to a configuration file that contains the settings for the tree search algorithm.
    :param stats_name: The name of the file where the statistics of the tree search will be saved.
    :type stats_name: str (optional)
    :param retropaths_files_name: The name of the files that will be generated to store the retro paths.
    :type retropaths_files_name: str (optional)
    :param logging_file_name: The name of the log file that will be created during the execution of the tree search
    function. The log file will contain information about the progress and status of the tree searxh.
    :type logging_file_name: str (optional)
    :param log_level: The level of logging messages that will be recorded.
    :type log_level: int (optional)
    """

    # results folder
    results_root = Path(results_root)
    if not results_root.exists():
        results_root.mkdir()
        print(f"Created results directory at {results_root}")

    # logging molecules_path
    logging_file = results_root.joinpath(logging_file_name)
    logging.basicConfig(filename=logging_file, encoding='utf-8', level=log_level,
                        format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%d/%m/%Y %I:%M:%S %p')

    # targets molecules_path
    targets_file = Path(targets_file)
    assert targets_file.exists(), f"Target file at path {targets_file} does not exist"
    assert targets_file.suffix == ".txt", "Only txt files are accepted"

    # config molecules_path
    if config["Tree"]["init_new_node_value"] is None:
        logging.info(f"Evaluation strategy was chosen as extensive")
        if config["Tree"]["evaluation_agg"] == "max":
            logging.info(f"Update step will use max value of created children")
        elif config["Tree"]["evaluation_agg"] == "avg":
            logging.info(f"Update step will use average value of created children")
        else:
            raise ValueError(f"Parameter specified in Tree->evaluation_agg "
                             f"is unknown: {config['Tree']['evaluation_agg']}")
    else:
        logging.info(f"Evaluation strategy was chosen as greedy")

    # stats molecules_path
    if stats_name:
        if ".csv" not in stats_name:
            stats_name += ".csv"
    else:
        stats_name = targets_file.stem + ".csv"
    stats_header = ["target_smiles", "tree_size", "search_time", "found_paths", "newick_tree", "newick_meta"]
    stats_file = results_root.joinpath(stats_name)

    logging.info(f"Stats file will be saved at {stats_file}")

    # run search
    solved_trees = 0
    if retropaths_files_name is not None:
        retropaths_folder = results_root.joinpath('retropaths')
        retropaths_folder.mkdir(exist_ok=True)
    try:
        with open(targets_file) as inp, open(stats_file, "w", newline="\n") as csvfile:

            targets_list = [smiles(smi.strip()) for smi in inp.readlines()]
            targets_list = [m for m in targets_list if m]
            #
            statswriter = csv.DictWriter(csvfile, delimiter=",", fieldnames=stats_header)
            statswriter.writeheader()

            for ti, target in tqdm(enumerate(targets_list), total=len(targets_list), position=0):
                target.canonicalize()
                #
                tree = Tree(target=target, config=config)
                for solved, _ in tree:
                    if solved:
                        solved_trees += 1
                        break

                if retropaths_files_name is not None:
                    retropaths_file = retropaths_folder.joinpath(f"{retropaths_files_name}_target_{ti}.html")
                    to_table(tree, retropaths_file, extended=True)

                statistics = collect_stats(tree, target)
                statswriter.writerow(statistics)
                csvfile.flush()

        logging.info(f"Number of solved trees: {solved_trees}")

    except KeyboardInterrupt:
        logging.info(f"So far solved: {solved_trees}")