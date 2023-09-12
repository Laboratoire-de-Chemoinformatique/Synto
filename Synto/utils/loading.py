"""
Module containing functions for loading reaction rules and building blocks
"""

import pickle
import logging
import sqlite3
from time import time
from CGRtools import SMILESRead
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename
from CGRtools.reactor import Reactor


def load_reaction_rules(file):
    """
    The function loads reaction rules from a pickle file and converts them into a list of Reactor objects if necessary

    :param file: The path to the pickle file that stores the reaction rules
    :return: A list of reaction rules
    """
    with open(file, "rb") as f:
        reaction_rules = pickle.load(f)

    if not isinstance(reaction_rules[0], Reactor):
        reaction_rules = [Reactor(x) for x in reaction_rules]

    return reaction_rules

# def load_reaction_rules(file):
#     with open(file, "rb") as f:
#         reaction_rules = pickle.load(f)
#
#     if isinstance(reaction_rules, dict):  # TODO dict > list - fix format
#         from GRRtools.transformations import ReverseReaction
#
#         reaction_rules = list(reaction_rules.keys())
#
#         reverse_reaction = ReverseReaction()
#         reaction_rules = [reverse_reaction(i) for i in reaction_rules]
#
#     if not isinstance(reaction_rules[0], Reactor):
#         reaction_rules = [Reactor(x) for x in reaction_rules]
#
#     return reaction_rules


# def load_reaction_rules(file):
#     with open(file, "rb") as f:
#         reaction_rules = pickle.load(f)
#     reaction_rules = {k: v for k, v in reaction_rules.items() if len(v) >= 10}
#
#     if isinstance(reaction_rules, dict):  # TODO dict > list - fix format
#         from GRRtools.transformations import ReverseReaction
#
#         reaction_rules = list(reaction_rules.keys())
#
#         reverse_reaction = ReverseReaction()
#         reaction_rules = [reverse_reaction(i) for i in reaction_rules]
#
#     if not isinstance(reaction_rules[0], Reactor):
#         reaction_rules = [Reactor(x) for x in reaction_rules]
#
#     return reaction_rules


def load_building_blocks(file: str, canonicalize: bool = False):
    """
    Loads building blocks data from a file, either in text, SMILES, or pickle format, and returns a frozenset of
    building blocks.

    :param file: The path to the file containing the building blocks data
    :type file: str
    :param canonicalize: The `canonicalize` parameter determines whether the loaded building blocks should be
    canonicalized or not
    :type canonicalize: bool (optional)
    :return: Tthe a frozenset loaded building blocks
    """
    if not file:
        logging.warning("No external In-Stock data was loaded")
        return None

    start = time()
    if isinstance(file, FileStorage):
        filename = secure_filename(file.filename)
        if filename.endswith(".pickle") or filename.endswith(".pkl"):
            bb = pickle.load(file)
        elif filename.endswith(".txt") or filename.endswith(".smi"):
            bb = frozenset([mol.decode("utf-8") for mol in file])
        else:
            raise TypeError(
                "content of FileStorage is not appropriate for in-building_blocks dataloader, expected .txt, .smi, .pickle or .pkl"
            )
    elif isinstance(file, str):
        filetype = file.split(".")[-1]
        # Loading in-building_blocks substances data
        if filetype in {"txt", "smi", "smiles"}:
            with open(file, "r") as file:
                if canonicalize:
                    parser = SMILESRead.create_parser(ignore=True)
                    mols = [parser(str(mol)) for mol in file]
                    for mol in mols:
                        mol.canonicalize()
                    bb = frozenset([str(mol) for mol in mols])
                else:
                    bb = frozenset([str(mol) for mol in file])
        elif filetype == "pickle" or filetype == "pkl":
            with open(file, "rb") as file:
                bb = pickle.load(file)
                if isinstance(bb, list):
                    bb = frozenset(bb)
        else:
            raise TypeError(
                f"expected .txt, .smi, .pickle, .pkl or .db files, not {filetype}"
            )

    stop = time()
    logging.debug(f"{len(bb)} In-Stock Substances are loaded.\nTook {round(stop - start, 2)} seconds.")
    return bb
