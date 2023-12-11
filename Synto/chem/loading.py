"""
Module containing functions for loading reaction rules and building blocks
"""

import logging
import pickle
from time import time
from tqdm import tqdm

import pandas as pd

from CGRtools import SMILESRead
from CGRtools.reactor import Reactor
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename


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


def standardize_building_blocks(input_file, out_file):
    """
    The function canonicalizes custom building blocks

    :param out_file:
    :param input_file: The path to the txt file that stores the original building blocks
    :param input_file: The path to the txt file that stores the canonicalazied building blocks
    """

    parser = SMILESRead.create_parser(ignore=True)

    with open(input_file, "r") as file:
        mols = file.readlines()

    for n, mol in tqdm(enumerate(mols), total=len(mols)):
        try:
            mol = parser(str(mol))
            mol.canonicalize()
        except:
            continue
    mols = [i.strip() for i in mols]

    pd.DataFrame(mols).to_csv(out_file, header=None, index=None)

    return out_file

def load_building_blocks(file: str, canonicalize: bool = False):
    """
    Loads building blocks data from a file, either in text, SMILES, or pickle format, and returns a frozen set of
    building blocks.

    :param file: The path to the file containing the building blocks data
    :type file: str
    :param canonicalize: The `canonicalize` parameter determines whether the loaded building blocks should be
    canonicalized or not
    :type canonicalize: bool (optional)
    :return: The frozen set loaded building blocks
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
            bb = set([mol.decode("utf-8") for mol in file])
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
                    bb = set([str(mol).strip() for mol in mols])
                else:
                    bb = set([str(mol).strip() for mol in file])
        elif filetype == "pickle" or filetype == "pkl":
            with open(file, "rb") as file:
                bb = pickle.load(file)
                if isinstance(bb, list):
                    bb = set(bb)
        else:
            raise TypeError(
                f"expected .txt, .smi, .pickle, .pkl or .db files, not {filetype}"
            )

    stop = time()
    logging.debug(f"{len(bb)} In-Stock Substances are loaded.\nTook {round(stop - start, 2)} seconds.")
    return bb
