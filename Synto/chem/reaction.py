"""
Module containing classes and functions for manipulating reactions and reaction rules
"""

from Synto.chem.standardizer import Standardizer

from CGRtools import Reactor
from CGRtools.containers import MoleculeContainer
from CGRtools.containers import ReactionContainer
from CGRtools.files import RDFRead, RDFWrite
from multiprocessing import Queue, Process, Manager
from tqdm import tqdm
from logging import warning, getLogger
import os


class Reaction(ReactionContainer):
    """
    Reaction class can be used for a general representation of reaction for different chemoinformatics Python packages
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the reaction object.
        """
        super().__init__(*args, **kwargs)


def add_small_mols(big_mol, small_molecules=None):
    """
    The function takes a molecule and returns a list of modified molecules where each small molecule has been added to
    the big molecule.

    :param big_mol: A molecule
    :param small_molecules: A list of small molecules that need to be added to the molecule
    :return: Returns a list of molecules.
    """
    if small_molecules:
        tmp_mol = big_mol.copy()
        transition_mapping = {}
        for small_mol in small_molecules:

            for n, atom in small_mol.atoms():
                new_number = tmp_mol.add_atom(atom.atomic_symbol)
                transition_mapping[n] = new_number

            for atom, neighbor, bond in small_mol.bonds():
                tmp_mol.add_bond(transition_mapping[atom], transition_mapping[neighbor], bond)

            transition_mapping = {}
        return tmp_mol.split()
    else:
        return [big_mol]


def apply_reaction_rule(molecule: MoleculeContainer, reaction_rule: Reactor):
    """
    The function applies a reaction rule to a given molecule.

    :param molecule: A MoleculeContainer object representing the molecule on which the reaction rule will be applied
    :type molecule: MoleculeContainer
    :param reaction_rule: The reaction_rule is an instance of the Reactor class. It represents a reaction rule that
    can be applied to a molecule
    :type reaction_rule: Reactor
    """

    try:
        reactants = add_small_mols(molecule, small_molecules=False)

        unsorted_reactions = list(reaction_rule(reactants))
        sorted_reactions = sorted(unsorted_reactions,
                                  key=lambda reaction: len(list(filter(lambda x: len(x) > 6, reaction.products))),
                                  reverse=True)

        reactions = sorted_reactions[:3]  # Take top-N reactions from reactor
    except IndexError:
        reactions = []

    for reaction in reactions:
        yield reaction


def cleaner(reaction: ReactionContainer, logger):
    """
    Standardize a reaction according to external script

    :param reaction: ReactionContainer to clean/standardize
    :param logger: Logger - to avoid writing log
    :return: ReactionContainer or empty list
    """
    standardizer = Standardizer(skip_errors=True, keep_unbalanced_ions=False, id_tag='Reaction_ID', keep_reagents=False,
                                ignore_mapping=True, action_on_isotopes=2, skip_tautomerize=True, logger=logger)
    return standardizer.standardize(reaction)


def worker_cleaner(to_clean: Queue, to_write: Queue):
    """
    Launches standardizations using the Queue to_clean. Fills the to_write Queue with results

    :param to_clean: Queue of reactions to clean/standardize
    :param to_write: Standardized outputs to write
    """
    logger = getLogger()
    logger.disabled = True
    while True:
        raw_reaction = to_clean.get()
        if raw_reaction == "Quit":
            break
        res = cleaner(raw_reaction, logger)
        if res:
            to_write.put(res)
    logger.disabled = False


def cleaner_writer(output_file: str, to_write: Queue, remove_old=True):
    """
    Writes in output file the standardized reactions

    :param output_file: output file path
    :param to_write: Standardized ReactionContainer to write
    :param remove_old: whenever to remove or not an already existing file
    """

    if remove_old and os.path.isfile(output_file):
        os.remove(output_file)
        warning(f"Removed {output_file}")

    with RDFWrite(output_file) as out:
        while True:
            res = to_write.get()
            if res == "Quit":
                break
            out.write(res)


def reactions_cleaner(input_file: str, output_file: str, num_cpus: int, batch_prep_size: int = 100):
    """
    Writes in output file the standardized reactions

    :param input_file: input RDF file path
    :param output_file: output RDF file path
    :param num_cpus: number of CPU to be parallelized
    :param batch_prep_size: size of each batch per CPU
    """
    with Manager() as m:
        to_clean = m.Queue(maxsize=num_cpus * batch_prep_size)
        to_write = m.Queue(maxsize=batch_prep_size)

        writer = Process(target=cleaner_writer, args=(output_file, to_write,))
        writer.start()

        workers = []
        for _ in range(num_cpus - 2):
            w = Process(target=worker_cleaner, args=(to_clean, to_write))
            w.start()
            workers.append(w)

        with RDFRead(input_file, indexable=True) as reactions:
            reactions.reset_index()
            for n, raw_reaction in tqdm(enumerate(reactions), total=len(reactions)):
                to_clean.put(raw_reaction)

        for _ in workers:
            to_clean.put("Quit")
        for w in workers:
            w.join()

        to_write.put("Quit")
        writer.join()
