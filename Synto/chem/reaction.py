"""
Module containing classes and functions for manipulating reactions and reaction rules
"""

from CGRtools.containers import MoleculeContainer
from CGRtools.containers import ReactionContainer
from CGRtools.reactor import Reactor


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
