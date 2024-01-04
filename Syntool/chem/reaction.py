"""
Module containing classes and functions for manipulating reactions and reaction rules
"""

from CGRtools.reactor import Reactor
from CGRtools.containers import MoleculeContainer, ReactionContainer
from CGRtools.exceptions import InvalidAromaticRing


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


def apply_reaction_rule(
        molecule: MoleculeContainer,
        reaction_rule: Reactor,
        sort_reactions: bool = False,
        top_reactions_num: int = 3,
        validate_products: bool = True,
        rebuild_with_cgr: bool = False,
) -> list[MoleculeContainer]:
    """
    The function applies a reaction rule to a given molecule.

    :param rebuild_with_cgr:
    :param validate_products:
    :param sort_reactions:
    :param top_reactions_num:
    :param molecule: A MoleculeContainer object representing the molecule on which the reaction rule will be applied
    :type molecule: MoleculeContainer
    :param reaction_rule: The reaction_rule is an instance of the Reactor class. It represents a reaction rule that
    can be applied to a molecule
    :type reaction_rule: Reactor
    """

    reactants = add_small_mols(molecule, small_molecules=False)

    try:
        if sort_reactions:
            unsorted_reactions = list(reaction_rule(reactants))
            sorted_reactions = sorted(
                unsorted_reactions,
                key=lambda react: len(list(filter(lambda mol: len(mol) > 6, react.products))),
                reverse=True
            )
            reactions = sorted_reactions[:top_reactions_num]  # Take top-N reactions from reactor
        else:
            reactions = []
            for reaction in reaction_rule(reactants):
                reactions.append(reaction)
                if len(reactions) == top_reactions_num:
                    break
    except IndexError:
        reactions = []

    for reaction in reactions:
        if rebuild_with_cgr:
            cgr = reaction.compose()
            products = cgr.decompose()[1].split()
        else:
            products = reaction.products
        products = [mol for mol in products if len(mol) > 0]
        if validate_products:
            for molecule in products:
                try:
                    molecule.kekule()
                    if molecule.check_valence():
                        yield None
                    molecule.thiele()
                except InvalidAromaticRing:
                    yield None
        yield products
