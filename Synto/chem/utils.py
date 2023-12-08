from typing import List, Iterable, Tuple

from CGRtools.containers import MoleculeContainer, ReactionContainer, QueryContainer
from CGRtools.exceptions import InvalidAromaticRing


def query_to_mol(query: QueryContainer) -> MoleculeContainer:
    """
    Converts a QueryContainer object into a MoleculeContainer object.

    :param query: A QueryContainer object representing the query structure.
    :return: A MoleculeContainer object that replicates the structure of the query.
    """
    new_mol = MoleculeContainer()
    for n, atom in query.atoms():
        new_mol.add_atom(atom.atomic_symbol, n, charge=atom.charge, is_radical=atom.is_radical)
    for i, j, bond in query.bonds():
        new_mol.add_bond(i, j, int(bond))
    return new_mol


def reaction_query_to_reaction(rule: ReactionContainer) -> ReactionContainer:
    """
    Converts a ReactionContainer object with query structures into a ReactionContainer with molecular structures.

    :param rule: A ReactionContainer object where reactants and products are QueryContainer objects.
    :return: A new ReactionContainer
    :return: A new ReactionContainer object where reactants and products are MoleculeContainer objects.
    """
    reactants = [query_to_mol(q) for q in rule.reactants]
    products = [query_to_mol(q) for q in rule.products]
    reagents = [query_to_mol(q) for q in rule.reagents]  # Assuming reagents are also part of the rule
    reaction = ReactionContainer(reactants, products, reagents, rule.meta)
    reaction.name = rule.name
    return reaction


def unite_molecules(molecules: Iterable[MoleculeContainer]) -> MoleculeContainer:
    """
    Unites a list of MoleculeContainer objects into a single MoleculeContainer.

    This function takes multiple molecules and combines them into one larger molecule.
    The first molecule in the list is taken as the base, and subsequent molecules are united with it sequentially.

    :param molecules: A list of MoleculeContainer objects to be united.
    :return: A single MoleculeContainer object representing the union of all input molecules.
    """
    new_mol = MoleculeContainer()
    for mol in molecules:
        new_mol = new_mol.union(mol)
    return new_mol


def safe_canonicalization(molecule: MoleculeContainer):
    """
    Attempts to canonicalize a molecule, handling any exceptions.

    This function tries to canonicalize the given molecule.
    If the canonicalization process fails due to an InvalidAromaticRing exception,
    it safely returns the original molecule.

    :param molecule: The given molecule to be canonicalized.
    :return: The canonicalized molecule if successful, otherwise the original molecule.
    """
    molecule._atoms = dict(sorted(molecule._atoms.items()))

    tmp = molecule.copy()
    try:
        tmp.canonicalize()
        return tmp
    except InvalidAromaticRing:
        return molecule


def split_molecules(molecules: Iterable, number_of_atoms: int) -> Tuple[List, List]:
    """
    Splits molecules according to the number of heavy atoms.

    :param molecules: Iterable of molecules.
    :param number_of_atoms: Threshold for splitting molecules.
    :return: Tuple of lists containing "big" molecules and "small" molecules.
    """
    big_molecules, small_molecules = [], []
    for molecule in molecules:
        if len(molecule) > number_of_atoms:
            big_molecules.append(molecule)
        else:
            small_molecules.append(molecule)

    return big_molecules, small_molecules


def remove_small_molecules(
        reaction: ReactionContainer,
        number_of_atoms: int = 6,
        small_molecules_to_meta: bool = True
) -> ReactionContainer:
    """
    Processes a reaction by removing small molecules.

    :param reaction: ReactionContainer object.
    :param number_of_atoms: Molecules with the number of atoms equal to or below this will be removed.
    :param small_molecules_to_meta: If True, deleted molecules are saved to meta.
    :return: Processed ReactionContainer without small molecules.
    """
    new_reactants, small_reactants = split_molecules(reaction.reactants, number_of_atoms)
    new_products, small_products = split_molecules(reaction.products, number_of_atoms)

    new_reaction = ReactionContainer(new_reactants, new_products, reaction.reagents, reaction.meta)
    new_reaction.name = reaction.name

    if small_molecules_to_meta:
        united_small_reactants = unite_molecules(small_reactants)
        new_reaction.meta["small_reactants"] = str(united_small_reactants)

        united_small_products = unite_molecules(small_products)
        new_reaction.meta["small_products"] = str(united_small_products)

    return new_reaction


def rebalance_reaction(reaction: ReactionContainer) -> ReactionContainer:
    """
    Rebalances the reaction by assembling CGR and then decomposing it. Works for all reactions for which the correct
    CGR can be assembled

    :param reaction: a reaction object
    :return: a rebalanced reaction
    """
    tmp_reaction = ReactionContainer(reaction.reactants, reaction.products)
    cgr = ~tmp_reaction
    reactants, products = ~cgr
    rebalanced_reaction = ReactionContainer(reactants.split(), products.split(), reaction.reagents, reaction.meta)
    rebalanced_reaction.name = reaction.name
    return rebalanced_reaction


def reverse_reaction(reaction: ReactionContainer) -> ReactionContainer:
    """
    Reverses given reaction

    :param reaction: a reaction object
    :return: the reversed reaction
    """
    reversed_reaction = ReactionContainer(reaction.products, reaction.reactants, reaction.reagents, reaction.meta)
    reversed_reaction.name = reaction.name

    return reversed_reaction


def remove_reagents(
        reaction: ReactionContainer,
        keep_reagents: bool = True,
        reagents_max_size: int = 7
) -> ReactionContainer:
    """
    Removes reagents (not changed molecules or molecules not involved in the reaction) from reactants and products

    :param reaction: a reaction object
    :param keep_reagents: if True, the reagents are written to ReactionContainer
    :param reagents_max_size: max size of molecules that are called reagents, bigger are deleted
    :return: cleaned reaction
    """
    not_changed_molecules = set(reaction.reactants).intersection(reaction.products)

    cgr = ~reaction
    center_atoms = set(cgr.center_atoms)

    new_reactants = []
    new_products = []
    new_reagents = []

    for molecule in reaction.reactants:
        if center_atoms.isdisjoint(molecule) or molecule in not_changed_molecules:
            new_reagents.append(molecule)
        else:
            new_reactants.append(molecule)

    for molecule in reaction.products:
        if center_atoms.isdisjoint(molecule) or molecule in not_changed_molecules:
            new_reagents.append(molecule)
        else:
            new_products.append(molecule)

    if keep_reagents:
        new_reagents = {molecule for molecule in new_reagents if len(molecule) <= reagents_max_size}
    else:
        new_reagents = []

    new_reaction = ReactionContainer(new_reactants, new_products, new_reagents, reaction.meta)
    new_reaction.name = reaction.name

    return new_reaction
