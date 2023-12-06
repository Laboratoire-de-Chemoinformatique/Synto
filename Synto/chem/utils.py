from typing import List

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


def unite_molecules(molecules: List[MoleculeContainer]) -> MoleculeContainer:
    """
    Unites a list of MoleculeContainer objects into a single MoleculeContainer.

    This function takes multiple molecules and combines them into one larger molecule.
    The first molecule in the list is taken as the base, and subsequent molecules are united with it sequentially.

    :param molecules: A list of MoleculeContainer objects to be united.
    :return: A single MoleculeContainer object representing the union of all input molecules.
    """
    new_mol = molecules[0]
    for mol in molecules[1:]:
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


