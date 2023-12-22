"""
Module containing a class Retron that represents a retron (extend molecule object) in the search tree
"""

from CGRtools.containers import MoleculeContainer
from CGRtools.exceptions import InvalidAromaticRing

from Synto.chem.utils import safe_canonicalization


class Retron:
    """
    Retron class is used to extend the molecule behavior needed for interaction with a tree in MCTS
    """

    def __init__(self, molecule: MoleculeContainer, canonicalize: bool = True):
        """
        It initializes a Retron object with a molecule container as a parameter.

        :param molecule: The `molecule` parameter is of type `MoleculeContainer`.
        :type molecule: MoleculeContainer
        """
        self._molecule = safe_canonicalization(molecule) if canonicalize else molecule
        self._mapping = None
        self.prev_retrons = []

    def __len__(self):
        """
        Return the number of atoms in Retron.
        """
        return len(self._molecule)

    def __hash__(self):
        """
        Returns the hash value of Retron.
        """
        return hash(self._molecule)

    def __str__(self):
        return str(self._molecule)

    def __eq__(self, other: "Retron"):
        """
        The function checks if the current Retron is equal to another Retron of the same class.

        :param other: The "other" parameter is a reference to another object of the same class "Retron". It is used to
        compare the current Retron with the other Retron to check if they are equal
        :type other: "Retron"
        """
        return self._molecule == other._molecule

    def validate_molecule(self):
        molecule = self._molecule.copy()
        try:
            molecule.kekule()
            if molecule.check_valence():
                return False
            molecule.thiele()
        except InvalidAromaticRing:
            return False
        return True

    @property
    def molecule(self) -> MoleculeContainer:
        """
        Returns a remapped MoleculeContainer object if self._mapping=True.
        """
        if self._mapping:
            remapped = self._molecule.copy()
            try:
                remapped = self._molecule.remap(self._mapping, copy=True)
            except ValueError:
                pass
            return remapped
        return self._molecule.copy()

    def __repr__(self):
        """
        Returns a SMILES of the retron
        """
        return str(self._molecule)

    def is_building_block(self, stock, min_mol_size=6):
        """
        The function checks if a Retron is a building block.

        :param min_mol_size:
        :param stock: The list of building blocks. Each building block is represented by a smiles.
        """
        if len(self._molecule) <= min_mol_size:
            return True
        else:
            return str(self._molecule) in stock


def compose_retrons(
    retrons: list = None, exclude_small=True, min_mol_size=6
) -> MoleculeContainer:
    """
    The function takes a list of retrons, excludes small retrons if specified, and composes them into a single molecule.
    This molecule is used for the prediction of synthesisability of the characterizing the possible success of the path
    including the nodes with the given retrons.

    :param retrons: The list of retrons to be composed.
    :type retrons: list
    :param exclude_small: The parameter that determines whether small retrons should be
    excluded from the composition process. If `exclude_small` is set to `True`, only retrons with a length greater than
    6 will be considered for composition.
    :return: A composed retrons as a MoleculeContainer object.
    """

    if len(retrons) == 1:
        return retrons[0].molecule
    elif len(retrons) > 1:
        if exclude_small:
            big_retrons = [
                retron for retron in retrons if len(retron.molecule) > min_mol_size
            ]
            if big_retrons:
                retrons = big_retrons
        tmp_mol = retrons[0].molecule.copy()
        transition_mapping = {}
        for mol in retrons[1:]:
            for n, atom in mol.molecule.atoms():
                new_number = tmp_mol.add_atom(atom.atomic_symbol)
                transition_mapping[n] = new_number
            for atom, neighbor, bond in mol.molecule.bonds():
                tmp_mol.add_bond(
                    transition_mapping[atom], transition_mapping[neighbor], bond
                )
            transition_mapping = {}
        return tmp_mol
