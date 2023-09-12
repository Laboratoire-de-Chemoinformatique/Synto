"""
Module containing a class Retron that represents a retron (extend molecule object) in the search tree
"""

from CGRtools.containers import MoleculeContainer

from Synto.training.preprocessing import safe_canonicalization


class Retron:
    """
    Retron class is used to extend the molecule behavior needed for interaction with a tree in MCTS
    """

    def __init__(self, molecule: MoleculeContainer):
        """
        It initializes a Retron object with a molecule container as a parameter.

        :param molecule: The `molecule` parameter is of type `MoleculeContainer`.
        :type molecule: MoleculeContainer
        """
        self._molecule = safe_canonicalization(molecule)  # TODO this is from visualisation path_graph - fix it
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

    def __eq__(self, other: "Retron"):
        """
        The function checks if the current Retron is equal to another Retron of the same class.

        :param other: The "other" parameter is a reference to another object of the same class "Retron". It is used to
        compare the current Retron with the other Retron to check if they are equal
        :type other: "Retron"
        """
        return self._molecule == other._molecule

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

    # def is_building_block(self, building_blocks):
    #     if len(self._molecule) <= 6:
    #         return True
    #     tmp = self._molecule.copy()
    #     try:
    #         tmp.canonicalize()
    #         return str(tmp) in building_blocks
    #
    #     except InvalidAromaticRing:
    #         return str(self._molecule) in building_blocks

    def is_building_block(self, stock):
        """
        The function checks if a Retron is a building block.

        :param stock: The list of building blocks. Each building block is represented by a smiles.
        """
        if len(self._molecule) <= 6:
            return True
        else:
            return str(self._molecule) in stock
