from typing import Iterable, Tuple

import numpy as np
from CGRtools.containers import ReactionContainer, MoleculeContainer, CGRContainer
from StructureFingerprint import MorganFingerprint


def tanimoto_kernel(x, y):
    """
    Calculate the Tanimoto coefficient between each element of arrays x and y.

    Parameters
    ----------
    x : array-like
        A 2D array of features.
    y : array-like
        A 2D array of features.

    Notes
    -----
    Features in arrays x and y should be equal in number and ordered in the same way.

    Returns
    -------
    ndarray
        A 2D array containing pairwise Tanimoto coefficients.

    Examples
    --------
    >>> x = np.array([[1, 2], [3, 4]])
    >>> y = np.array([[5, 6], [7, 8]])
    >>> tanimoto_kernel(x, y)
    array([[...]])
    """
    x_dot = np.dot(x, y.T)
    x2 = np.sum(x ** 2, axis=1)
    y2 = np.sum(y ** 2, axis=1)

    denominator = (np.array([x2] * len(y2)).T + np.array([y2] * len(x2)) - x_dot)
    result = np.divide(x_dot, denominator, out=np.zeros_like(x_dot), where=denominator != 0)

    return result


class IsCompeteProducts:
    """Allows to check if there are compete reactions"""

    def __call__(self, reaction: ReactionContainer) -> bool:
        """
        Returns True if there are compete reactions, else False

        :param reaction: input reaction
        :return: True or False
        """
        mf = MorganFingerprint()
        is_compete = False
        for mol, other_mol in reaction.reagents + reaction.products:
            if len(mol) > 6 and len(other_mol) > 6:
                molf = mf.transform([mol])
                other_molf = mf.transform([other_mol])
                fingerprint_tanimoto = tanimoto_kernel(molf, other_molf)[0][0]
                if fingerprint_tanimoto > 0.3:
                    try:
                        clique_size = len(next(mol.get_mcs_mapping(other_mol, limit=100)))
                        mcs_tanimoto = clique_size / (len(mol) + len(other_mol) - clique_size)
                        if mcs_tanimoto > 0.6:
                            is_compete = True
                            break
                    except StopIteration:
                        continue
        return is_compete


class CheckCGRConnectedComponents:
    """Allows to check if CGR contains unrelated components (without reagents)"""

    def __call__(self, reaction: ReactionContainer) -> bool:
        """
        Returns True if CGR contains unrelated components (without reagents), else False

        :param reaction: input reaction
        :return: True or False
        """
        tmp_reaction = ReactionContainer(reaction.reactants, reaction.products)
        cgr = ~tmp_reaction
        if cgr.connected_components_count > 1:
            return True
        else:
            return False


class CheckRings:
    """Allows to check if there is changing rings number in the reaction"""

    def __call__(self, reaction: ReactionContainer):
        """
        Returns True if there are valence mistakes in the reaction or there is a reaction with mismatch numbers of all
        rings or aromatic rings in reactants and products (reaction in rings)

        :param reaction: input reaction
        :return: True or False
        """
        reaction.kekule()
        reaction.thiele()
        r_rings, r_arom_rings = self._calc_rings(reaction.reactants)
        p_rings, p_arom_rings = self._calc_rings(reaction.products)
        if r_arom_rings != p_arom_rings:
            return True
        elif r_rings != p_rings:
            return True
        else:
            return False


class CheckDynamicBondsNumber:
    """Allows to check if there is unacceptable number of dynamic bonds in CGR"""

    def __init__(self, min_bonds_number: int = 1, max_bonds_number: int = 6):
        """
        :param min_bonds_number: min acceptable number of dynamic bonds in CGR
        :param max_bonds_number: max acceptable number of dynamic bonds in CGR
        """
        self.min_bonds_number = min_bonds_number
        self.max_bonds_number = max_bonds_number

    def __call__(self, reaction: ReactionContainer) -> bool:
        """
        Returns True if there is unacceptable number of dynamic bonds in CGR, else False

        :param reaction: input reaction
        :return: True or False
        """
        cgr = ~reaction
        if self.min_bonds_number <= len(cgr.center_bonds) <= self.max_bonds_number:
            return False
        return True


class CheckSmallMolecules:
    """Allows to check if there are only small molecules in the reaction or there is only one small reactant or
    product"""

    def __init__(self, limit: int = 6):
        """
        :param limit: max number of heavy atoms in "small" molecules
        """
        self.limit = limit

    def __call__(self, reaction: ReactionContainer) -> ReactionContainer:
        """
        Returns True if there are only small molecules in the reaction or there is only one small reactant or product,
        else False

        :param reaction: input reaction
        :return: True or False
        """
        if len(reaction.reactants) == 1 and self.are_only_small_molecules(reaction.reactants):
            return True
        elif len(reaction.products) == 1 and self.are_only_small_molecules(reaction.products):
            return True
        elif self.are_only_small_molecules(reaction.reactants) and self.are_only_small_molecules(reaction.products):
            return True
        return False

    def are_only_small_molecules(self, molecules: Iterable[MoleculeContainer]) -> bool:
        """
        Returns True if there are only small molecules in input, else False

        :param molecules: set of molecules
        :return: True or False
        """
        only_small_mols = True
        for molecule in molecules:
            if len(molecule) > self.limit:
                only_small_mols = False
                break
        return only_small_mols


class CheckStrangeCarbons:
    """Allows to check if there are "strange" carbons in the reaction"""

    def __call__(self, reaction: ReactionContainer) -> bool:
        """
        Checks for the presence of methane or C molecules with only one type of bond (not aromatic) in the set of
        molecules

        :param reaction: input reaction
        :return: True or False
        """
        free_carbons = False
        for molecule in reaction.reactants + reaction.products:
            atoms_types = list(set(a.atomic_symbol for _, a in molecule.atoms()))  # atoms types in molecule
            if len(atoms_types) == 1:
                if atoms_types[0] == 'C':
                    if len(molecule) == 1:  # methane
                        free_carbons = True
                        break
                    else:
                        bond_types = list(set(int(b) for _, _, b in molecule.bonds()))
                        if len(bond_types) == 1:
                            if bond_types[0] != 4:
                                free_carbons = True  # C molecules with only one type of bond (not aromatic)
                                break
        return free_carbons


class CheckNoReaction:
    """Allows to check if there is no reaction"""

    def __call__(self, reaction: ReactionContainer) -> bool:
        """
        Returns True if there is no reaction, else False

        :param reaction: input reaction
        :return: True or False
        """
        cgr = ~reaction
        if not cgr.center_atoms and not cgr.center_bonds:
            return True
        return False


class CheckMultiCenterReaction:
    """Allows to check if there is multicenter reaction"""

    def __call__(self, reaction: ReactionContainer) -> bool:
        """
        Returns True if there is multicenter reaction, else False

        :param reaction: input reaction
        :return: True or False
        """
        cgr = ~reaction
        if len(cgr.centers_list) > 1:
            return True
        return False

    @staticmethod
    def _calc_rings(molecules: Iterable) -> Tuple[int, int]:
        """
        Calculates number of all rings and number of aromatic rings in molecules

        :param molecules: set of molecules
        :return: number of all rings and number of aromatic rings in molecules
        """
        rings, arom_rings = 0, 0
        for mol in molecules:
            rings += mol.rings_count
            arom_rings += len(mol.aromatic_rings)
        return rings, arom_rings


class CheckWrongCHBreaking:
    """
    Class to check if there is a C-C bond formation from breaking a C-H bond.
    This excludes condensation reactions and reactions with carbens.
    """

    def __call__(self, reaction: ReactionContainer) -> bool:
        """
        Determines if a reaction involves incorrect C-C bond formation from breaking a C-H bond.

        :param reaction: The reaction to be checked.
        :return: True if incorrect C-C bond formation is found, False otherwise.
        """
        reaction.kekule()
        reaction.thiele()
        copy_reaction = reaction.copy()
        copy_reaction.explicify_hydrogens()
        cgr = ~copy_reaction
        reduced_cgr = cgr.augmented_substructure(cgr.center_atoms, deep=1)

        return self.is_wrong_c_h_breaking(reduced_cgr)

    @staticmethod
    def is_wrong_c_h_breaking(cgr: CGRContainer) -> bool:
        """
        Checks for incorrect C-C bond formation from breaking a C-H bond in a CGR.
        :param cgr: The CGR with explicified hydrogens.
        :return: True if incorrect C-C bond formation is found, False otherwise.
        """
        for atom_id in cgr.center_atoms:
            if cgr.atom(atom_id).atomic_symbol == 'C':
                is_c_h_breaking, is_c_c_formation = False, False
                c_with_h_id, another_c_id = None, None

                for neighbour_id, bond in cgr._bonds[atom_id].items():
                    neighbour = cgr.atom(neighbour_id)

                    if bond.order and not bond.p_order and neighbour.atomic_symbol == 'H':
                        is_c_h_breaking = True
                        c_with_h_id = atom_id

                    elif not bond.order and bond.p_order and neighbour.atomic_symbol == 'C':
                        is_c_c_formation = True
                        another_c_id = neighbour_id

                if is_c_h_breaking and is_c_c_formation:
                    # Check for presence of heteroatoms in the first environment of 2 bonding carbons
                    if any(
                        cgr.atom(neighbour_id).atomic_symbol not in ('C', 'H')
                        for neighbour_id in cgr._bonds[c_with_h_id]
                    ) or any(
                        cgr.atom(neighbour_id).atomic_symbol not in ('C', 'H')
                        for neighbour_id in cgr._bonds[another_c_id]
                    ):
                        return False
                    return True

        return False


class CheckCCsp3Breaking:
    """Allows to check if there is C(sp3)-C bonds breaking"""

    def __call__(self, reaction: ReactionContainer) -> bool:
        """
        Returns True if there is C(sp3)-C bonds breaking, else False

        :param reaction: input reaction
        :return: True or False
        """
        cgr = ~reaction
        reaction_center = cgr.augmented_substructure(cgr.center_atoms, deep=1)
        for atom_id, neighbour_id, bond in reaction_center.bonds():
            atom = reaction_center.atom(atom_id)
            neighbour = reaction_center.atom(neighbour_id)

            is_bond_broken = bond.order is not None and bond.p_order is None
            are_atoms_carbons = atom.atomic_symbol == 'C' and neighbour.atomic_symbol == 'C'
            is_atom_sp3 = atom.hybridization == 1 or neighbour.hybridization == 1

            if is_bond_broken and are_atoms_carbons and is_atom_sp3:
                return True
        return False


class CheckCCRingBreaking:
    """Allows to check if there is ring C-C bonds breaking (5, 6, 7 atoms in rings)"""

    def __call__(self, reaction: ReactionContainer) -> bool:
        """
        Returns True if there is ring C-C bonds breaking (5, 6, 7 atoms in rings), else False

        :param reaction: input reaction
        :return: True or False
        """
        cgr = ~reaction

        reactants_center_atoms = {}
        reactants_rings = ()
        for reactant in reaction.reactants:
            reactants_rings += reactant.sssr
            for n, atom in reactant.atoms():
                if n in cgr.center_atoms:
                    reactants_center_atoms[n] = atom

        reaction_center = cgr.augmented_substructure(cgr.center_atoms, deep=0)
        for atom_id, neighbour_id, bond in reaction_center.bonds():
            try:
                atom = reactants_center_atoms[atom_id]
                neighbour = reactants_center_atoms[neighbour_id]
            except KeyError:
                continue
            else:
                is_bond_broken = bond.order is not None and bond.p_order is None
                are_atoms_carbons = atom.atomic_symbol == 'C' and neighbour.atomic_symbol == 'C'
                are_atoms_in_ring = bool({5, 6, 7}.intersection(atom.ring_sizes)) and bool(
                    {5, 6, 7}.intersection(neighbour.ring_sizes)) and any(
                    atom_id in ring and neighbour_id in ring for ring in reactants_rings)

            if is_bond_broken and are_atoms_carbons and are_atoms_in_ring:
                return True

        return False


class CheckRulesByPopularity:
    """Allows to check if there is a rare rule (was extracted from small number of reactions)"""

    def __init__(self, min_popularity: int = 3):
        """
        :param min_popularity: min acceptable number of reactions from which rule was extracted
        """
        self.min_popularity = min_popularity

    def __call__(self, rule: ReactionContainer) -> bool:
        """
        Returns True if there is a rare rule (was extracted from small number of reactions), else False

        :param rule: unique reaction rule with information in meta about the number of reactions from which it was
        extracted
        :return: True or False
        """
        # return True if int(rule.meta['Number_of_reactions']) < self.min_popularity else False
        return True if len(rule._Reactor__meta['reaction_ids']) < self.min_popularity else False
