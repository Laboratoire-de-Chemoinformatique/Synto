import logging
import yaml
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
from CGRtools.containers import ReactionContainer, MoleculeContainer, CGRContainer
from CGRtools.files import RDFRead, RDFWrite
from StructureFingerprint import MorganFingerprint
from tqdm import tqdm

from Synto.chem.utils import remove_small_molecules, rebalance_reaction, remove_reagents


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
    """Checks if there are compete reactions"""

    def __init__(self, fingerprint_tanimoto_threshold: float = 0.3, mcs_tanimoto_threshold: float = 0.6):
        self.fingerprint_tanimoto_threshold = fingerprint_tanimoto_threshold
        self.mcs_tanimoto_threshold = mcs_tanimoto_threshold

    def __call__(self, reaction: ReactionContainer) -> bool:
        """
        Returns True if the reaction has competing products, else False

        :param reaction: input reaction
        :return: True or False
        """
        mf = MorganFingerprint()
        is_compete = False

        # Check for compete products using both fingerprint similarity and maximum common substructure (MCS) similarity
        for mol in reaction.reagents:
            for other_mol in reaction.products:
                if len(mol) > 6 and len(other_mol) > 6:
                    # Compute fingerprint similarity
                    molf = mf.transform([mol])
                    other_molf = mf.transform([other_mol])
                    fingerprint_tanimoto = tanimoto_kernel(molf, other_molf)[0][0]

                    # If fingerprint similarity is high enough, check for MCS similarity
                    if fingerprint_tanimoto > self.fingerprint_tanimoto_threshold:
                        try:
                            # Find the maximum common substructure (MCS) and compute its size
                            clique_size = len(next(mol.get_mcs_mapping(other_mol, limit=100)))

                            # Calculate MCS similarity based on MCS size
                            mcs_tanimoto = clique_size / (len(mol) + len(other_mol) - clique_size)

                            # If MCS similarity is also high enough, mark the reaction as having compete products
                            if mcs_tanimoto > self.mcs_tanimoto_threshold:
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

    def __call__(self, reaction: ReactionContainer) -> bool:
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
                    if any(cgr.atom(neighbour_id).atomic_symbol not in ('C', 'H') for neighbour_id in
                           cgr._bonds[c_with_h_id]) or any(
                        cgr.atom(neighbour_id).atomic_symbol not in ('C', 'H') for neighbour_id in
                        cgr._bonds[another_c_id]):
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
    """Checks if a reaction involves ring C-C bond breaking"""

    def __call__(self, reaction: ReactionContainer) -> bool:
        """
        Returns True if the reaction involves ring C-C bond breaking, else False

        :param reaction: input reaction
        :return: True or False
        """
        cgr = ~reaction

        # Extract reactants' center atoms and their rings
        reactants_center_atoms = {}
        reactants_rings = set()
        for reactant in reaction.reactants:
            reactants_rings.update(reactant.sssr)
            for n, atom in reactant.atoms():
                if n in cgr.center_atoms:
                    reactants_center_atoms[n] = atom

        # Identify reaction center based on center atoms
        reaction_center = cgr.augmented_substructure(centers=cgr.center_atoms, deep=0)

        # Iterate over bonds in the reaction center and check for ring C-C bond breaking
        for atom_id, neighbour_id, bond in reaction_center.bonds():
            try:
                # Retrieve corresponding atoms from reactants
                atom = reactants_center_atoms[atom_id]
                neighbour = reactants_center_atoms[neighbour_id]
            except KeyError:
                continue
            else:
                # Check if the bond is broken and both atoms are carbons in rings of size 5, 6, or 7
                is_bond_broken = (bond.order is not None) and (bond.p_order is None)
                are_atoms_carbons = atom.atomic_symbol == 'C' and neighbour.atomic_symbol == 'C'
                are_atoms_in_ring = (atom.ring_sizes.intersection({5, 6, 7}) and neighbour.ring_sizes.intersection(
                    {5, 6, 7}) and any(atom_id in ring and neighbour_id in ring for ring in reactants_rings))

                # If all conditions are met, indicate ring C-C bond breaking
                if is_bond_broken and are_atoms_carbons and are_atoms_in_ring:
                    return True

        return False


class ReactionCheckConfig:
    """
    A class to hold and manage configuration settings for various reaction checkers.

    The configuration can specify which checkers to use and their parameters.

    Attributes:
        config (dict): A dictionary where keys are checker names and values are their parameters.
    """

    _default_config = {
        'reaction_database_file_name': 'path/to/reaction_database.rdf',
        'result_directory_name': './',
        'filtered_reactions_file_name': 'filtered_reactions.rdf',
        'remove_old_results': True,
        'min_popularity': 3,
        'remove_small_molecules': {
            'enabled': False,
            'number_of_atoms': 6,
            'small_molecules_to_meta': True
        },
        'remove_reagents': {
            'enabled': True,
            'keep_reagents': True,
            'reagents_max_size': 7
        },
        'rebalance_reaction': {
            'enabled': False
        },
        'checkers': {
            'CheckDynamicBondsNumber': {
                'min_bonds_number': 1,
                'max_bonds_number': 6
            },
            'CheckSmallMolecules': {
                'limit': 6
            },
            'CheckStrangeCarbons': {},
            'IsCompeteProducts': {
                'fingerprint_tanimoto_threshold': 0.3,
                'mcs_tanimoto_threshold': 0.6
            },
            'CheckCGRConnectedComponents': {},
            'CheckRings': {},
            'CheckNoReaction': {},
            'CheckMultiCenterReaction': {},
            'CheckWrongCHBreaking': {},
            'CheckCCsp3Breaking': {},
            'CheckCCRingBreaking': {}
        }
    }

    def __init__(self, config=None):
        """
        Initializes the ReactionCheckConfig with specified checker configurations.

        :param config: A dictionary with checker names as keys and their parameters as values.
        """
        if config is None:
            config = self._default_config
        self.config = config

    def to_yaml(self, file_path):
        """
        Serializes the configuration to a YAML file.

        :param file_path: The path to the file where the configuration will be saved.
        """
        with open(file_path, 'w') as file:
            yaml.dump(self.config, file, default_flow_style=False)

    @classmethod
    def from_yaml(cls, file_path):
        """
        Deserializes a YAML file into a ReactionCheckConfig object.

        :param file_path: The path to the YAML file to be loaded.
        :return: An instance of ReactionCheckConfig.
        """
        with open(file_path, 'r') as file:
            checkers_config = yaml.load(file, Loader=yaml.FullLoader)
            return cls(checkers_config)

    def create_checkers(self):
        """
        Creates instances of enabled checker classes based on the stored configuration.

        :return: A list of enabled checker instances.
        """
        checker_instances = []

        if 'CheckDynamicBondsNumber' in self.config['checkers']:
            params = self.config['checkers']['CheckDynamicBondsNumber']
            checker_instances.append(CheckDynamicBondsNumber(**params))

        if 'CheckSmallMolecules' in self.config['checkers']:
            params = self.config['checkers']['CheckSmallMolecules']
            checker_instances.append(CheckSmallMolecules(**params))

        if 'CheckStrangeCarbons' in self.config['checkers']:
            checker_instances.append(CheckStrangeCarbons())

        if 'IsCompeteProducts' in self.config['checkers']:
            params = self.config['checkers']['IsCompeteProducts']
            checker_instances.append(IsCompeteProducts(**params))

        if 'CheckCGRConnectedComponents' in self.config['checkers']:
            checker_instances.append(CheckCGRConnectedComponents())

        if 'CheckRings' in self.config['checkers']:
            checker_instances.append(CheckRings())

        if 'CheckNoReaction' in  self.config['checkers']:
            checker_instances.append(CheckNoReaction())

        if 'CheckMultiCenterReaction' in self.config['checkers']:
            checker_instances.append(CheckMultiCenterReaction())

        if 'CheckWrongCHBreaking' in self.config['checkers']:
            checker_instances.append(CheckWrongCHBreaking())

        if 'CheckCCsp3Breaking' in self.config['checkers']:
            checker_instances.append(CheckCCsp3Breaking())

        if 'CheckCCRingBreaking' in self.config['checkers']:
            checker_instances.append(CheckCCRingBreaking())

        return checker_instances

    # Example usage
    """
    Example usage:

    # Creating a configuration object
    config = ReactionCheckConfig(dynamic_bonds_min=2, dynamic_bonds_max=5, small_molecules_limit=5)

    # Saving to YAML
    config.to_yaml('config.yml')

    # Loading from YAML
    loaded_config = ReactionCheckConfig.from_yaml('config.yml')

    # Creating checker instances
    checkers = loaded_config.create_checkers()

    # Assuming you have a 'ReactionContainer' object named 'my_reaction'
    for checker in checkers:
        result = checker(my_reaction)
        print(f"{checker.__class__.__name__}: {result}")
    """


def remove_files_if_exists(directory: Path, file_names):
    for file_name in file_names:
        file_path = directory / file_name
        if file_path.is_file():
            file_path.unlink()
            logging.warning(f"Removed {file_path}")


def filter_reactions(config: ReactionCheckConfig) -> None:
    """
    Processes a database of chemical reactions, applying checks based on the provided configuration,
    and writes the results to specified files. All configurations are provided by the ReactionCheckConfig object.

    :param config: ReactionCheckConfig object containing all configuration settings.
    :return: None. The function writes the processed reactions to specified RDF and pickle files.
             Unique reactions are written if save_only_unique is True.
    """
    result_directory = Path(config.config['result_directory_name'])
    result_directory.mkdir(parents=True, exist_ok=True)

    if config.config["remove_old_results"]:
        remove_files_if_exists(
            result_directory,
            [
                config.config['filtered_reactions_file_name'],
                config.config['result_reactions_file_name'],
            ]
        )

    checkers = config.create_checkers()

    with RDFRead(config.config['reaction_database_file_name'], indexable=True) as reactions, \
         RDFWrite(str(result_directory / config.config['filtered_reactions_file_name']), append=True) as filtered_file, \
         RDFWrite(str(result_directory / config.config['result_reactions_file_name']), append=True) as result_file:

        for reaction_index, reaction in tqdm(enumerate(reactions), total=len(reactions)):

            if config.config['remove_small_molecules']['enabled']:
                reaction = remove_small_molecules(reaction,
                                                  number_of_atoms=config.config['remove_small_molecules'][
                                                      'number_of_atoms'],
                                                  small_molecules_to_meta=config.config['remove_small_molecules'][
                                                      'small_molecules_to_meta'])

            if config.config['remove_reagents']['enabled']:
                reaction = remove_reagents(reaction,
                                           keep_reagents=config.config['remove_reagents']['keep_reagents'],
                                           reagents_max_size=config.config['remove_reagents']['reagents_max_size'])

            if config.config['rebalance_reaction']['enabled']:
                reaction = rebalance_reaction(reaction)

            if any(checker(reaction) for checker in checkers):
                filtered_file.write(reaction)
                continue
            else:
                reaction.meta['reaction_index'] = reaction_index
                result_file.write(reaction)
