import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Tuple, Dict, Any, Optional

import numpy as np
import ray
import yaml
from CGRtools.containers import ReactionContainer, MoleculeContainer, CGRContainer
from CGRtools.files import RDFRead, RDFWrite
from StructureFingerprint import MorganFingerprint
from tqdm.auto import tqdm

from Syntool.chem.utils import (
    remove_small_molecules,
    rebalance_reaction,
    remove_reagents,
    to_reaction_smiles_record,
)
from Syntool.utils.config import ConfigABC, convert_config_to_dict


@dataclass
class CompeteProductsConfig(ConfigABC):
    fingerprint_tanimoto_threshold: float = 0.3
    mcs_tanimoto_threshold: float = 0.6

    @staticmethod
    def from_dict(config_dict: Dict[str, Any]):
        """Create an instance of CompeteProductsConfig from a dictionary."""
        return CompeteProductsConfig(**config_dict)

    @staticmethod
    def from_yaml(file_path: str):
        """Deserialize a YAML file into a CompeteProductsConfig object."""
        with open(file_path, "r") as file:
            config_dict = yaml.safe_load(file)
        return CompeteProductsConfig.from_dict(config_dict)

    def _validate_params(self, params: Dict[str, Any]):
        """Validate configuration parameters."""
        if not isinstance(params.get("fingerprint_tanimoto_threshold"), float) or not (
            0 <= params["fingerprint_tanimoto_threshold"] <= 1
        ):
            raise ValueError(
                "Invalid 'fingerprint_tanimoto_threshold'; expected a float between 0 and 1"
            )

        if not isinstance(params.get("mcs_tanimoto_threshold"), float) or not (
            0 <= params["mcs_tanimoto_threshold"] <= 1
        ):
            raise ValueError(
                "Invalid 'mcs_tanimoto_threshold'; expected a float between 0 and 1"
            )


class CompeteProductsChecker:
    """Checks if there are compete reactions."""

    def __init__(
        self,
        fingerprint_tanimoto_threshold: float = 0.3,
        mcs_tanimoto_threshold: float = 0.6,
    ):
        self.fingerprint_tanimoto_threshold = fingerprint_tanimoto_threshold
        self.mcs_tanimoto_threshold = mcs_tanimoto_threshold

    @staticmethod
    def from_config(config: CompeteProductsConfig):
        """Creates an instance of CompeteProductsChecker from a configuration object."""
        return CompeteProductsChecker(
            config.fingerprint_tanimoto_threshold, config.mcs_tanimoto_threshold
        )

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
                            clique_size = len(
                                next(mol.get_mcs_mapping(other_mol, limit=100))
                            )

                            # Calculate MCS similarity based on MCS size
                            mcs_tanimoto = clique_size / (
                                len(mol) + len(other_mol) - clique_size
                            )

                            # If MCS similarity is also high enough, mark the reaction as having compete products
                            if mcs_tanimoto > self.mcs_tanimoto_threshold:
                                is_compete = True
                                break
                        except StopIteration:
                            continue

        return is_compete


@dataclass
class DynamicBondsConfig(ConfigABC):
    min_bonds_number: int = 1
    max_bonds_number: int = 6

    @staticmethod
    def from_dict(config_dict: Dict[str, Any]):
        """Create an instance of DynamicBondsConfig from a dictionary."""
        return DynamicBondsConfig(**config_dict)

    @staticmethod
    def from_yaml(file_path: str):
        """Deserialize a YAML file into a DynamicBondsConfig object."""
        with open(file_path, "r") as file:
            config_dict = yaml.safe_load(file)
        return DynamicBondsConfig.from_dict(config_dict)

    def _validate_params(self, params: Dict[str, Any]):
        """Validate configuration parameters."""
        if (
            not isinstance(params.get("min_bonds_number"), int)
            or params["min_bonds_number"] < 0
        ):
            raise ValueError(
                "Invalid 'min_bonds_number'; expected a non-negative integer"
            )

        if (
            not isinstance(params.get("max_bonds_number"), int)
            or params["max_bonds_number"] < 0
        ):
            raise ValueError(
                "Invalid 'max_bonds_number'; expected a non-negative integer"
            )

        if params["min_bonds_number"] > params["max_bonds_number"]:
            raise ValueError(
                "'min_bonds_number' cannot be greater than 'max_bonds_number'"
            )


class DynamicBondsChecker:
    """Checks if there is an unacceptable number of dynamic bonds in CGR."""

    def __init__(self, min_bonds_number: int = 1, max_bonds_number: int = 6):
        self.min_bonds_number = min_bonds_number
        self.max_bonds_number = max_bonds_number

    @staticmethod
    def from_config(config: DynamicBondsConfig):
        """Creates an instance of DynamicBondsChecker from a configuration object."""
        return DynamicBondsChecker(config.min_bonds_number, config.max_bonds_number)

    def __call__(self, reaction: ReactionContainer) -> bool:
        cgr = ~reaction
        return not (
            self.min_bonds_number <= len(cgr.center_bonds) <= self.max_bonds_number
        )


@dataclass
class SmallMoleculesConfig(ConfigABC):
    limit: int = 6

    @staticmethod
    def from_dict(config_dict: Dict[str, Any]):
        """Create an instance of SmallMoleculesConfig from a dictionary."""
        return SmallMoleculesConfig(**config_dict)

    @staticmethod
    def from_yaml(file_path: str):
        """Deserialize a YAML file into a SmallMoleculesConfig object."""
        with open(file_path, "r") as file:
            config_dict = yaml.safe_load(file)
        return SmallMoleculesConfig.from_dict(config_dict)

    def _validate_params(self, params: Dict[str, Any]):
        """Validate configuration parameters."""
        if not isinstance(params.get("limit"), int) or params["limit"] < 1:
            raise ValueError("Invalid 'limit'; expected a positive integer")


class SmallMoleculesChecker:
    """Checks if there are only small molecules in the reaction or if there is only one small reactant or product."""

    def __init__(self, limit: int = 6):
        self.limit = limit

    @staticmethod
    def from_config(config: SmallMoleculesConfig):
        """Creates an instance of SmallMoleculesChecker from a configuration object."""
        return SmallMoleculesChecker(config.limit)

    def __call__(self, reaction: ReactionContainer) -> bool:
        if len(reaction.reactants) == 1 and self.are_only_small_molecules(
            reaction.reactants
        ):
            return True
        elif len(reaction.products) == 1 and self.are_only_small_molecules(
            reaction.products
        ):
            return True
        elif self.are_only_small_molecules(
            reaction.reactants
        ) and self.are_only_small_molecules(reaction.products):
            return True
        return False

    def are_only_small_molecules(self, molecules: Iterable[MoleculeContainer]) -> bool:
        """Checks if all molecules in the given iterable are small molecules."""
        return all(len(molecule) <= self.limit for molecule in molecules)


@dataclass
class CGRConnectedComponentsConfig:
    pass


class CGRConnectedComponentsChecker:
    """Allows to check if CGR contains unrelated components (without reagents)."""

    @staticmethod
    def from_config(config: CGRConnectedComponentsConfig):
        """Creates an instance of CGRConnectedComponentsChecker from a configuration object."""
        return CGRConnectedComponentsChecker()

    def __call__(self, reaction: ReactionContainer) -> bool:
        tmp_reaction = ReactionContainer(reaction.reactants, reaction.products)
        cgr = ~tmp_reaction
        return cgr.connected_components_count > 1


@dataclass
class RingsChangeConfig:
    pass


class RingsChangeChecker:
    """Allows to check if there is changing rings number in the reaction."""

    @staticmethod
    def from_config(config: RingsChangeConfig):
        """Creates an instance of RingsChecker from a configuration object."""
        return RingsChangeChecker()

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


@dataclass
class StrangeCarbonsConfig:
    # Currently empty, but can be extended in the future if needed
    pass


class StrangeCarbonsChecker:
    """Checks if there are 'strange' carbons in the reaction."""

    @staticmethod
    def from_config(config: StrangeCarbonsConfig):
        """Creates an instance of StrangeCarbonsChecker from a configuration object."""
        return StrangeCarbonsChecker()

    def __call__(self, reaction: ReactionContainer) -> bool:
        for molecule in reaction.reactants + reaction.products:
            atoms_types = {
                a.atomic_symbol for _, a in molecule.atoms()
            }  # atoms types in molecule
            if len(atoms_types) == 1 and atoms_types.pop() == "C":
                if len(molecule) == 1:  # methane
                    return True
                bond_types = {int(b) for _, _, b in molecule.bonds()}
                if len(bond_types) == 1 and bond_types.pop() != 4:
                    return True  # C molecules with only one type of bond (not aromatic)
        return False


@dataclass
class NoReactionConfig:
    # Currently empty, but can be extended in the future if needed
    pass


class NoReactionChecker:
    """Checks if there is no reaction in the provided reaction container."""

    @staticmethod
    def from_config(config: NoReactionConfig):
        """Creates an instance of NoReactionChecker from a configuration object."""
        return NoReactionChecker()

    def __call__(self, reaction: ReactionContainer) -> bool:
        cgr = ~reaction
        return not cgr.center_atoms and not cgr.center_bonds


@dataclass
class MultiCenterConfig:
    pass


class MultiCenterChecker:
    """Checks if there is a multicenter reaction."""

    @staticmethod
    def from_config(config: MultiCenterConfig):
        return MultiCenterChecker()

    def __call__(self, reaction: ReactionContainer) -> bool:
        cgr = ~reaction
        return len(cgr.centers_list) > 1


@dataclass
class WrongCHBreakingConfig:
    pass


class WrongCHBreakingChecker:
    """Checks for incorrect C-C bond formation from breaking a C-H bond."""

    @staticmethod
    def from_config(config: WrongCHBreakingConfig):
        return WrongCHBreakingChecker()

    def __call__(self, reaction: ReactionContainer) -> bool:
        """
        Determines if a reaction involves incorrect C-C bond formation from breaking a C-H bond.

        :param reaction: The reaction to be checked.
        :return: True if incorrect C-C bond formation is found, False otherwise.
        """
        reaction.kekule()
        if reaction.check_valence():
            return False
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
            if cgr.atom(atom_id).atomic_symbol == "C":
                is_c_h_breaking, is_c_c_formation = False, False
                c_with_h_id, another_c_id = None, None

                for neighbour_id, bond in cgr._bonds[atom_id].items():
                    neighbour = cgr.atom(neighbour_id)

                    if (
                        bond.order
                        and not bond.p_order
                        and neighbour.atomic_symbol == "H"
                    ):
                        is_c_h_breaking = True
                        c_with_h_id = atom_id

                    elif (
                        not bond.order
                        and bond.p_order
                        and neighbour.atomic_symbol == "C"
                    ):
                        is_c_c_formation = True
                        another_c_id = neighbour_id

                if is_c_h_breaking and is_c_c_formation:
                    # Check for presence of heteroatoms in the first environment of 2 bonding carbons
                    if any(
                        cgr.atom(neighbour_id).atomic_symbol not in ("C", "H")
                        for neighbour_id in cgr._bonds[c_with_h_id]
                    ) or any(
                        cgr.atom(neighbour_id).atomic_symbol not in ("C", "H")
                        for neighbour_id in cgr._bonds[another_c_id]
                    ):
                        return False
                    return True

        return False


@dataclass
class CCsp3BreakingConfig:
    pass


class CCsp3BreakingChecker:
    """Checks if there is C(sp3)-C bond breaking."""

    @staticmethod
    def from_config(config: CCsp3BreakingConfig):
        return CCsp3BreakingChecker()

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
            are_atoms_carbons = (
                atom.atomic_symbol == "C" and neighbour.atomic_symbol == "C"
            )
            is_atom_sp3 = atom.hybridization == 1 or neighbour.hybridization == 1

            if is_bond_broken and are_atoms_carbons and is_atom_sp3:
                return True
        return False


@dataclass
class CCRingBreakingConfig:
    pass


class CCRingBreakingChecker:
    """Checks if a reaction involves ring C-C bond breaking."""

    @staticmethod
    def from_config(config: CCRingBreakingConfig):
        return CCRingBreakingChecker()

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
        reaction_center = cgr.augmented_substructure(atoms=cgr.center_atoms, deep=0)

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
                are_atoms_carbons = (
                    atom.atomic_symbol == "C" and neighbour.atomic_symbol == "C"
                )
                are_atoms_in_ring = (
                    set(atom.ring_sizes).intersection({5, 6, 7})
                    and set(neighbour.ring_sizes).intersection({5, 6, 7})
                    and any(
                        atom_id in ring and neighbour_id in ring
                        for ring in reactants_rings
                    )
                )

                # If all conditions are met, indicate ring C-C bond breaking
                if is_bond_broken and are_atoms_carbons and are_atoms_in_ring:
                    return True

        return False


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
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    x_dot = np.dot(x, y.T)
    x2 = np.sum(x**2, axis=1)
    y2 = np.sum(y**2, axis=1)

    denominator = np.array([x2] * len(y2)).T + np.array([y2] * len(x2)) - x_dot
    result = np.divide(
        x_dot, denominator, out=np.zeros_like(x_dot), where=denominator != 0
    )

    return result


@dataclass
class ReactionCheckConfig(ConfigABC):
    """
    Configuration class for reaction checks, inheriting from ConfigABC.

    This class manages configuration settings for various reaction checkers, including paths, file formats,
    and checker-specific parameters.

    Attributes:
        dynamic_bonds_config: Configuration for dynamic bonds checking.
        small_molecules_config: Configuration for small molecules checking.
        strange_carbons_config: Configuration for strange carbons checking.
        compete_products_config: Configuration for competing products checking.
        cgr_connected_components_config: Configuration for CGR connected components checking.
        rings_change_config: Configuration for rings change checking.
        no_reaction_config: Configuration for no reaction checking.
        multi_center_config: Configuration for multi-center checking.
        wrong_ch_breaking_config: Configuration for wrong C-H breaking checking.
        cc_sp3_breaking_config: Configuration for CC sp3 breaking checking.
        cc_ring_breaking_config: Configuration for CC ring breaking checking.
    """

    # Configuration for reaction checkers
    dynamic_bonds_config: Optional[DynamicBondsConfig] = None
    small_molecules_config: Optional[SmallMoleculesConfig] = None
    strange_carbons_config: Optional[StrangeCarbonsConfig] = None
    compete_products_config: Optional[CompeteProductsConfig] = None
    cgr_connected_components_config: Optional[CGRConnectedComponentsConfig] = None
    rings_change_config: Optional[RingsChangeConfig] = None
    no_reaction_config: Optional[NoReactionConfig] = None
    multi_center_config: Optional[MultiCenterConfig] = None
    wrong_ch_breaking_config: Optional[WrongCHBreakingConfig] = None
    cc_sp3_breaking_config: Optional[CCsp3BreakingConfig] = None
    cc_ring_breaking_config: Optional[CCRingBreakingConfig] = None

    # Other configuration parameters
    rebalance_reaction: bool = False
    remove_reagents: bool = True
    reagents_max_size: int = 7
    remove_small_molecules: bool = False
    small_molecules_max_size: int = 6

    def to_dict(self):
        """
        Converts the configuration into a dictionary.
        """
        config_dict = {
            "dynamic_bonds_config": convert_config_to_dict(
                self.dynamic_bonds_config, DynamicBondsConfig
            ),
            "small_molecules_config": convert_config_to_dict(
                self.small_molecules_config, SmallMoleculesConfig
            ),
            "compete_products_config": convert_config_to_dict(
                self.compete_products_config, CompeteProductsConfig
            ),
            "cgr_connected_components_config": {}
            if self.cgr_connected_components_config is not None
            else None,
            "rings_change_config": {} if self.rings_change_config is not None else None,
            "strange_carbons_config": {}
            if self.strange_carbons_config is not None
            else None,
            "no_reaction_config": {} if self.no_reaction_config is not None else None,
            "multi_center_config": {} if self.multi_center_config is not None else None,
            "wrong_ch_breaking_config": {}
            if self.wrong_ch_breaking_config is not None
            else None,
            "cc_sp3_breaking_config": {}
            if self.cc_sp3_breaking_config is not None
            else None,
            "cc_ring_breaking_config": {}
            if self.cc_ring_breaking_config is not None
            else None,
            "rebalance_reaction": self.rebalance_reaction,
            "remove_reagents": self.remove_reagents,
            "reagents_max_size": self.reagents_max_size,
            "remove_small_molecules": self.remove_small_molecules,
            "small_molecules_max_size": self.small_molecules_max_size,
        }

        filtered_config_dict = {k: v for k, v in config_dict.items() if v is not None}

        return filtered_config_dict

    @staticmethod
    def from_dict(config_dict: Dict[str, Any]):
        """
        Create an instance of ReactionCheckConfig from a dictionary.
        """
        # Instantiate configuration objects if their corresponding dictionary is present
        dynamic_bonds_config = (
            DynamicBondsConfig(**config_dict["dynamic_bonds_config"])
            if "dynamic_bonds_config" in config_dict
            else None
        )
        small_molecules_config = (
            SmallMoleculesConfig(**config_dict["small_molecules_config"])
            if "small_molecules_config" in config_dict
            else None
        )
        compete_products_config = (
            CompeteProductsConfig(**config_dict["compete_products_config"])
            if "compete_products_config" in config_dict
            else None
        )
        cgr_connected_components_config = (
            CGRConnectedComponentsConfig()
            if "cgr_connected_components_config" in config_dict
            else None
        )
        rings_change_config = (
            RingsChangeConfig()
            if "rings_change_config" in config_dict
            else None
        )
        strange_carbons_config = (
            StrangeCarbonsConfig()
            if "strange_carbons_config" in config_dict
            else None
        )
        no_reaction_config = (
            NoReactionConfig()
            if "no_reaction_config" in config_dict
            else None
        )
        multi_center_config = (
            MultiCenterConfig()
            if "multi_center_config" in config_dict
            else None
        )
        wrong_ch_breaking_config = (
            WrongCHBreakingConfig()
            if "wrong_ch_breaking_config" in config_dict
            else None
        )
        cc_sp3_breaking_config = (
            CCsp3BreakingConfig()
            if "cc_sp3_breaking_config" in config_dict
            else None
        )
        cc_ring_breaking_config = (
            CCRingBreakingConfig()
            if "cc_ring_breaking_config" in config_dict
            else None
        )

        # Extract other simple configuration parameters
        rebalance_reaction = config_dict.get("rebalance_reaction", False)
        remove_reagents = config_dict.get("remove_reagents", True)
        reagents_max_size = config_dict.get("reagents_max_size", 7)
        remove_small_molecules = config_dict.get("remove_small_molecules", False)
        small_molecules_max_size = config_dict.get("small_molecules_max_size", 6)

        return ReactionCheckConfig(
            dynamic_bonds_config=dynamic_bonds_config,
            small_molecules_config=small_molecules_config,
            compete_products_config=compete_products_config,
            cgr_connected_components_config=cgr_connected_components_config,
            rings_change_config=rings_change_config,
            strange_carbons_config=strange_carbons_config,
            no_reaction_config=no_reaction_config,
            multi_center_config=multi_center_config,
            wrong_ch_breaking_config=wrong_ch_breaking_config,
            cc_sp3_breaking_config=cc_sp3_breaking_config,
            cc_ring_breaking_config=cc_ring_breaking_config,
            rebalance_reaction=rebalance_reaction,
            remove_reagents=remove_reagents,
            reagents_max_size=reagents_max_size,
            remove_small_molecules=remove_small_molecules,
            small_molecules_max_size=small_molecules_max_size,
        )

    @staticmethod
    def from_yaml(file_path):
        """
        Deserializes a YAML file into a ReactionCheckConfig object.
        """
        with open(file_path, "r") as file:
            config_dict = yaml.safe_load(file)
        return ReactionCheckConfig.from_dict(config_dict)

    def _validate_params(self, params: Dict[str, Any]):
        if not isinstance(params["rebalance_reaction"], bool):
            raise ValueError("rebalance_reaction must be a boolean.")

        if not isinstance(params["remove_reagents"], bool):
            raise ValueError("remove_reagents must be a boolean.")

        if not isinstance(params["reagents_max_size"], int):
            raise ValueError("reagents_max_size must be an int.")

        if not isinstance(params["remove_small_molecules"], bool):
            raise ValueError("remove_small_molecules must be a boolean.")

        if not isinstance(params["small_molecules_max_size"], int):
            raise ValueError("small_molecules_max_size must be an int.")

    def create_checkers(self):
        checker_instances = []

        if self.dynamic_bonds_config is not None:
            checker_instances.append(
                DynamicBondsChecker.from_config(self.dynamic_bonds_config)
            )

        if self.small_molecules_config is not None:
            checker_instances.append(
                SmallMoleculesChecker.from_config(self.small_molecules_config)
            )

        if self.strange_carbons_config is not None:
            checker_instances.append(
                StrangeCarbonsChecker.from_config(self.strange_carbons_config)
            )

        if self.compete_products_config is not None:
            checker_instances.append(
                CompeteProductsChecker.from_config(self.compete_products_config)
            )

        if self.cgr_connected_components_config is not None:
            checker_instances.append(
                CGRConnectedComponentsChecker.from_config(
                    self.cgr_connected_components_config
                )
            )

        if self.rings_change_config is not None:
            checker_instances.append(
                RingsChangeChecker.from_config(self.rings_change_config)
            )

        if self.no_reaction_config is not None:
            checker_instances.append(
                NoReactionChecker.from_config(self.no_reaction_config)
            )

        if self.multi_center_config is not None:
            checker_instances.append(
                MultiCenterChecker.from_config(self.multi_center_config)
            )

        if self.wrong_ch_breaking_config is not None:
            checker_instances.append(
                WrongCHBreakingChecker.from_config(self.wrong_ch_breaking_config)
            )

        if self.cc_sp3_breaking_config is not None:
            checker_instances.append(
                CCsp3BreakingChecker.from_config(self.cc_sp3_breaking_config)
            )

        if self.cc_ring_breaking_config is not None:
            checker_instances.append(
                CCRingBreakingChecker.from_config(self.cc_ring_breaking_config)
            )

        return checker_instances


def remove_files_if_exists(directory: Path, file_names):
    for file_name in file_names:
        file_path = directory / file_name
        if file_path.is_file():
            file_path.unlink()
            logging.warning(f"Removed {file_path}")


def filter_reaction(
    reaction: ReactionContainer,
    config: ReactionCheckConfig,
    checkers: list,
    output_files_format: str = "rdf",
):
    is_filtered = False
    if config.remove_small_molecules:
        new_reaction = remove_small_molecules(
            reaction,
            number_of_atoms=config.small_molecules_max_size,
        )
    else:
        new_reaction = reaction.copy()

    if new_reaction is None:
        is_filtered = True

    if config.remove_reagents and not is_filtered:
        new_reaction = remove_reagents(
            new_reaction,
            keep_reagents=True,
            reagents_max_size=config.reagents_max_size,
        )

    if new_reaction is None:
        is_filtered = True
        new_reaction = reaction.copy()

    if not is_filtered:
        if config.rebalance_reaction:
            new_reaction = rebalance_reaction(new_reaction)
        for checker in checkers:
            if checker(new_reaction):
                new_reaction.meta["filtration_log"] = checker.__class__.__name__
                is_filtered = True
                break

    if output_files_format == "smiles":
        new_reaction = to_reaction_smiles_record(new_reaction)

    return is_filtered, new_reaction


@ray.remote
def process_batch(batch, config: ReactionCheckConfig, checkers, output_files_format):
    results = []
    for index, reaction in batch:
        is_filtered, processed_reaction = filter_reaction(
            reaction, config, checkers, output_files_format
        )
        results.append((index, is_filtered, processed_reaction))
    return results


def process_completed_batches(futures, filtered_file, result_file, pbar, batch_size):
    done, _ = ray.wait(list(futures.keys()), num_returns=1)
    completed_batch = ray.get(done[0])

    # Write results of the completed batch to file
    for index, is_filtered, reaction in completed_batch:
        if is_filtered:
            filtered_file.write(reaction)
        else:
            result_file.write(reaction)

    # Remove completed future and update progress bar
    del futures[done[0]]
    pbar.update(batch_size)


def filter_reactions(
    config: ReactionCheckConfig,
    reaction_database_path: str,
    result_directory_path: str = "./",
    result_reactions_file_name: str = "clean_reactions",
    filtered_reactions_file_name: str = "removed_reactions",
    output_files_format: str = "rdf",
    append_results: bool = False,
    num_cpus: int = 1,
    batch_size: int = 10,
) -> None:
    """
    Processes a database of chemical reactions, applying checks based on the provided configuration,
    and writes the results to specified files. All configurations are provided by the ReactionCheckConfig object.

    :param config: ReactionCheckConfig object containing all configuration settings.
    :param reaction_database_path: Path to the reaction database file.
    :param result_directory_path: Name of the directory to store results.
    :param output_files_format: Format of the output files (e.g., 'rdf').
    :param result_reactions_file_name: Name for the file containing cleaned reactions.
    :param filtered_reactions_file_name: Name for the file containing filtered reactions.
    :param append_results: Flag indicating whether to append results to existing files.
    :param num_cpus: Number of CPUs to use for processing.
    :param batch_size: Size of the batch for processing reactions.
    :return: None. The function writes the processed reactions to specified RDF and pickle files.
             Unique reactions are written if save_only_unique is True.
    """
    result_directory = Path(result_directory_path)
    result_directory.mkdir(parents=True, exist_ok=True)

    checkers = config.create_checkers()

    ray.init(num_cpus=num_cpus, ignore_reinit_error=True)

    max_concurrent_batches = num_cpus  # Limit the number of concurrent batches

    if output_files_format == "smiles":
        open_mode = "a" if append_results else "w"
        result_file = open(
            str(result_directory / f"{result_reactions_file_name}.smiles"),
            open_mode,
        )
        filtered_file = open(
            str(result_directory / f"{filtered_reactions_file_name}.smiles"),
            open_mode,
        )
    elif output_files_format == "rdf":
        result_file = RDFWrite(
            str(result_directory / f"{result_reactions_file_name}.rdf"),
            append=append_results,
        )
        filtered_file = RDFWrite(
            str(result_directory / f"{filtered_reactions_file_name}.rdf"),
            append=append_results,
        )
    else:
        raise ValueError(
            f"I don't know this output files format: {output_files_format}"
        )

    with RDFRead(reaction_database_path, indexable=True) as reactions_file:
        total_reactions = len(reactions_file)
        pbar = tqdm(total=total_reactions)

        futures = {}
        batch = []

        for index, reaction in enumerate(reactions_file):
            reaction.meta["reaction_index"] = index
            batch.append((index, reaction))
            if len(batch) == batch_size:
                future = process_batch.remote(
                    batch, config, checkers, output_files_format
                )
                futures[future] = None
                batch = []

                # Check and process completed tasks if we've reached the concurrency limit
                while len(futures) >= max_concurrent_batches:
                    process_completed_batches(
                        futures,
                        filtered_file,
                        result_file,
                        pbar,
                        batch_size,
                    )

        # Process the last batch if it's not empty
        if batch:
            future = process_batch.remote(batch, config, checkers, output_files_format)
            futures[future] = None

        # Process remaining batches
        while futures:
            process_completed_batches(
                futures, filtered_file, result_file, pbar, batch_size
            )

        pbar.close()
    result_file.close()
    filtered_file.close()
    ray.shutdown()

    # Example usage
    """
    Example usage:
    # Importing config and reaction filtering function
    from Syntool.chem.filtering import ReactionCheckConfig, filter_reactions

    # Creating a configuration object with default parameters
    config = ReactionCheckConfig()

    # Changing default parameters
    config.config["reaction_database_path"] = ./uspto.rdf
    config.config["result_directory_name"] = results/
    config.config['num_cpus'] = 8
    config.config['batch_size'] = 20

    # Launching filtering of reactions
    filter_reactions(config)
    """
