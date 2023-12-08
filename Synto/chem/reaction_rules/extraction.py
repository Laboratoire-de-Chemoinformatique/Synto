"""
Module containing functions with fixed protocol for reaction rules extraction
"""
from itertools import islice
from pathlib import Path
from typing import List, Union, Literal

import yaml
from CGRtools.containers import MoleculeContainer, QueryContainer, ReactionContainer
from CGRtools.reactor import Reactor
from CGRtools.files import RDFRead, RDFWrite
from tqdm import tqdm
from Synto.chem.utils import reaction_query_to_reaction


class ExtractRuleConfig:
    def __init__(
            self,
            reaction_database_file_name,
            result_directory_name,
            rules_file_name,
            multicenter_rules: bool = True,
            as_query_container: bool = True,
            reactor_validation: bool = False,
            include_func_groups: bool = False,
            func_groups_list: List[Union[MoleculeContainer, QueryContainer]] = None,
            include_rings: bool = False,
            keep_leaving_groups: bool = False,
            keep_incoming_groups: bool = False,
            keep_reagents: bool = True,
            environment_atom_count: int = 1,
            keep_metadata: bool = True,
            atom_info_retention: Literal["none", "reaction_center", "all"] = "reaction_center",
            info_to_clean: Union[frozenset[str], str] = frozenset(
                {"neighbors", "hybridization", "implicit_hydrogens", "ring_sizes"})
    ):
        """
        Initializes the configuration for extracting reaction rules.

        :param multicenter_rules: If True, extracts a single rule encompassing all centers.
        If False, extracts separate reaction rules for each reaction center in a multicenter reaction.
        :param as_query_container: If True, the extracted rules are generated as QueryContainer objects,
        analogous to SMARTS objects for pattern matching in chemical structures.
        :param reactor_validation: If True, validates each generated rule in a chemical reactor to ensure correct
                                    generation of products from reactants.
        :param include_func_groups: If True, includes specific functional groups in the reaction rule in addition
                                          to the reaction center and its environment.
        :param func_groups_list: A list of functional groups to be considered when include_functional_groups
                                          is True.
        :param include_rings: If True, includes ring structures in the reaction rules.
        :param keep_leaving_groups: If True, retains leaving groups in the extracted reaction rule.
        :param keep_incoming_groups: If True, retains incoming groups in the extracted reaction rule.
        :param keep_reagents: If True, includes reagents in the extracted reaction rule.
        :param environment_atom_count: Defines the size of the environment around the reaction center to be included
                                       in the rule (0 for only the reaction center, 1 for the first environment, etc.).
        :param keep_metadata: If True, retains metadata associated with the reaction in the extracted rule.
        :param atom_info_retention: Controls the amount of information about each atom to retain ('none',
                                    'reaction_center', or 'all').
        :param info_to_clean: Specifies the types of information to be removed from atoms when generating query
                              containers.

        The configuration settings provided in this method allow for a detailed and customized approach to the
        extraction and representation of chemical reaction rules.
        """
        self.reaction_database_file = Path(reaction_database_file_name)
        self.result_directory = Path(result_directory_name)
        self.result_directory.mkdir(parents=True, exist_ok=True)
        self.rules_file_name = rules_file_name

        self.multicenter_rules = multicenter_rules
        self.as_query_container = as_query_container
        self.reactor_validation = reactor_validation
        self.include_func_groups = include_func_groups
        self.func_groups_list = func_groups_list
        self.include_rings = include_rings
        self.keep_leaving_groups = keep_leaving_groups
        self.keep_incoming_groups = keep_incoming_groups
        self.keep_reagents = keep_reagents
        self.environment_atom_count = environment_atom_count
        self.keep_metadata = keep_metadata
        self.atom_info_retention = atom_info_retention
        self.info_to_clean = info_to_clean

    def to_yaml(self, filepath):
        with open(filepath, 'w') as file:
            yaml.dump(self, file, default_flow_style=False)

    @classmethod
    def from_yaml(cls, filepath):
        with open(filepath, 'r') as file:
            return yaml.load(file, Loader=yaml.FullLoader)


def extract_rules_from_reactions(config: ExtractRuleConfig):
    with RDFRead(config.reaction_database_file, indexable=True) as reactions, \
            RDFWrite(str(config.result_directory / config.rules_file_name), append=True) as result_file:
        for reaction_index, reaction in tqdm(enumerate(reactions), total=len(reactions)):
            rule = extract_rules(config, reaction)

            result_file.write(rule)


def cgr_from_rule(rule: ReactionContainer):
    reaction_rule = reaction_query_to_reaction(rule)
    cgr_rule = ~reaction_rule
    return cgr_rule


def extract_rules(config: ExtractRuleConfig, reaction: ReactionContainer) -> List[ReactionContainer]:
    """
    Extracts reaction rules from a given reaction based on the specified configuration.

    :param config: An instance of ExtractRuleConfig, which contains various configuration settings
                   for rule extraction, such as whether to include multicenter rules, functional groups,
                   ring structures, leaving and incoming groups, etc.
    :param reaction: The reaction object (ReactionContainer) from which to extract rules. The reaction
                     object represents a chemical reaction with specified reactants, products, and possibly reagents.
    :return: A list of ReactionContainer objects, each representing a distinct reaction rule. If
             config.multicenter_rules is True, a single rule encompassing all reaction centers is returned.
             Otherwise, separate rules for each reaction center are extracted, up to a maximum of 15 distinct centers.
    """
    if config.multicenter_rules:
        # Extract a single rule encompassing all reaction centers
        return [create_rule(config, reaction)]
    else:
        # Extract separate rules for each distinct reaction center
        distinct_rules = set()
        for center_reaction in islice(reaction.enumerate_centers(), 15):
            single_rule = create_rule(config, center_reaction)
            distinct_rules.add(single_rule)
        return list(distinct_rules)


def create_rule(config: ExtractRuleConfig, reaction: ReactionContainer) -> ReactionContainer:
    """
    Creates a reaction rule from a given reaction based on the specified configuration.

    :param config: An instance of ExtractRuleConfig, containing various settings that determine how
                   the rule is created, such as environmental atom count, inclusion of functional groups,
                   rings, leaving and incoming groups, and other parameters.
    :param reaction: The reaction object (ReactionContainer) from which to create the rule. This object
                     represents a chemical reaction with specified reactants, products, and possibly reagents.
    :return: A ReactionContainer object representing the extracted reaction rule. This rule includes
             various elements of the reaction as specified by the configuration, such as reaction centers,
             environmental atoms, functional groups, and others.

    The function processes the reaction to create a rule that matches the configuration settings. It handles
    the inclusion of environmental atoms, functional groups, ring structures, and leaving and incoming groups.
    It also constructs substructures for reactants, products, and reagents, and cleans molecule representations
    if required. Optionally, it validates the rule using a reactor.
    """
    cgr = ~reaction
    center_atoms = set(cgr.center_atoms)

    # Add atoms of reaction environment based on config settings
    center_atoms = add_environment_atoms(
        cgr,
        center_atoms,
        config.environment_atom_count
    )

    # Include functional groups in the rule if specified in config
    if config.include_func_groups:
        rule_atoms = add_functional_groups(
            reaction,
            center_atoms,
            config.func_groups_list
        )
    else:
        rule_atoms = center_atoms.copy()

    # Include ring structures in the rule if specified in config
    if config.include_rings:
        rule_atoms = add_ring_structures(
            cgr,
            rule_atoms,
        )

    # Add leaving and incoming groups to the rule based on config settings
    rule_atoms, meta_debug = add_leaving_incoming_groups(
        reaction,
        rule_atoms,
        config.keep_leaving_groups,
        config.keep_incoming_groups
    )

    # Create substructures for reactants, products, and reagents
    reactant_substructures, product_substructures, reagents = create_substructures_and_reagents(
        reaction,
        rule_atoms,
        config.as_query_container,
        config.keep_reagents
    )

    # Clean atom marks in the molecules if they are being converted to query containers
    if config.as_query_container:
        reactant_substructures = clean_molecules(
            reactant_substructures,
            reaction.reactants,
            center_atoms,
            config.atom_info_retention, config.info_to_clean)
        product_substructures = clean_molecules(
            product_substructures,
            reaction.products,
            center_atoms,
            config.atom_info_retention,
            config.info_to_clean
        )

    # Assemble the final rule including metadata if specified
    rule = assemble_final_rule(
        reactant_substructures,
        product_substructures,
        reagents,
        meta_debug,
        config.keep_metadata,
        reaction
    )

    # Validate the rule using a reactor if validation is enabled in config
    if config.reactor_validation:
        validate_rule(rule, reaction)

    return rule


def add_environment_atoms(cgr, center_atoms, environment_atom_count):
    """
    Adds environment atoms to the set of center atoms based on the specified depth.

    :param cgr: A complete graph representation of a reaction (ReactionContainer object).
    :param center_atoms: A set of atom identifiers representing the center atoms of the reaction.
    :param environment_atom_count: An integer specifying the depth of the environment around
                                   the reaction center to be included. If it's 0, only the
                                   reaction center is included. If it's 1, the first layer of
                                   surrounding atoms is included, and so on.
    :return: A set of atom identifiers including the center atoms and their environment atoms
             up to the specified depth. If environment_atom_count is 0, the original set of
             center atoms is returned unchanged.
    """
    if environment_atom_count:
        env_cgr = cgr.augmented_substructure(center_atoms, deep=environment_atom_count)
        # Combine the original center atoms with the new environment atoms
        return center_atoms | set(env_cgr)

    # If no environment is to be included, return the original center atoms
    return center_atoms


def add_functional_groups(reaction, center_atoms, func_groups_list):
    """
    Augments the set of rule atoms with functional groups if specified.

    :param reaction: The reaction object (ReactionContainer) from which molecules are extracted.
    :param center_atoms: A set of atom identifiers representing the center atoms of the reaction.
    :param func_groups_list: A list of functional group objects (MoleculeContainer or QueryContainer)
                             to be considered when including functional groups. These objects define
                             the structure of the functional groups to be included.
    :return: A set of atom identifiers representing the rule atoms, including atoms from the
             specified functional groups if include_func_groups is True. If include_func_groups
             is False, the original set of center atoms is returned.
    """
    rule_atoms = center_atoms.copy()
    # Iterate over each molecule in the reaction
    for molecule in reaction.molecules():
        # For each functional group specified in the list
        for func_group in func_groups_list:
            # Find mappings of the functional group in the molecule
            for mapping in func_group.get_mapping(molecule):
                # Remap the functional group based on the found mapping
                func_group.remap(mapping)
                # If the functional group intersects with center atoms, include it
                if set(func_group.atoms_numbers) & center_atoms:
                    rule_atoms |= set(func_group.atoms_numbers)
                # Reset the mapping to its original state for the next iteration
                func_group.remap({v: k for k, v in mapping.items()})
    return rule_atoms


def add_ring_structures(cgr, rule_atoms):
    """
    Appends ring structures to the set of rule atoms if they intersect with the reaction center atoms.

    :param cgr: A condensed graph representation of a reaction (CGRContainer object).
    :param rule_atoms: A set of atom identifiers representing the center atoms of the reaction.
    :return: A set of atom identifiers including the original rule atoms and the included ring structures.
    """
    for ring in cgr.sssr:
        # Check if the current ring intersects with the set of rule atoms
        if set(ring) & rule_atoms:
            # If the intersection exists, include all atoms in the ring to the rule atoms
            rule_atoms |= set(ring)
    return rule_atoms


def add_leaving_incoming_groups(reaction, rule_atoms, keep_leaving_groups, keep_incoming_groups):
    """
    Identifies and includes leaving and incoming groups to the rule atoms based on specified flags.

    :param reaction: The reaction object (ReactionContainer) from which leaving and incoming groups are extracted.
    :param rule_atoms: A set of atom identifiers representing the center atoms of the reaction.
    :param keep_leaving_groups: A boolean flag indicating whether to include leaving groups in the rule.
    :param keep_incoming_groups: A boolean flag indicating whether to include incoming groups in the rule.
    :return: Updated set of rule atoms including leaving and incoming groups if specified, and metadata about added groups.
    """
    meta_debug = {"leaving": set(), "incoming": set()}

    # Extract atoms from reactants and products
    reactant_atoms = {atom for reactant in reaction.reactants for atom in reactant}
    product_atoms = {atom for product in reaction.products for atom in product}

    # Identify leaving groups (reactant atoms not in products)
    if keep_leaving_groups:
        leaving_atoms = reactant_atoms - product_atoms
        new_leaving_atoms = leaving_atoms - rule_atoms
        # Include leaving atoms in the rule atoms
        rule_atoms |= leaving_atoms
        # Add leaving atoms to metadata
        meta_debug["leaving"] |= new_leaving_atoms

    # Identify incoming groups (product atoms not in reactants)
    if keep_incoming_groups:
        incoming_atoms = product_atoms - reactant_atoms
        new_incoming_atoms = incoming_atoms - rule_atoms
        # Include incoming atoms in the rule atoms
        rule_atoms |= incoming_atoms
        # Add incoming atoms to metadata
        meta_debug["incoming"] |= new_incoming_atoms

    return rule_atoms, meta_debug


def clean_molecules(
        rule_mols: (tuple, list),
        react_mols: (tuple, list),
        center_atoms: set,
        keep_info: str,
        info_to_remove
) -> list:
    """
    Cleans rule molecules by removing specified information about atoms.

    :param rule_mols: a list of rule molecules
    :param react_mols: a list of reaction molecules
    :param center_atoms: atoms in the reaction center
    """
    cleaned_mols = []

    for rule_mol in rule_mols:
        for react_mol in react_mols:
            if set(rule_mol.atoms_numbers) <= set(react_mol.atoms_numbers):
                query_react_mol = react_mol.substructure(as_query=True)
                query_rule_mol = query_react_mol.substructure(rule_mol)

                if keep_info == "reaction_center":
                    for atom_num in set(rule_mol.atoms_numbers) - center_atoms:
                        query_rule_mol = clean_atom(query_rule_mol, info_to_remove, atom_num)
                elif keep_info == "none":
                    for atom_num in rule_mol.atoms_numbers:
                        query_rule_mol = clean_atom(query_rule_mol, info_to_remove, atom_num)

                cleaned_mols.append(query_rule_mol)
                break

    return cleaned_mols


def clean_atom(
        query_mol: QueryContainer,
        info_to_remove,
        atom_num: int
) -> QueryContainer:
    """
    Removes specified information from a given atom in a query molecule.

    :param query_mol: the query molecule
    :param atom_num: the number of the atom to be modified
    """
    for info in info_to_remove:
        if info == "neighbors":
            query_mol.atom(atom_num).neighbors = None
        elif info == "hybridization":
            query_mol.atom(atom_num).hybridization = None
        elif info == "implicit_hydrogens":
            query_mol.atom(atom_num).implicit_hydrogens = None
        elif info == "ring_sizes":
            query_mol.atom(atom_num).ring_sizes = None

    return query_mol


def create_substructures_and_reagents(reaction, rule_atoms, as_query_container, keep_reagents):
    """
    Creates substructures for reactants and products, and optionally includes reagents, based on specified parameters.

    :param reaction: The reaction object (ReactionContainer) from which to extract substructures. This object
                     represents a chemical reaction with specified reactants, products, and possibly reagents.
    :param rule_atoms: A set of atom identifiers that define the rule atoms. These are used to identify relevant
                       substructures in reactants and products.
    :param as_query_container: A boolean flag indicating whether the substructures should be converted to query containers.
                               Query containers are used for pattern matching in chemical structures.
    :param keep_reagents: A boolean flag indicating whether reagents should be included in the resulting structures.
                          Reagents are additional substances that are present in the reaction but are not reactants or products.

    :return: A tuple containing three elements:
             - A list of reactant substructures, each corresponding to a part of the reactants that matches the rule atoms.
             - A list of product substructures, each corresponding to a part of the products that matches the rule atoms.
             - A list of reagents, included as is or as substructures, depending on the as_query_container flag.

    The function processes the reaction to create substructures for reactants and products based on the rule atoms.
    It also handles the inclusion of reagents based on the keep_reagents flag and converts these structures to query
    containers if required.
    """
    reactant_substructures = [reactant.substructure(rule_atoms.intersection(reactant.atoms_numbers))
                              for reactant in reaction.reactants if rule_atoms.intersection(reactant.atoms_numbers)]

    product_substructures = [product.substructure(rule_atoms.intersection(product.atoms_numbers))
                             for product in reaction.products if rule_atoms.intersection(product.atoms_numbers)]

    reagents = []
    if keep_reagents:
        if as_query_container:
            reagents = [reagent.substructure(reagent, as_query=True) for reagent in reaction.reagents]
        else:
            reagents = reaction.reagents

    return reactant_substructures, product_substructures, reagents


def assemble_final_rule(reactant_substructures, product_substructures, reagents, meta_debug, keep_metadata, reaction):
    """
    Assembles the final reaction rule from the provided substructures and metadata.

    :param reactant_substructures: A list of substructures derived from the reactants of the reaction.
                                   These substructures represent parts of reactants that are relevant to the rule.
    :param product_substructures: A list of substructures derived from the products of the reaction.
                                  These substructures represent parts of products that are relevant to the rule.
    :param reagents: A list of reagents involved in the reaction. These may be included as-is or as substructures,
                     depending on earlier processing steps.
    :param meta_debug: A dictionary containing additional metadata about the reaction, such as leaving and incoming groups.
    :param keep_metadata: A boolean flag indicating whether to retain the metadata associated with the reaction in the rule.
    :param reaction: The original reaction object (ReactionContainer) from which the rule is being created.

    :return: A ReactionContainer object representing the assembled reaction rule. This container includes
             the reactant and product substructures, reagents, and any additional metadata if keep_metadata is True.

    This function brings together the various components of a reaction rule, including reactant and product substructures,
    reagents, and metadata. It creates a comprehensive representation of the reaction rule, which can be used for further
    processing or analysis.
    """
    rule_metadata = meta_debug if keep_metadata else {}
    rule_metadata.update(reaction.meta if keep_metadata else {})

    rule = ReactionContainer(reactant_substructures, product_substructures, reagents, rule_metadata)

    if keep_metadata:
        rule.name = reaction.name

    rule.flush_cache()
    return rule


def validate_rule(rule: ReactionContainer, reaction: ReactionContainer):
    """
    Validates a reaction rule by ensuring it can correctly generate the products from the reactants.

    :param rule: The reaction rule to be validated. This is a ReactionContainer object representing a chemical reaction rule,
                 which includes the necessary information to perform a reaction.
    :param reaction: The original reaction object (ReactionContainer) against which the rule is to be validated. This object
                     contains the actual reactants and products of the reaction.

    :return: The validated rule if the rule correctly generates the products from the reactants.

    :raises ValueError: If the rule does not correctly generate the products from the reactants, indicating
                        an incorrect or incomplete rule.

    The function uses a chemical reactor to simulate the reaction based on the provided rule. It then compares
    the products generated by the simulation with the actual products of the reaction. If they match, the rule
    is considered valid. If not, a ValueError is raised, indicating an issue with the rule.
    """
    # Create a reactor with the given rule
    reactor = Reactor(rule)

    # Simulate the reaction and compare the generated products with the actual reactants
    for result in reactor(reaction.products):
        if sorted(result.products) == sorted(reaction.reactants):
            return rule

    # Raise an error if the rule does not correctly generate the products
    raise ValueError("Incorrect product generation in reactor from the rule.")
