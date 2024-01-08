"""
Module containing functions with fixed protocol for reaction rules extraction
"""
import logging
import pickle
from collections import defaultdict
from itertools import islice
from pathlib import Path
from typing import List, Union, Tuple, IO, Dict, Set, Iterable, Any

import ray
from CGRtools.containers import MoleculeContainer, QueryContainer, ReactionContainer
from CGRtools.exceptions import InvalidAromaticRing
from CGRtools.files import RDFRead, RDFWrite
from CGRtools.reactor import Reactor
from tqdm.auto import tqdm

from Syntool.chem.utils import reverse_reaction
from Syntool.utils.config import ExtractRuleConfig


def extract_rules_from_reactions(
    config: ExtractRuleConfig,
    reaction_file: str,
    results_root: str,
    rules_file_name: str = 'reaction_rules',
    num_cpus: int = 1,
    batch_size: int = 10,
) -> None:
    """
    Extracts reaction rules from a set of reactions based on the given configuration.

    This function initializes a Ray environment for distributed computing and processes each reaction
    in the provided reaction database to extract reaction rules. It handles the reactions in batches,
    parallelize the rule extraction process. Extracted rules are written to RDF files and their statistics
    are recorded. The function also sorts the rules based on their popularity and saves the sorted rules.

    :param config: Configuration settings for rule extraction, including file paths, batch size, and other parameters.
    :param reaction_file: Path to the file containing reaction database.
    :param results_root: Path of the directory where the results will be stored.
    :param rules_file_name: Name of the file to store the extracted rules.
    :param num_cpus: Number of CPU cores to use for processing. Defaults to 1.
    :param batch_size: Number of reactions to process in each batch. Defaults to 10.

    :return: None
    """
    # create reaction rules config
    rules_config = ExtractRuleConfig(
        min_popularity=config['ReactionRules']['min_popularity'],
        keep_leaving_groups=config['ReactionRules']['keep_leaving_groups'],
        atom_info_retention={
            "reaction_center": {
                "neighbors": config['ReactionRules']['reaction_center_neighbors'],
                "hybridization": config['ReactionRules']['reaction_center_hybridization'],
                "implicit_hydrogens": config['ReactionRules']['reaction_center_implicit_hydrogens'],
                "ring_sizes": config['ReactionRules']['reaction_center_ring_sizes'],
            },
            "environment": {
                "neighbors": config['ReactionRules']['environment_neighbors'],
                "hybridization": config['ReactionRules']['environment_hybridization'],
                "implicit_hydrogens": config['ReactionRules']['environment_implicit_hydrogens'],
                "ring_sizes": config['ReactionRules']['environment_ring_sizes'],
            },
        }
    )

    # read files
    reaction_file = Path(reaction_file).resolve(strict=True)
    results_root = Path(results_root)
    results_root.mkdir(parents=True, exist_ok=True)

    ray.init(num_cpus=num_cpus, ignore_reinit_error=True, logging_level=logging.ERROR)

    with RDFRead(reaction_file, indexable=True) as reactions:
        total_reactions = len(reactions)
        pbar = tqdm(total=total_reactions, disable=False)  # TODO progress bar disappears after finishing

        futures = {}
        batch = []
        max_concurrent_batches = num_cpus

        rules_statistics = defaultdict(list)
        with RDFWrite(
            results_root / f"{rules_file_name}_full.rdf", append=True
        ) as result_file:
            for index, reaction in enumerate(reactions):
                batch.append((index, reaction))
                if len(batch) == batch_size:
                    future = process_reaction_batch.remote(batch, rules_config)
                    futures[future] = None
                    batch = []

                    while len(futures) >= max_concurrent_batches:
                        process_completed_batches(
                            futures, result_file, rules_statistics, pbar, batch_size
                        )

            if batch:
                future = process_reaction_batch.remote(batch, rules_config)
                futures[future] = None

            while futures:
                process_completed_batches(
                    futures, result_file, rules_statistics, pbar, batch_size
                )

            pbar.close()

        with open(
            results_root / f"{rules_file_name}_full.pickle", "wb"
        ) as statistics_file:
            pickle.dump(rules_statistics, statistics_file)

        sorted_rules = sort_rules(
            rules_statistics,
            min_popularity=rules_config.min_popularity,
            single_reactant_only=rules_config.single_reactant_only,
        )

        with open(
            results_root / f"{rules_file_name}_filtered.pickle", "wb"
        ) as statistics_file:
            pickle.dump(sorted_rules, statistics_file)

    ray.shutdown()


@ray.remote
def process_reaction_batch(
    batch: List[Tuple[int, ReactionContainer]], config: ExtractRuleConfig
) -> list[tuple[int, list[ReactionContainer]]]:
    """
    Processes a batch of reactions to extract reaction rules based on the given configuration.

    This function operates as a remote task in a distributed system using Ray. It takes a batch of reactions,
    where each reaction is paired with an index. For each reaction in the batch, it extracts reaction rules
    as specified by the configuration object. The extracted rules for each reaction are then returned along
    with the corresponding index.

    :param batch: A list where each element is a tuple containing an index (int) and a ReactionContainer object.
                  The index is typically used to keep track of the reaction's position in a larger dataset.
    :type batch: List[Tuple[int, ReactionContainer]]

    :param config: An instance of ExtractRuleConfig that provides settings and parameters for the rule extraction process.
    :type config: ExtractRuleConfig

    :return: A list where each element is a tuple. The first element of the tuple is an index (int), and the second
             is a list of ReactionContainer objects representing the extracted rules for the corresponding reaction.
    :rtype: list[tuple[int, list[ReactionContainer]]]

    This function is intended to be used in a distributed manner with Ray to parallelize the rule extraction
    process across multiple reactions.
    """
    processed_batch = []
    for index, reaction in batch:
        extracted_rules = extract_rules(config, reaction)
        processed_batch.append((index, extracted_rules))
    return processed_batch


def process_completed_batches(
    futures: dict,
    result_file: IO,
    rules_statistics: Dict[ReactionContainer, List[int]],
    pbar: tqdm,
    batch_size: int,
) -> None:
    """
    Processes completed batches of reactions, updating the rules statistics and writing rules to a file.

    This function waits for the completion of a batch of reactions processed in parallel (using Ray),
    updates the statistics for each extracted rule, and writes the rules to a result file if they are new.
    It also updates the progress bar with the size of the processed batch.

    :param futures: A dictionary of futures representing ongoing batch processing tasks.
    :type futures: dict

    :param result_file: An open file object to which extracted rules are written.
    :type result_file: IO

    :param rules_statistics: A dictionary to keep track of statistics for each rule.
    :type rules_statistics: Dict[ReactionContainer, List[int]]

    :param pbar: A tqdm progress bar instance for updating the progress of batch processing.
    :type pbar: tqdm

    :param batch_size: The number of reactions processed in each batch.
    :type batch_size: int

    :return: None
    """
    done, _ = ray.wait(list(futures.keys()), num_returns=1)
    completed_batch = ray.get(done[0])

    for index, extracted_rules in completed_batch:
        for rule in extracted_rules:
            prev_stats_len = len(rules_statistics)
            rules_statistics[rule].append(index)
            if len(rules_statistics) != prev_stats_len:
                rule.meta["first_reaction_index"] = index
                result_file.write(rule)

    del futures[done[0]]
    pbar.update(batch_size)


def extract_rules(
    config: ExtractRuleConfig, reaction: ReactionContainer
) -> list[ReactionContainer]:
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


def create_rule(
    config: ExtractRuleConfig, reaction: ReactionContainer
) -> ReactionContainer:
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
        cgr, center_atoms, config.environment_atom_count
    )

    # Include functional groups in the rule if specified in config
    if config.include_func_groups:
        rule_atoms = add_functional_groups(
            reaction, center_atoms, config.func_groups_list
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
        reaction, rule_atoms, config.keep_leaving_groups, config.keep_incoming_groups
    )

    # Create substructures for reactants, products, and reagents
    (
        reactant_substructures,
        product_substructures,
        reagents,
    ) = create_substructures_and_reagents(
        reaction, rule_atoms, config.as_query_container, config.keep_reagents
    )

    # Clean atom marks in the molecules if they are being converted to query containers
    if config.as_query_container:
        reactant_substructures = clean_molecules(
            reactant_substructures,
            reaction.reactants,
            center_atoms,
            config.atom_info_retention,
        )
        product_substructures = clean_molecules(
            product_substructures,
            reaction.products,
            center_atoms,
            config.atom_info_retention,
        )

    # Assemble the final rule including metadata if specified
    rule = assemble_final_rule(
        reactant_substructures,
        product_substructures,
        reagents,
        meta_debug,
        config.keep_metadata,
        reaction,
    )

    if config.reverse_rule:
        rule = reverse_reaction(rule)
        reaction = reverse_reaction(reaction)

    # Validate the rule using a reactor if validation is enabled in config
    if config.reactor_validation:
        if validate_rule(rule, reaction):
            rule.meta["reactor_validation"] = "passed"
        else:
            rule.meta["reactor_validation"] = "failed"

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


def add_leaving_incoming_groups(
    reaction, rule_atoms, keep_leaving_groups, keep_incoming_groups
):
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
    rule_molecules: Iterable[QueryContainer],
    reaction_molecules: Iterable[MoleculeContainer],
    reaction_center_atoms: Set[int],
    atom_retention_details: Dict[str, Dict[str, bool]],
) -> List[QueryContainer]:
    """
    Cleans rule molecules by removing specified information about atoms based on retention details provided.

    :param rule_molecules: A list of query container objects representing the rule molecules.
    :param reaction_molecules: A list of molecule container objects involved in the reaction.
    :param reaction_center_atoms: A set of integers representing atom numbers in the reaction center.
    :param atom_retention_details: A dictionary specifying what atom information to retain or remove.
                                   This dictionary should have two keys: "reaction_center" and "environment",
                                   each mapping to another dictionary. The nested dictionaries should have
                                   keys representing atom attributes (like "neighbors", "hybridization",
                                   "implicit_hydrogens", "ring_sizes") and boolean values. A value of True
                                   indicates that the corresponding attribute should be retained,
                                   while False indicates it should be removed from the atom.

                                   For example:
                                   {
                                       "reaction_center": {"neighbors": True, "hybridization": False, ...},
                                       "environment": {"neighbors": True, "implicit_hydrogens": False, ...}
                                   }

    Returns:
        A list of QueryContainer objects representing the cleaned rule molecules.
    """
    cleaned_rule_molecules = []

    for rule_molecule in rule_molecules:
        for reaction_molecule in reaction_molecules:
            if set(rule_molecule.atoms_numbers) <= set(reaction_molecule.atoms_numbers):
                query_reaction_molecule = reaction_molecule.substructure(
                    reaction_molecule, as_query=True
                )
                query_rule_molecule = query_reaction_molecule.substructure(
                    rule_molecule
                )

                # Clean environment atoms
                if not all(
                    atom_retention_details["environment"].values()
                ):  # if everything True, we keep all marks
                    local_environment_atoms = (
                        set(rule_molecule.atoms_numbers) - reaction_center_atoms
                    )
                    for atom_number in local_environment_atoms:
                        query_rule_molecule = clean_atom(
                            query_rule_molecule,
                            atom_retention_details["environment"],
                            atom_number,
                        )

                # Clean reaction center atoms
                if not all(
                    atom_retention_details["reaction_center"].values()
                ):  # if everything True, we keep all marks
                    local_reaction_center_atoms = (
                        set(rule_molecule.atoms_numbers) & reaction_center_atoms
                    )
                    for atom_number in local_reaction_center_atoms:
                        query_rule_molecule = clean_atom(
                            query_rule_molecule,
                            atom_retention_details["reaction_center"],
                            atom_number,
                        )

                cleaned_rule_molecules.append(query_rule_molecule)
                break

    return cleaned_rule_molecules


def clean_atom(
    query_molecule: QueryContainer,
    attributes_to_keep: Dict[str, bool],
    atom_number: int,
) -> QueryContainer:
    """
    Removes specified information from a given atom in a query molecule.

    :param query_molecule: The QueryContainer of molecule.
    :param attributes_to_keep: Dictionary indicating which attributes to keep in the atom.
                                 The keys should be strings representing the attribute names, and
                                 the values should be booleans indicating whether to retain (True)
                                 or remove (False) that attribute. Expected keys are:
                                 - "neighbors": Indicates if neighbors of the atom should be removed.
                                 - "hybridization": Indicates if hybridization information of the atom should be removed.
                                 - "implicit_hydrogens": Indicates if implicit hydrogen information of the atom should be removed.
                                 - "ring_sizes": Indicates if ring size information of the atom should be removed.
    :param atom_number: The number of the atom to be modified in the query molecule.
    """
    target_atom = query_molecule.atom(atom_number)

    if not attributes_to_keep["neighbors"]:
        target_atom.neighbors = None
    if not attributes_to_keep["hybridization"]:
        target_atom.hybridization = None
    if not attributes_to_keep["implicit_hydrogens"]:
        target_atom.implicit_hydrogens = None
    if not attributes_to_keep["ring_sizes"]:
        target_atom.ring_sizes = None

    return query_molecule


def create_substructures_and_reagents(
    reaction, rule_atoms, as_query_container, keep_reagents
):
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
    reactant_substructures = [
        reactant.substructure(rule_atoms.intersection(reactant.atoms_numbers))
        for reactant in reaction.reactants
        if rule_atoms.intersection(reactant.atoms_numbers)
    ]

    product_substructures = [
        product.substructure(rule_atoms.intersection(product.atoms_numbers))
        for product in reaction.products
        if rule_atoms.intersection(product.atoms_numbers)
    ]

    reagents = []
    if keep_reagents:
        if as_query_container:
            reagents = [
                reagent.substructure(reagent, as_query=True)
                for reagent in reaction.reagents
            ]
        else:
            reagents = reaction.reagents

    return reactant_substructures, product_substructures, reagents


def assemble_final_rule(
    reactant_substructures,
    product_substructures,
    reagents,
    meta_debug,
    keep_metadata,
    reaction,
):
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

    rule = ReactionContainer(
        reactant_substructures, product_substructures, reagents, rule_metadata
    )

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
    try:
        for result_reaction in reactor(reaction.reactants):
            result_products = []
            for result_product in result_reaction.products:
                tmp = result_product.copy()
                try:
                    tmp.kekule()
                    if tmp.check_valence():
                        continue
                except InvalidAromaticRing:
                    continue
                result_products.append(result_product)
            if set(reaction.products) == set(result_products) and len(
                reaction.products
            ) == len(result_products):
                return True
    except (KeyError, IndexError):
        # KeyError - iteration over reactor is finished and products are different from the original reaction
        # IndexError - mistake in __contract_ions, possibly problems with charges in rule?
        return False


def sort_rules(
    rules_stats: Dict[ReactionContainer, List[int]],
    min_popularity: int = 3,
    single_reactant_only: bool = True,
) -> List[Tuple[ReactionContainer, List[int]]]:
    """
    Sorts reaction rules based on their popularity and validation status.

    This function sorts the given rules according to their popularity (i.e., the number of times they have been
    applied) and filters out rules that haven't passed reactor validation or are less popular than the specified
    minimum popularity threshold.

    :param rules_stats: A dictionary where each key is a reaction rule and the value is a list of integers.
                        Each integer represents an index where the rule was applied.
    :type rules_stats: Dict[ReactionContainer, List[int]]

    :param min_popularity: The minimum number of times a rule must be applied to be considered. Default is 3.
    :type min_popularity: int

    :param single_reactant_only: Whether to keep only reaction rules with a single molecule on the right side
    of reaction arrow. Default is True.

    :return: A list of tuples, where each tuple contains a reaction rule and a list of indices representing
             the rule's applications. The list is sorted in descending order of the rule's popularity.
    :rtype: List[Tuple[ReactionContainer, List[int]]]
    """
    return sorted(
        (
            (rule, indices)
            for rule, indices in rules_stats.items()
            if len(indices) >= min_popularity
            and rule.meta["reactor_validation"] == "passed"
            and (not single_reactant_only or len(rule.reactants) == 1)
        ),
        key=lambda x: -len(x[1]),
    )
