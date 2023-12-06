"""
Module containing functions with fixed protocol for reaction rules extraction
"""

from .extracted_rules.filters import CheckCCsp3Breaking, CheckCCRingBreaking, CheckRulesByPopularity
from .extracted_rules.filters import CheckCGRConnectedComponents, CheckDynamicBondsNumber, CheckSmallMolecules
from .extracted_rules.filters import CheckNoReaction, CheckMultiCenterReaction, CheckWrongCHBreaking
from .extracted_rules.processing import reaction_database_processing
from .extracted_rules.transformations import ExtractRule

filters = [
    CheckCGRConnectedComponents(),
    CheckDynamicBondsNumber(),
    CheckSmallMolecules(),
    CheckNoReaction(),
    CheckMultiCenterReaction(),
    CheckWrongCHBreaking(),
    CheckCCsp3Breaking(),
    CheckCCRingBreaking(),
]

transformations = [
    ExtractRule(
        rules_from_multistage_reaction=False,
        environment_atoms_number=1,
        rule_with_functional_groups=False,
        functional_groups_list=None,
        include_rings=True,
        keep_leaving_groups=True,
        keep_coming_groups=True,
        keep_reagents=False,
        keep_meta=False,
        as_query=True,
        keep_atom_info="reaction_center",
        clean_info=frozenset({"neighbors", "hybridization", "implicit_hydrogens"}),
        check_in_reactor=False,
    )
]


def extract_reaction_rules(reaction_file: str = None, results_root: str = None, min_popularity: int = 3):
    """
    The function extracts reaction rules from a reaction file and saves the results to a specified directory.

    :param min_popularity: the rule should appear at least min_popularity times in the reactions set to be considered.
    :type reaction_file: int
    :param reaction_file: The path to the file containing the reaction rules. This file should be in a specific format.
    (RDF) that allows the extraction of the reaction rules
    :type reaction_file: str
    :param results_root: The string that specifies the root directory where the extracted reaction rules will be stored.
    :type results_root: str
    """
    return reaction_database_processing(
        reaction_file,
        transformations=transformations,
        filters=filters,
        result_directory_name=results_root,
        save_only_unique=True,
        min_popularity=min_popularity
    )
