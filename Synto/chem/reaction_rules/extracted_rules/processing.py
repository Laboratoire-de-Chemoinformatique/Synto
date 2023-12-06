import logging
import os
from collections import defaultdict
from pickle import dump
from typing import Tuple, Dict, List, Optional

from tqdm import tqdm

from CGRtools.files import RDFRead, RDFWrite
from CGRtools.containers import ReactionContainer
from Synto.chem.reaction_rules.extracted_rules.transformations import ReverseReaction


def apply_transformations(transformations: list, reaction: ReactionContainer) -> ReactionContainer:
    for transform in transformations:
        reaction = transform(reaction) if isinstance(reaction, ReactionContainer) else [transform(r) for r in reaction]
    return reaction


def apply_filters(reaction: ReactionContainer, reaction_filters) -> Tuple[bool, ReactionContainer]:
    is_filtered = False
    for reaction_filter in reaction_filters:
        if reaction_filter(reaction):
            reaction.meta[reaction_filter.__class__.__name__] = 'True'
            is_filtered = True
    return is_filtered, reaction


def process_reaction(reaction: ReactionContainer,
                     filters: Optional[List[callable]],
                     transformations: Optional[List[callable]],
                     unique_reactions: Optional[Dict[ReactionContainer, List[int]]],
                     reaction_index: int,
                     save_only_unique: bool) -> Optional[ReactionContainer]:
    """
    Process a single reaction with given filters and transformations.

    :param reaction: The reaction to be processed.
    :param filters: A list of filter functions to be applied to the reaction.
    :param transformations: A list of transformation functions to be applied to the reaction.
    :param unique_reactions: A dictionary to track unique reactions if save_only_unique is True.
    :param reaction_index: The index of the current reaction.
    :param save_only_unique: Flag to indicate if only unique reactions should be saved.
    :return: Processed reaction or None if filtered out.
    """
    if filters:
        for reaction_filter in filters:
            if reaction_filter(reaction):
                return None

    if transformations:
        for transform in transformations:
            reaction = transform(reaction)

    reaction.clean2d()
    if save_only_unique:
        unique_reactions[reaction].append(reaction_index)

    return reaction


def reaction_database_processing(
        reaction_database_file_name: str,
        transformations: list = None,
        filters: list = None,
        save_only_unique: bool = False,
        result_directory_name: str = './',
        filtered_reactions_file_name: str = 'filtered_reactions.rdf',
        result_reactions_file_name: str = 'reaction_rules.rdf',
        result_reactions_pkl_file_name: str = 'reaction_rules.pickle',
        remove_old_results: bool = True,
        min_popularity: int = 3
):
    """
        Processes a database of chemical reactions, applying given transformations and filters,
        and writes the results to specified files.

        :param reaction_database_file_name: Path to the reaction database file in RDF format.
        :param transformations: A list of transformation functions to be applied to each reaction. Default is None.
        :param filters: A list of filter functions to apply to each reaction. Reactions that pass the filters are
        written to the filtered reactions file. Default is None.
        :param save_only_unique: If True, only unique reactions are saved, based on their frequency and
        the min_popularity parameter. Default is False.
        :param result_directory_name: Directory path where the result files will be saved. Default is './'.
        :param filtered_reactions_file_name: Filename for the RDF file where filtered reactions are saved.
        Default is 'filtered_reactions.rdf'.
        :param result_reactions_file_name: Filename for the RDF file where processed
        (transformed and non-filtered) reactions are saved. Default is 'reaction_rules.rdf'.
        :param result_reactions_pkl_file_name: Filename for the pickle file where processed unique reactions are saved,
        if save_only_unique is True. Default is 'reaction_rules.pickle'.
        :param remove_old_results: If True, any existing files with the same names in the result directory will be
        removed before processing starts. Default is True.
        :param min_popularity: Minimum frequency for a reaction to be considered popular and saved when
        save_only_unique is True. Default is 3.

        :return: None. The function writes the processed reactions to specified RDF and pickle files.
        Unique reactions are written if save_only_unique is True.

    """
    os.makedirs(result_directory_name, exist_ok=True)
    remove_files_if_exists(result_directory_name, [filtered_reactions_file_name, result_reactions_file_name,
                                                   f"unique_{result_reactions_file_name}"], remove_old_results)

    unique_reactions = defaultdict(list) if save_only_unique else None

    with RDFRead(reaction_database_file_name, indexable=True) as reactions, \
            RDFWrite(f'{result_directory_name}/{filtered_reactions_file_name}', append=True) as filtered_file, \
            RDFWrite(f'{result_directory_name}/{result_reactions_file_name}', append=True) as result_file:

        for reaction_index, reaction in tqdm(enumerate(reactions), total=len(reactions)):
            processed_reaction = process_reaction(
                reaction,
                filters,
                transformations,
                unique_reactions,
                reaction_index,
                save_only_unique
            )

            if processed_reaction is None:  # Reaction was filtered out
                filtered_file.write(reaction)
            else:
                if save_only_unique:
                    unique_reactions[processed_reaction].append(reaction_index)
                else:
                    processed_reaction.meta['reaction_index'] = reaction_index
                    result_file.write(processed_reaction)

    if save_only_unique:
        write_unique_reactions(unique_reactions, result_directory_name, result_reactions_file_name,
                               result_reactions_pkl_file_name, min_popularity)


def remove_files_if_exists(directory, file_names, remove_flag):
    for file_name in file_names:
        file_path = os.path.join(directory, file_name)
        if remove_flag and os.path.isfile(file_path):
            os.remove(file_path)
            logging.warning(f"Removed {file_path}")


def write_unique_reactions(unique_reactions, directory, rdf_file_name, pkl_file_name, min_popularity):
    popular_reactions = [reaction for reaction, ids in unique_reactions.items() if len(ids) >= min_popularity]
    with RDFWrite(f'{directory}/unique_{rdf_file_name}') as unique_file:
        for reaction in popular_reactions:
            unique_file.write(reaction)

    with open(f'{directory}/{pkl_file_name}', 'wb') as pickle_file:
        reverse_reaction = ReverseReaction()
        reversed_popular_reactions = [reverse_reaction(r) for r in popular_reactions]
        dump(reversed_popular_reactions, pickle_file)

    logging.info(f"{len(popular_reactions)} reaction rules were extracted")
