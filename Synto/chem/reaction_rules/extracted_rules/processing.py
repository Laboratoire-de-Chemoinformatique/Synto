import logging
import os
from collections import defaultdict
from pickle import dump
from typing import Tuple

from CGRtools import RDFRead, RDFWrite, ReactionContainer
from tqdm import tqdm

from .transformations import ReverseReaction


def apply_transformations(transformations: list, reaction: ReactionContainer) -> ReactionContainer:
    """
    Applies transformations to given reaction

    :param transformations: list of transformations
    :param reaction: the reaction to be transformed
    :return: result reaction
    """
    for transform in transformations:
        if isinstance(reaction, list):  # like rules list obtained from 1 reaction
            reaction = [transform(real_reaction) for real_reaction in reaction]
        else:
            reaction = transform(reaction)
    return reaction


def apply_filters(reaction: ReactionContainer, reaction_filters) -> Tuple[bool, ReactionContainer]:
    """
    Applies filters to given reaction, return a reaction and bool

    :param reaction: input reaction
    :param reaction_filters: list of filters
    :return: reaction with marks in meta and True if there are at least one filter returned True
    """
    is_filtered = False
    for reaction_filter in reaction_filters:
        if reaction_filter(reaction):
            reaction.meta[reaction_filter.__class__.__name__] = 'True'
            is_filtered = True
    return is_filtered, reaction


def reaction_database_processing(reaction_database_file_name: str, transformations: list = None, filters: list = None,
                                 save_only_unique: bool = False, result_directory_name: str = './',
                                 filtered_reactions_file_name: str = 'filtered_reactions.rdf',
                                 result_reactions_file_name: str = 'reaction_rules.rdf',
                                 result_reactions_pkl_file_name: str = 'reaction_rules.pickle',
                                 remove_old_results: bool = True, min_popularity: int = 3):
    """
    Applies given transformations and filters to reactions from the reaction database. Returns result reactions files in
    RDF and pickle (if save_only_unique is True) formats and filtered reactions file in RDF format

    :param reaction_database_file_name: path to the reaction database (.rdf format)
    :param transformations: list of transformations
    :param filters: list of filters
    :param save_only_unique: if True, then only unique reactions with information about frequency are saved
    :param result_directory_name: result directory name
    :param filtered_reactions_file_name: filtered and error reactions file name (.rdf)
    :param result_reactions_file_name: result reactions file name (.rdf)
    :param result_reactions_pkl_file_name: result reactions file name (.pickle)
    :param remove_old_results: if previously extracted reactions and rules are removed.
    :param min_popularity: the rule should appear at least min_popularity times in the reactions set to be considered.
    """
    os.makedirs(result_directory_name, exist_ok=True)

    if remove_old_results:
        if os.path.isfile(f'{result_directory_name}/{filtered_reactions_file_name}'):
            os.remove(f'{result_directory_name}/{filtered_reactions_file_name}')
            logging.warning(f"Removed {result_directory_name}/{filtered_reactions_file_name}")

        if os.path.isfile(f'{result_directory_name}/{result_reactions_file_name}'):
            os.remove(f'{result_directory_name}/{result_reactions_file_name}')
            logging.warning(f"Removed {result_directory_name}/{result_reactions_file_name}")

        if os.path.isfile(f'{result_directory_name}/unique_{result_reactions_file_name}'):
            os.remove(f'{result_directory_name}/unique_{result_reactions_file_name}')
            logging.warning(f"Removed {result_directory_name}/unique_{result_reactions_file_name}")

    # filtered_file_exist = os.path.isfile(f'{result_directory_name}/{filtered_reactions_file_name}')
    # results_file_exist = os.path.isfile(f'{result_directory_name}/{filtered_reactions_file_name}')
    # unique_file_exist = os.path.isfile(f'{result_directory_name}/unique_{result_reactions_file_name}')

    if save_only_unique:
        unique_reactions = defaultdict(list)

    with RDFRead(reaction_database_file_name, indexable=True) as reactions, RDFWrite(
        f'{result_directory_name}/{filtered_reactions_file_name}', append=True) as filtered_file, RDFWrite(
        f'{result_directory_name}/{result_reactions_file_name}', append=True) as result_file:
        reactions.reset_index()
        for n, reaction in tqdm(enumerate(reactions), total=len(reactions)):
            try:
                if filters:
                    is_filtered, reaction = apply_filters(reaction, filters)
                    if is_filtered:
                        filtered_file.write(reaction)
                        continue
                if transformations:
                    reaction = apply_transformations(transformations, reaction)
            except Exception as e:
                print(e)
                reaction.meta['Error'] = 'True'
                filtered_file.write(reaction)
            else:
                if type(reaction) != list:
                    reaction = [reaction]
                for real_reaction in reaction:
                    real_reaction.clean2d()
                    if save_only_unique:
                        unique_reactions[real_reaction].append(n)
                    else:
                        real_reaction.meta['reaction_index'] = n
                    result_file.write(real_reaction)

    if save_only_unique:

        pop_reactions = []
        for result_reaction, reaction_ids in unique_reactions.items():
            result_reaction.meta['reaction_ids'] = tuple(i for i in reaction_ids)
            if len(result_reaction.meta['reaction_ids']) >= min_popularity:
                pop_reactions.append(result_reaction)
            else:
                pass

        with RDFWrite(f'{result_directory_name}/unique_{result_reactions_file_name}') as unique_file:
            for result_reaction in pop_reactions:
                unique_file.write(result_reaction)
            print(f'{len(pop_reactions)} reaction rules were extracted')

        with open(f'{result_directory_name}/{result_reactions_pkl_file_name}', 'wb') as pickle_file:

            # reverse reaction rules
            reverse_reaction = ReverseReaction()
            pop_rections = [reverse_reaction(i) for i in pop_reactions]

            dump(pop_rections, pickle_file)
