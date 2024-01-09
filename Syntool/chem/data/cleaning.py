import os
from multiprocessing import Queue, Process, Manager, Value
from logging import getLogger
from tqdm import tqdm
from CGRtools.containers import ReactionContainer

from .standardizer import Standardizer
from Syntool.utils.files import ReactionReader, ReactionWriter


def cleaner(reaction: ReactionContainer, logger):
    """
    Standardize a reaction according to external script

    :param reaction: ReactionContainer to clean/standardize
    :param logger: Logger - to avoid writing log
    :return: ReactionContainer or empty list
    """
    standardizer = Standardizer(skip_errors=True, keep_unbalanced_ions=False, id_tag='Reaction_ID', keep_reagents=False,
                                ignore_mapping=True, action_on_isotopes=2, skip_tautomerize=True, logger=logger)
    return standardizer.standardize(reaction)


def worker_cleaner(to_clean: Queue, to_write: Queue):
    """
    Launches standardizations using the Queue to_clean. Fills the to_write Queue with results

    :param to_clean: Queue of reactions to clean/standardize
    :param to_write: Standardized outputs to write
    """
    logger = getLogger()
    logger.disabled = True
    while True:
        raw_reaction = to_clean.get()
        if raw_reaction == "Quit":
            break
        res = cleaner(raw_reaction, logger)
        to_write.put(res)
    logger.disabled = False


def cleaner_writer(output_file: str, to_write: Queue, cleaned_nb: Value, remove_old=True):
    """
    Writes in output file the standardized reactions

    :param output_file: output file path
    :param to_write: Standardized ReactionContainer to write
    :param cleaned_nb: number of final reactions
    :param remove_old: whenever to remove or not an already existing file
    """

    if remove_old and os.path.isfile(output_file):
        os.remove(output_file)

    counter = 0
    seen_reactions = []
    with ReactionWriter(output_file) as out:
        while True:
            res = to_write.get()
            if res:
                if res == "Quit":
                    cleaned_nb.set(counter)
                    break
                elif isinstance(res, ReactionContainer):
                    smi = format(res, "m")
                    if smi not in seen_reactions:
                        out.write(res)
                        counter += 1
                        seen_reactions.append(smi)


def reactions_cleaner(input_file: str, output_file: str, num_cpus: int, batch_prep_size: int = 100):
    """
    Writes in output file the standardized reactions

    :param input_file: input RDF file path
    :param output_file: output RDF file path
    :param num_cpus: number of CPU to be parallelized
    :param batch_prep_size: size of each batch per CPU
    """
    with Manager() as m:
        to_clean = m.Queue(maxsize=num_cpus * batch_prep_size)
        to_write = m.Queue(maxsize=batch_prep_size)
        cleaned_nb = m.Value(int, 0)

        writer = Process(target=cleaner_writer, args=(output_file, to_write, cleaned_nb))
        writer.start()

        workers = []
        for _ in range(num_cpus - 2):
            w = Process(target=worker_cleaner, args=(to_clean, to_write))
            w.start()
            workers.append(w)

        n = 0
        with ReactionReader(input_file) as reactions:
            for raw_reaction in tqdm(reactions):
                if 'Reaction_ID' not in raw_reaction.meta:
                    raw_reaction.meta['Reaction_ID'] = n
                to_clean.put(raw_reaction)
                n += 1

        for _ in workers:
            to_clean.put("Quit")
        for w in workers:
            w.join()

        to_write.put("Quit")
        writer.join()

        n_removed = n - cleaned_nb.get()
        print(f'Initial number of reactions: {n}'),
        print(f'Removed number of reactions: {n_removed} ({100 * n_removed / n:.2f} %)')
