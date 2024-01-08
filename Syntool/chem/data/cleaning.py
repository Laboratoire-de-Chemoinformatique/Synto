import os
from multiprocessing import Queue, Process, Manager
from logging import warning, getLogger
from tqdm import tqdm
from CGRtools.containers import ReactionContainer
from CGRtools.files import RDFRead, RDFWrite
from .standardizer import Standardizer


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
        if res:
            to_write.put(res)
    logger.disabled = False


def cleaner_writer(output_file: str, to_write: Queue, remove_old=True):
    """
    Writes in output file the standardized reactions

    :param output_file: output file path
    :param to_write: Standardized ReactionContainer to write
    :param remove_old: whenever to remove or not an already existing file
    """

    if remove_old and os.path.isfile(output_file):
        os.remove(output_file)
        # warning(f"Removed {output_file}")

    with RDFWrite(output_file) as out:
        while True:
            res = to_write.get()
            if res == "Quit":
                break
            out.write(res)


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

        writer = Process(target=cleaner_writer, args=(output_file, to_write,))
        writer.start()

        workers = []
        for _ in range(num_cpus - 2):
            w = Process(target=worker_cleaner, args=(to_clean, to_write))
            w.start()
            workers.append(w)

        with RDFRead(input_file, indexable=True) as reactions:
            reactions.reset_index()
            print(f'Total number of reactions: {len(reactions)}')
            for n, raw_reaction in tqdm(enumerate(reactions), total=len(reactions)):
                to_clean.put(raw_reaction)
        #
        # TODO finish it
        # n_removed = len(reactions) - len(RDFRead(output_file, indexable=True))
        # print(f'Removed number of reactions: {n_removed} ({100 * n_removed / len(reactions):.1f} %)')

        for _ in workers:
            to_clean.put("Quit")
        for w in workers:
            w.join()

        to_write.put("Quit")
        writer.join()
