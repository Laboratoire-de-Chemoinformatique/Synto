from pathlib import Path
from os.path import splitext
from typing import Union
from tqdm import tqdm

from chython import smiles, RDFRead, RDFWrite, ReactionContainer
from chython.exceptions import MappingError

from Syntool.utils import path_type


def remove_reagents_and_map(rea: ReactionContainer) -> Union[ReactionContainer, None]:
    """
    Maps atoms of the reaction using chytorch.

    :param rea: reaction to map
    :type rea: ReactionContainer

    :return: ReactionContainer or None
    """
    try:
        rea.reset_mapping()
    except MappingError:
        rea.reset_mapping()
    try:
        rea.remove_reagents()
        return rea
    except:
        # print("Error", str(rea))
        return None


def remove_reagents_and_map_from_file(input_file: path_type, output_file: path_type) -> None:
    """
    Reads a file of reactions and maps atoms of the reactions using chytorch.

    :param input_file: the path and name of the input file
    :type input_file: path_type

    :param output_file: the path and name of the output file
    :type output_file: path_type

    :return: None
    """
    input_file = str(Path(input_file).resolve(strict=True))
    _, input_ext = splitext(input_file)
    if input_ext == ".smi":
        input_file = open(input_file, "r")
    elif input_ext == ".rdf":
        input_file = RDFRead(input_file, indexable=True)
    else:
        raise ValueError("File extension not recognized. File:", input_file,
                         "- Please use smi or rdf file")
    enumerator = input_file if input_ext == ".rdf" else input_file.readlines()

    _, out_ext = splitext(output_file)
    if out_ext == ".smi":
        output_file = open(output_file, "w")
    elif out_ext == ".rdf":
        output_file = RDFWrite(output_file)
    else:
        raise ValueError("File extension not recognized. File:", output_file,
                         "- Please use smi or rdf file")

    mapping_errors = 0
    for rea_raw in tqdm(enumerator):
        rea = remove_reagents_and_map(smiles(rea_raw.strip('\n')) if input_ext == ".smi"
                                      else rea_raw)
        if rea:
            rea_output = format(rea, "m") + "\n" if out_ext == ".smi" else rea
            output_file.write(rea_output)
        else:
            mapping_errors += 1

    input_file.close()
    output_file.close()

    if mapping_errors:
        print(mapping_errors, "reactions couldn't be mapped")
