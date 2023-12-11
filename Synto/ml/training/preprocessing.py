"""
Module containing functions for preparation of the training sets for policy and value network
"""

import os
import pickle
from abc import ABC
from multiprocessing import Manager, Pool
from pathlib import Path
from typing import List

import ray
import torch
from CGRtools import smiles
from CGRtools.containers import MoleculeContainer, ReactionContainer
from CGRtools.exceptions import InvalidAromaticRing
from CGRtools.files import SMILESRead, RDFRead
from CGRtools.reactor import Reactor
# from queue import Queue, Empty
from ray.util.queue import Queue, Empty
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.data.makedirs import makedirs
from torch_geometric.transforms import ToUndirected
from tqdm import tqdm

from Synto.chem.loading import load_reaction_rules
from Synto.chem.utils import unite_molecules


class ValueNetworkDataset(InMemoryDataset, ABC):
    """
    Value network dataset
    """

    def __init__(self, processed_molecules_path=None):
        """
        Initializes a value network dataset object.

        :param processed_molecules_path: The path to a file containing processed molecules (retrons) extracted from
        search tree.
        """
        super().__init__(None, None, None)

        if processed_molecules_path:
            self.data, self.slices = self.prepare_from_path(processed_molecules_path)

    def prepare_pyg(self, molecule):
        """
        It takes a molecule as input, and converts the molecule to a PyTorch geometric graph,
        assigns the reward value (label) to the graph, and returns the graph.

        :param molecule: The molecule object that represents a chemical compound.
        :return: a PyTorch Geometric (PyG) graph representation of a molecule. If the molecule has a "label" key in its
        metadata, the function sets the reward variable to the value of the "label" key converted to a float. Otherwise,
        the reward variable is set to 0.
        """
        if len(molecule) > 2:
            if "label" in molecule.meta.keys():
                reward = float(molecule.meta["label"])
            else:
                reward = 0

            pyg = mol_to_pyg(molecule)
            if pyg:
                pyg.y = torch.tensor([reward])
                return pyg
            else:
                return None

    def prepare_from_path(self, processed_molecules_path):
        """
        The function prepares processed data from a given file path by reading SMILES data, converting it to
        PyTorch geometric graph format, and returning the processed data and slices.

        :param processed_molecules_path: The path to a file containing processed molecules. It is assumed that the file
        is in a format that can be read by the SMILESRead class, and that it has a header row with a column
        named "label"
        :return: data (PyTorch geometric graphs) and slices.
        """
        processed_data = []
        with SMILESRead(processed_molecules_path, header=["label"]) as inp:
            for mol in inp:
                pyg = self.prepare_pyg(mol)
                if pyg:
                    processed_data.append(pyg)
        data, slices = self.collate(processed_data)
        return data, slices


class RankingPolicyDataset(InMemoryDataset):
    """
    Policy network dataset
    """

    def __init__(self, reactions_path, reaction_rules_path, output_path):
        """
        Initializes a policy network dataset object.

        :param reactions_path: The path to the file containing the reaction data used for extraction of reaction rules.
        :param reaction_rules_path: The path to the file containing the reaction rules.
        :param output_path: The output path is the location where policy network dataset will be stored.
        """
        super().__init__(None, None, None)

        self.reactions_path = Path(reactions_path).resolve(strict=True)
        self.reaction_rules_path = Path(reaction_rules_path).resolve(strict=True)
        self.output_path = output_path

        if output_path and os.path.exists(output_path):
            self.data, self.slices = torch.load(self.output_path)
        else:
            self.data, self.slices = self.prepare_data()

    @property
    def num_classes(self) -> int:
        return self._infer_num_classes(self._data.y_rules)

    def prepare_data(self):
        """
        The function prepares data by loading reaction rules, initializing Ray, preprocessing the molecules, collating
        the data, and returning the data and slices.
        :return: data (PyTorch geometric graphs) and slices.
        """

        with open(self.reaction_rules_path, "rb") as inp:
            reaction_rules = pickle.load(inp)

        dataset = {}
        for rule_i, (_, reactions_ids) in enumerate(reaction_rules):
            for reaction_id in reactions_ids:
                dataset[reaction_id] = rule_i
        del reaction_rules
        dataset = dict(sorted(dataset.items()))

        processed_data = []
        with RDFRead(self.reactions_path, indexable=True) as inp:
            inp.reset_index()
            for reaction_id, rule_id in tqdm(dataset.items()):
                reaction: ReactionContainer = inp[reaction_id]
                molecule = unite_molecules(reaction.products)
                pyg_graph = mol_to_pyg(molecule)
                if pyg_graph is not None:
                    pyg_graph.y_rules = torch.tensor([rule_id], dtype=torch.long)
                    processed_data.append(pyg_graph)

        data, slices = self.collate(processed_data)
        if self.output_path:
            makedirs(os.path.dirname(self.output_path))
            torch.save((data, slices), self.output_path)

        return data, slices


class FilteringPolicyDataset(InMemoryDataset):
    """
    Policy network dataset
    """

    def __init__(self, molecules_path, reaction_rules_path, output_path, num_cpus=1):
        """
        Initializes a policy network dataset object.

        :param molecules_path: The path to the file containing the molecules data
        :param reaction_rules_path: The path to the file containing the reaction rules.
        :param output_path: The output path is the location where policy network dataset will be stored.
        :param num_cpus: Specifies the number of CPUs to be used for the data preparation.
        """
        super().__init__(None, None, None)

        self.molecules_path = molecules_path
        self.reaction_rules_path = reaction_rules_path
        self.output_path = output_path
        self.num_cpus = num_cpus
        self.batch_prep_size = 10

        if output_path and os.path.exists(output_path):
            self.data, self.slices = torch.load(self.output_path)
        else:
            self.data, self.slices = self.prepare_data()

    @property
    def num_classes(self) -> int:
        return self._data.y_rules.shape[1]

    def prepare_data(self):
        """
        The function prepares data by loading reaction rules, initializing Ray, preprocessing the molecules, collating
        the data, and returning the data and slices.
        :return: data (PyTorch geometric graphs) and slices.
        """

        ray.init(num_cpus=self.num_cpus, ignore_reinit_error=True)
        reaction_rules = load_reaction_rules(self.reaction_rules_path)
        reaction_rules_ids = ray.put(reaction_rules)

        to_process = Queue(maxsize=self.batch_prep_size * self.num_cpus)
        processed_data = []
        results_ids = [preprocess_filtering_policy_molecules.remote(to_process, reaction_rules_ids) for _ in range(self.num_cpus)]

        with open(self.molecules_path, "r") as inp_data:
            for molecule in tqdm(inp_data.read().splitlines()):
                to_process.put(molecule)

        results = [graph for res in ray.get(results_ids) if res for graph in res]
        processed_data.extend(results)

        ray.shutdown()

        for pyg in processed_data:
            pyg.y_rules = pyg.y_rules.to_dense()
            pyg.y_priority = pyg.y_priority.to_dense()

        data, slices = self.collate(processed_data)
        if self.output_path:
            makedirs(os.path.dirname(self.output_path))
            torch.save((data, slices), self.output_path)

        return data, slices

    def prepare_data_no_ray(self):
        #######
        # /!\ Possible alternatives to ray, has to be checked once pytorch updated
        #######

        global reaction_rules
        reaction_rules = load_reaction_rules(self.reaction_rules_path)

        with Manager() as m, Pool() as p:
            to_process = m.Queue()

            processed_data = []
            # print(f'{len(mols_batches)} batches were created with {len(mols_batches[0])} molecules each')
            mols_batch = []
            with open(self.molecules_path, "r") as inp_data:
                for molecule in tqdm(inp_data.read().splitlines()):
                    mols_batch.append(molecule)
                    if len(mols_batch) == self.batch_prep_size:  # * self.num_cpus:
                        for mol in mols_batch:
                            to_process.put(mol)
                        mols_batch = []
                        workers_results = m.list()

                        workers = [p.apply_async(preprocess_filtering_policy_molecules, (to_process, workers_results)) for _ in range(40)]
                        print([res.get() for res in workers])
                        # # for i in range(40):
                        # #     w = Process(target=preprocess_policy_molecules, args=(to_process, workers_results))
                        # #     w.start()
                        # #     workers.append(w)
                        # # for w in workers:
                        # #     w.join()
                        # results = [graph for res in ray.get(results_ids) if res for graph in res]
                        # processed_data.extend(list(workers_results))
                        # mols_batch = []

        for pyg in processed_data:
            pyg.y_rules = pyg.y_rules.to_dense()
            pyg.y_priority = pyg.y_priority.to_dense()

        data, slices = self.collate(processed_data)
        if self.output_path:
            makedirs(os.path.dirname(self.output_path))
            torch.save((data, slices), self.output_path)

        return data, slices


def reaction_rules_appliance(molecule, reaction_rules):
    """
    The function applies each rule from the list of reaction rules to a given molecule and returns the indexes of
    the successfully applied regular rules and the indexes of the prioritized rules.

    :param molecule: The given molecule
    :param reaction_rules: The list of reaction rules
    :return: two lists: indexes of successfully applied regular and priority reaction rules.
    """

    applied_rules, priority_rules = [], []
    for i, rule in enumerate(reaction_rules):

        rule_applied = False
        rule_prioritized = False

        try:
            tmp = [molecule.copy()]
            for reaction in rule(tmp):
                for prod in reaction.products:

                    prod.kekule()
                    if prod.check_valence():
                        break
                    else:
                        rule_applied = True

                    # check priority rules
                    if len(reaction.products) > 1:
                        # check coupling retro manual
                        if all(len(mol) > 6 for mol in reaction.products):
                            if sum(len(mol) for mol in reaction.products) - len(reaction.reactants[0]) < 6:
                                rule_prioritized = True
                    else:
                        # check cyclization retro manual
                        if sum(len(mol.sssr) for mol in reaction.products) < sum(
                                len(mol.sssr) for mol in reaction.reactants):
                            rule_prioritized = True
            #
            if rule_applied:
                applied_rules.append(i)
                #
                if rule_prioritized:
                    priority_rules.append(i)

        except:
            continue

    return applied_rules, priority_rules


@ray.remote
def preprocess_filtering_policy_molecules(to_process: Queue, reaction_rules: List[Reactor]):
    """
    The function preprocesses a list of molecules by applying reaction rules and converting molecules into PyTorch
    geometric graphs. Successfully applied rules are converted to binary vectors for policy network training.

    :param to_process: The queue containing SMILES of molecules to be converted to the training data.
    :type to_process: Queue
    :param reaction_rules: The list of reaction rules.
    :type reaction_rules: List[Reactor]
    :return: a list of PyGraph objects.
    """

    pyg_graphs = []
    while True:
        try:
            molecule = smiles(to_process.get(timeout=30))
            if not isinstance(molecule, MoleculeContainer):
                continue

            # reaction reaction_rules application
            applied_rules, priority_rules = reaction_rules_appliance(molecule, reaction_rules)
            y_rules = torch.sparse_coo_tensor([applied_rules], torch.ones(len(applied_rules)),
                                              (len(reaction_rules),), dtype=torch.uint8)
            y_priority = torch.sparse_coo_tensor([priority_rules], torch.ones(len(priority_rules)),
                                                 (len(reaction_rules),), dtype=torch.uint8)

            y_rules = torch.unsqueeze(y_rules, 0)
            y_priority = torch.unsqueeze(y_priority, 0)

            pyg_graph = mol_to_pyg(molecule)
            if not pyg_graph:
                continue
            pyg_graph.y_rules = y_rules
            pyg_graph.y_priority = y_priority
            pyg_graphs.append(pyg_graph)
        except Empty:
            break
    return pyg_graphs


def preprocess_policy_molecules_no_ray(to_process: Queue, workers_results):
    #######
    # /!\ Possible alternatives to ray, has to be checked once pytorch updated
    #######

    pyg_graphs = []
    while True:
        try:
            molecule_str = to_process.get(timeout=1)
            molecule = smiles(molecule_str)
            if not isinstance(molecule, MoleculeContainer):
                continue

            # reaction reaction_rules application
            applied_rules, priority_rules = reaction_rules_appliance(molecule, reaction_rules)
            y_rules = torch.sparse_coo_tensor([applied_rules], torch.ones(len(applied_rules)), (len(reaction_rules),),
                                              dtype=torch.uint8)
            y_priority = torch.sparse_coo_tensor([priority_rules], torch.ones(len(priority_rules)),
                                                 (len(reaction_rules),), dtype=torch.uint8)

            y_rules = torch.unsqueeze(y_rules, 0)
            y_priority = torch.unsqueeze(y_priority, 0)

            pyg_graph = mol_to_pyg(molecule)
            if pyg_graph:
                pyg_graph.y_rules = y_rules
                pyg_graph.y_priority = y_priority
            else:
                continue

            pyg_graphs.append(pyg_graph)
        except Empty:
            break
    workers_results.extend(pyg_graphs) #[pyg for pyg in pyg_graphs if pyg])
    return pyg_graphs


def atom_to_vector(atom):
    """
    Given an atom, return a vector of length 8 with the following information:

    1. Atomic number
    2. Period
    3. Group
    4. Number of electrons + atom's charge
    5. Shell
    6. Total number of hydrogens
    7. Whether the atom is in a ring
    8. Number of neighbors

    :param atom: the atom object
    :return: The vector of the atom.
    """
    vector = torch.zeros(8, dtype=torch.uint8)
    period, group, shell, electrons = MENDEL_INFO[atom.atomic_symbol]
    vector[0] = atom.atomic_number
    vector[1] = period
    vector[2] = group
    vector[3] = electrons + atom.charge
    vector[4] = shell
    vector[5] = atom.total_hydrogens
    vector[6] = int(atom.in_ring)
    vector[7] = atom.neighbors
    return vector


def bonds_to_vector(molecule: MoleculeContainer, atom_ind: int):
    """
    The function takes a molecule and an atom index as input, and returns a vector representing the bond
    orders of the atom's bonds.

    :param molecule: The given molecule
    :type molecule: MoleculeContainer
    :param atom_ind: The the index of the atom in the molecule for which we want to calculate the bond vector.
    :type atom_ind: int
    :return: a torch tensor of size 3, with each element representing the order of bonds connected to the atom
    with the given index in the molecule.
    """
    vector = torch.zeros(3, dtype=torch.uint8)
    for b_order in molecule._bonds[atom_ind].values():
        vector[int(b_order) - 1] += 1
    return vector


def mol_to_matrix(molecule: MoleculeContainer):
    """
    Given a target, it returns a vector of shape (max_atoms, 12) where each row is an atom and each
    column is a feature.

    :param molecule: The target to be converted to a vector
    :type molecule: MoleculeContainer
    :return: The atoms_vectors array
    """

    atoms_vectors = torch.zeros((len(molecule), 11), dtype=torch.uint8)
    for n, atom in molecule.atoms():
        atoms_vectors[n - 1][:8] = atom_to_vector(atom)
    for n, _ in molecule.atoms():
        atoms_vectors[n - 1][8:] = bonds_to_vector(molecule, n)

    return atoms_vectors


def mol_to_pyg(molecule: MoleculeContainer):
    """
    It takes a list of molecules and returns a list of PyTorch Geometric graphs,
    a one-hot encoded vectors of the atoms, and a matrices of the bonds.

    :param molecule: The molecule to be converted to PyTorch Geometric graph.
    :return: A list of pyg graphs
    """

    molecule = molecule.copy()
    try:
        molecule.canonicalize()
        molecule.kekule()
        if molecule.check_valence():
            return None
    except InvalidAromaticRing:
        return None

    # remapping target for torch_geometric because
    # it is necessary that the elements in edge_index only hold nodes_idx in the range { 0, ..., num_nodes - 1}
    new_mappings = {n: i for i, (n, _) in enumerate(molecule.atoms(), 1)}
    molecule.remap(new_mappings)

    # get edge indexes from target mapping
    edge_index = []
    for atom, neighbour, bond in molecule.bonds():
        edge_index.append([atom - 1, neighbour - 1])
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    #
    x = mol_to_matrix(molecule)

    mol_pyg_graph = Data(x=x, edge_index=edge_index.t().contiguous())
    mol_pyg_graph = ToUndirected()(mol_pyg_graph)

    assert mol_pyg_graph.is_undirected()
    return mol_pyg_graph


MENDEL_INFO = {"Ag": (5, 11, 1, 1), "Al": (3, 13, 2, 1), "Ar": (3, 18, 2, 6), "As": (4, 15, 2, 3), "B": (2, 13, 2, 1),
    "Ba": (6, 2, 1, 2), "Bi": (6, 15, 2, 3), "Br": (4, 17, 2, 5), "C": (2, 14, 2, 2), "Ca": (4, 2, 1, 2),
    "Ce": (6, None, 1, 2), "Cl": (3, 17, 2, 5), "Cr": (4, 6, 1, 1), "Cs": (6, 1, 1, 1), "Cu": (4, 11, 1, 1),
    "Dy": (6, None, 1, 2), "Er": (6, None, 1, 2), "F": (2, 17, 2, 5), "Fe": (4, 8, 1, 2), "Ga": (4, 13, 2, 1),
    "Gd": (6, None, 1, 2), "Ge": (4, 14, 2, 2), "Hg": (6, 12, 1, 2), "I": (5, 17, 2, 5), "In": (5, 13, 2, 1),
    "K": (4, 1, 1, 1), "La": (6, 3, 1, 2), "Li": (2, 1, 1, 1), "Mg": (3, 2, 1, 2), "Mn": (4, 7, 1, 2),
    "N": (2, 15, 2, 3), "Na": (3, 1, 1, 1), "Nd": (6, None, 1, 2), "O": (2, 16, 2, 4), "P": (3, 15, 2, 3),
    "Pb": (6, 14, 2, 2), "Pd": (5, 10, 3, 10), "Pr": (6, None, 1, 2), "Rb": (5, 1, 1, 1), "S": (3, 16, 2, 4),
    "Sb": (5, 15, 2, 3), "Se": (4, 16, 2, 4), "Si": (3, 14, 2, 2), "Sm": (6, None, 1, 2), "Sn": (5, 14, 2, 2),
    "Sr": (5, 2, 1, 2), "Te": (5, 16, 2, 4), "Ti": (4, 4, 1, 2), "Tl": (6, 13, 2, 1), "Yb": (6, None, 1, 2),
    "Zn": (4, 12, 1, 2), }
