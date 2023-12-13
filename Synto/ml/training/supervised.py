"""
Module for the preparation and training of a policy network used in the expansion of nodes in Monte Carlo Tree Search (MCTS).
This module includes functions for creating training datasets and running the training process for the policy network.
"""

import os.path as osp
import warnings
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import random_split
from torch_geometric.data.lightning import LightningDataset

from Synto.ml.networks.policy import PolicyNetwork
from Synto.ml.training.preprocessing import RankingPolicyDataset, FilteringPolicyDataset
from Synto.utils.logging import DisableLogger, HiddenPrints

warnings.filterwarnings('ignore')


def create_policy_dataset(
        reaction_rules_path: str,
        molecules_or_reactions_path: str,
        output_path: str,
        dataset_type: str = 'filtering',
        batch_size: int = 100,
        num_cpus: int = 1,
        training_data_ratio: float = 0.8
):
    """
    Generic function to create a training dataset for a policy network.

    :param dataset_type: Type of the dataset to be created ('ranking' or 'filtering').
    :param reaction_rules_path: Path to the reaction rules file.
    :param molecules_or_reactions_path: Path to the molecules or reactions file.
    :param output_path: Path to store the processed dataset.
    :param batch_size: Size of each data batch.
    :param num_cpus: Number of CPUs to use for data processing.
    :param training_data_ratio: Ratio of training data to total data.
    :return: A `LightningDataset` object containing training and validation datasets.
    """
    with DisableLogger():
        if dataset_type == 'filtering':
            full_dataset = FilteringPolicyDataset(reaction_rules_path=reaction_rules_path,
                                                  molecules_path=molecules_or_reactions_path,
                                                  output_path=output_path,
                                                  num_cpus=num_cpus)
        elif dataset_type == 'ranking':
            full_dataset = RankingPolicyDataset(reaction_rules_path=reaction_rules_path,
                                                reactions_path=molecules_or_reactions_path,
                                                output_path=output_path)
        else:
            raise ValueError("Invalid dataset type. Must be 'ranking' or 'filtering'.")

    train_size = int(training_data_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        torch.Generator().manual_seed(42)
    )
    print(f'Training set size: {len(train_dataset)}, validation set size: {len(val_dataset)}')

    datamodule = LightningDataset(train_dataset, val_dataset, batch_size=batch_size, pin_memory=True, drop_last=True)
    return datamodule


def run_policy_training(
        datamodule: LightningDataset,
        config,
        results_path,
        silent=True
):
    """
    Trains a policy network using a given datamodule and training configuration.

    :param silent: If True (the default) all logging information will be not printed
    :param datamodule: The PyTorch Lightning `DataModule` class. It is responsible for loading and preparing the
                       training data for the model.
    :param config: The dictionary that contains various configuration settings for the policy training process.
    :param results_path: Path to store the training results and logs.
    """
    network = PolicyNetwork(
        vector_dim=config['PolicyNetwork']['vector_dim'],
        n_rules=datamodule.train_dataset.dataset.num_classes,
        batch_size=config['PolicyNetwork']['batch_size'],
        dropout=config['PolicyNetwork']['dropout'],
        num_conv_layers=config['PolicyNetwork']['num_conv_layers'],
        learning_rate=config['PolicyNetwork']['learning_rate'],
        mode=config['PolicyNetwork']['mode']
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    logger = CSVLogger(osp.join(results_path, 'logs'))

    checkpoint = ModelCheckpoint(
        dirpath=osp.join(results_path, 'weights'),
        filename='policy_network',
        monitor="val_loss",
        mode="min"
    )

    if silent:
        with DisableLogger(), HiddenPrints():
            trainer = Trainer(
                accelerator='gpu',
                devices=[0],
                max_epochs=config['PolicyNetwork']['num_epoch'],
                callbacks=[lr_monitor, checkpoint],
                logger=logger,
                gradient_clip_val=1.0,
                enable_progress_bar=False
            )

            trainer.fit(network, datamodule)
    else:
        trainer = Trainer(
            accelerator='gpu',
            devices=[0],
            max_epochs=config['PolicyNetwork']['num_epoch'],
            callbacks=[lr_monitor, checkpoint],
            logger=logger,
            gradient_clip_val=1.0,
            enable_progress_bar=True
        )

        trainer.fit(network, datamodule)
