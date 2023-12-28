"""
Module for the preparation and training of a policy network used in the expansion of nodes in Monte Carlo Tree Search (MCTS).
This module includes functions for creating training datasets and running the training process for the policy network.
"""

import warnings
from pathlib import Path

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import random_split
from torch_geometric.data.lightning import LightningDataset

from Synto.ml.networks.policy import PolicyNetwork, PolicyNetworkConfig
from Synto.ml.training.preprocessing import RankingPolicyDataset, FilteringPolicyDataset
from Synto.utils.logging import DisableLogger, HiddenPrints

warnings.filterwarnings("ignore")


def create_policy_dataset(
    reaction_rules_path: str,
    molecules_or_reactions_path: str,
    output_path: str,
    dataset_type: str = "filtering",
    batch_size: int = 100,
    num_cpus: int = 1,
    training_data_ratio: float = 0.8,
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
        if dataset_type == "filtering":
            full_dataset = FilteringPolicyDataset(
                reaction_rules_path=reaction_rules_path,
                molecules_path=molecules_or_reactions_path,
                output_path=output_path,
                num_cpus=num_cpus,
            )
        elif dataset_type == "ranking":
            full_dataset = RankingPolicyDataset(
                reaction_rules_path=reaction_rules_path,
                reactions_path=molecules_or_reactions_path,
                output_path=output_path,
            )
        else:
            raise ValueError("Invalid dataset type. Must be 'ranking' or 'filtering'.")

    train_size = int(training_data_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], torch.Generator().manual_seed(42)
    )
    print(
        f"Training set size: {len(train_dataset)}, validation set size: {len(val_dataset)}"
    )

    datamodule = LightningDataset(
        train_dataset,
        val_dataset,
        batch_size=batch_size,
        pin_memory=True,
        drop_last=True,
    )
    return datamodule


def run_policy_training(
    datamodule: LightningDataset,
    config: PolicyNetworkConfig,
    results_path: str,
    accelerator: str = "gpu",
    devices: list = [0],
    silent=True,
):
    """
    Trains a policy network using a given datamodule and training configuration.

    :param datamodule: A PyTorch Lightning `DataModule` class instance. It is responsible for
     loading, processing, and preparing the training data for the model.
    :param config: The dictionary that contains various configuration settings for the policy training process.
    :param results_path: Path to store the training results and logs.
    :param accelerator: The type of hardware accelerator to use for training (e.g., 'gpu', 'cpu').
                                     Defaults to "gpu".
    :param devices: A list of device indices to use for training. Defaults to [0].
    :param silent: If True (the default) all logging information will be not printed

    This function sets up the environment for training a policy network. It includes creating directories
    for storing logs and weights, initializing the network with the specified configuration, and setting up
    training callbacks like LearningRateMonitor and ModelCheckpoint. The Trainer from PyTorch Lightning is
    used to manage the training process. If 'silent' is set to True, the function suppresses the standard
    output and logging information during training.

    The function creates three subdirectories within the specified 'results_path':
        - 'logs/' for storing training logs.
        - 'weights/' for saving model checkpoints.
    """
    results_path = Path(results_path)
    results_path.mkdir(exist_ok=True)

    logs_path = results_path.joinpath("logs/")
    logs_path.mkdir(exist_ok=True)

    weights_path = results_path.joinpath("weights/")
    weights_path.mkdir(exist_ok=True)

    network = PolicyNetwork(
        vector_dim=config.vector_dim,
        n_rules=datamodule.train_dataset.dataset.num_classes,
        batch_size=config.batch_size,
        dropout=config.dropout,
        num_conv_layers=config.num_conv_layers,
        learning_rate=config.learning_rate,
        policy_type=config.policy_type,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    logger = CSVLogger(logs_path, name=results_path)

    checkpoint = ModelCheckpoint(
        dirpath=weights_path,
        filename='policy_network',
        monitor="val_loss",
        mode="min",
    )

    if silent:
        with DisableLogger(), HiddenPrints():
            trainer = Trainer(
                accelerator=accelerator,
                devices=devices,
                max_epochs=config.num_epoch,
                callbacks=[lr_monitor, checkpoint],
                logger=logger,
                gradient_clip_val=1.0,
                enable_progress_bar=False,
            )

            trainer.fit(network, datamodule)
    else:
        trainer = Trainer(
            accelerator=accelerator,
            devices=devices,
            max_epochs=config.num_epoch,
            callbacks=[lr_monitor, checkpoint],
            logger=logger,
            gradient_clip_val=1.0,
            enable_progress_bar=True,
        )

        trainer.fit(network, datamodule)

    ba = round(trainer.logged_metrics['train_balanced_accuracy_y_step'].item(), 3)
    print(f'Policy network balanced accuracy: {ba}')
