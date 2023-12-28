"""
Module containing functions for loading trained policy and value networks
"""

from torch import device


def load_value_net(model_class, value_network_path):
    """
     Loads a model from an external path or an internal path

     :param value_network_path:
     :param model_class: The model class you want to load
     :type model_class: pl.LightningModule
     model will be loaded from the external path
     """

    map_location = device("cpu")
    return model_class.load_from_checkpoint(value_network_path, map_location)


def load_policy_net(model_class, policy_network_path):
    """
    Loads a model from an external path or an internal path

    :param policy_network_path:
    :param model_class: The model class you want to load
    :type model_class: pl.LightningModule
    model will be loaded from the external path
    """

    map_location = device("cpu")
    # return model_class.load_from_checkpoint(policy_network_path, map_location, n_rules=n_rules,
    #                                         vector_dim=vector_dim, batch_size=1)

    return model_class.load_from_checkpoint(policy_network_path, map_location, batch_size=1)
