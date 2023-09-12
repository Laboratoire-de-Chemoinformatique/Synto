"""
Module containing functions for loading trained policy and value networks
"""

from torch import device


def load_value_net(model_class, config):
    """
     Loads a model from an external path or an internal path

     :param config:
     :param model_class: The model class you want to load
     :type model_class: pl.LightningModule
     model will be loaded from the external path
     """

    map_location = device("cpu")
    return model_class.load_from_checkpoint(config["ValueNetwork"]["weights_path"], map_location)


# def load_policy_net(model_class, config):
#     """
#     Loads a model from an external path or an internal path
#
#     :param config:
#     :param model_class: The model class you want to load
#     :type model_class: pl.LightningModule
#     model will be loaded from the external path
#     """
#
#     map_location = device("cpu")
#     return model_class.load_from_checkpoint(config["PolicyNetwork"]["weights_path"], map_location)

def load_policy_net(model_class, config):
    """
    Loads a model from an external path or an internal path

    :param config:
    :param model_class: The model class you want to load
    :type model_class: pl.LightningModule
    model will be loaded from the external path
    """

    map_location = device("cpu")
    return model_class.load_from_checkpoint(config["PolicyNetwork"]["weights_path"], map_location,
                                            n_rules=12278, vector_dim=512, batch_size=1)
