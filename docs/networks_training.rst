Networks training
===========================


Policy network training
---------------------------

The policy neural network is responsible for predicting reaction rules that can be applied to the current retron to generate
new retrons (node expansion stage). The policy neural network is based on a graph convolutional neural network where the
input molecule is transformed into a matrix where each atom is described by a vector of descriptors (atomic properties from
the periodic table and chemical bonds characteristics connecting each atom to another atoms).

Each input molecule corresponds to a binary vector, where 1 correspond to the reaction rules applicable to that molecule.

For training the neural network, the following command can be used:

.. code-block:: bash

    Synto_policy_training --config training_config.yaml

The preparation of a policy neural network is divided into two stages - preparation of the training set and neural network training.

The training set is prepared by applying the available reaction rules to a set of molecules
(in general, it can be any set of molecules). As a result, a training set is obtained in which each molecule is represented
by a matrix of atom descriptors and a binary vector where 1 positions correspond  to the numbers of reaction rules
successfully applied to this molecule.

Value network training
---------------------------

The  Synto tool.

The value neural network is trained using a self-learning approach, which implies alternating retrosynthetic
planning for the target molecules and then training the value neural network on the data successful and unsuccessful
expansions extracted from the tree search results. A neural network can be trained using the following command

.. code-block:: bash

    Synto_policy_training --config training_config.yaml

This command requires a set of input molecules for planning stage and preparing a training set for the value
network tuning.

There are two sections in the configuration file to prepare the neural network.

A section for setting neural network architecture and some network hyperparameters:

.. code-block:: yaml

    ValueNetwork:
      results_root: Synto_training/value_network
      weights_path: Synto_training/value_network/value_network.ckpt
      num_conv_layers: 5
      vector_dim: 512
      dropout: 0.4
      learning_rate: 0.0005
      num_epoch: 100
      batch_size: 500

And a section for setting self-learning parameters:

.. code-block:: yaml

    SelfLearning:
      results_root: Synto_training/value_network
      dataset_path: Synto_training/value_molecules/value_molecules.sdf
      num_simulations: 1
      batch_size: 5
      balance_positive: false


``num_simulations`` specifies the number of simulations (planning and learning steps) for a given set of target molecules.
``batch_size`` specifies the number of target molecules for planning and following training of the value neural network.
For example, if the ``batch_size: 100``, then the value neural network will be tuned every 100 planning runs.



