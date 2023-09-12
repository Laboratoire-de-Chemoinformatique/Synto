Value network training
===========================

The  GSLRetro tool.

The value neural network is trained using a self-learning approach, which implies alternating retrosynthetic
planning for the target molecules and then training the value neural network on the data successful and unsuccessful
expansions extracted from the tree search results. A neural network can be trained using the following command

.. code-block:: bash

    gslretro_policy_training --config training_config.yaml

This command requires a set of input molecules for planning stage and preparing a training set for the value
network tuning.

There are two sections in the configuration file to prepare the neural network.

A section for setting neural network architecture and some network hyperparameters:

.. code-block:: yaml

    ValueNetwork:
      results_root: gslretro_training/value_network
      weights_path: gslretro_training/value_network/value_network.ckpt
      num_conv_layers: 5
      vector_dim: 512
      dropout: 0.4
      learning_rate: 0.0005
      num_epoch: 100
      batch_size: 500

And a section for setting self-learning parameters:

.. code-block:: yaml

    SelfLearning:
      results_root: gslretro_training/value_network
      dataset_path: gslretro_training/value_molecules/value_molecules.sdf
      num_simulations: 1
      batch_size: 5
      balance_positive: false


``num_simulations`` specifies the number of simulations (planning and learning steps) for a given set of target molecules.
``batch_size`` specifies the number of target molecules for planning and following training of the value neural network.
For example, if the ``batch_size: 100``, then the value neural network will be tuned every 100 planning runs.


