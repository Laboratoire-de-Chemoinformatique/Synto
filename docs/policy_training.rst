Policy network training
===========================

The policy neural network is responsible for predicting reaction rules that can be applied to the current retron to generate
new retrons (node expansion stage). The policy neural network is based on a graph convolutional neural network where the
input molecule is transformed into a matrix where each atom is described by a vector of descriptors (atomic properties from
the periodic table and chemical bonds characteristics connecting each atom to another atoms).

Each input molecule corresponds to a binary vector, where 1 correspond to the reaction rules applicable to that molecule.

For training the neural network, the following command can be used:

.. code-block:: bash

    gslretro_policy_training --config training_config.yaml

The preparation of a policy neural network is divided into two stages - preparation of the training set and neural network training.

The training set is prepared by applying the available reaction rules to a set of molecules
(in general, it can be any set of molecules). As a result, a training set is obtained in which each molecule is represented
by a matrix of atom descriptors and a binary vector where 1 positions correspond  to the numbers of reaction rules
successfully applied to this molecule.
