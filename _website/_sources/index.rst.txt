Synto (合成道) - the way of chemical synthesis
========
Synto is inspired by the Japanese term *Gouseido* (合成道), which roughly translates to "the way of synthesis".
This repository is a toolbox for chemical synthesis planning based on Monte-Carlo Tree Search (MCTS)
with various implementations of policy and value functions.


Introduction
---------------------------

Synto learns to generate retrosynthetic paths from real chemical reactions stored in big databases.
To extract this knowledge from the reaction databases we need to apply machine learning algorithms, especially deep
neural networks. To generate the retrosynthesis path for the given molecule the tool needs to know which reaction to
apply to current substrates to get the intermediate product and how to establish the correct sequence of these reactions
(synthetic path) leading to the target molecule.

For the prediction of reactions to be applied we use the policy network trained on the real reactions. For the building
of the retrosynthetic paths from elementary reactions Synto uses value network for the prediction of the
synthesizability of the current intermediate substrate and the MCTS algorithm for navigating in the space of multiple
possible retrosynthetic paths.

Value network predicts the “probability” to reach the building blocks starting from the current intermediate product
(including target molecule). Then these predictions are used in calculation of MCTS statistics navigating the search.
To train the neural network we use self-learning technology, which includes alternating training and MCTS planning steps,
so that the value network learns from the previous planning experience.


.. image:: png/workflow.png


The workflow for training the Synto tool includes several steps. Reaction data are first cleaned and curated
automatically by in-house scripts. Then these data are used to train the policy network. This policy network and raw
(untrained) value network are used in MCTS planning for several target molecules. Planning results are then collected
and used in training the value network.

.. toctree::
    :hidden:

    quick_start
    installation
    interfaces
    configuration
    networks_training
