Synto
========
Synto (SYNthesis planning TOol) is a tool for chemical synthesis planning based on Monte-Carlo Tree Search (MCTS) with
various implementations of policy and value functions.

Synto combines Monte-Carlo Tree Search (MCTS) with graph neural networks for the prediction of reaction
rules and synthesizability of intermediate products. Synto can be directly used for retrosynthesis planning
with pre-trained policy/value neural networks (planning mode) or can be fine-tuned to the custom data
using an automated end-to-end training pipeline (training mode).

Training pipeline in Synto includes several steps:

1)	Reaction rules extraction
2)	Policy network training
3)	Value network training


.. image:: png/workflow.png


.. toctree::
    :hidden:

    installation
    interfaces
    planning
    training
    configuration
