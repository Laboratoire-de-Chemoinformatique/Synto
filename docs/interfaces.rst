Interfaces
================

Command line interface
---------------------------

Synto provides with CLI commands for training and planning modes.

    For retrosynthesis planning:

    * synto_planning_data
    * synto_building_blocks
    * synto_planning

    For retrosynthetic models training:

    * synto_training_data
    * synto_building_blocks
    * synto_extract_reaction_rules
    * synto_policy_training
    * synto_self_tuning

    For full end-to-end training mode (combine all commands for training mode):

    * synto_training


Python interface
---------------------------

Python interface for planning

.. code-block:: python

    from Synto.mcts import Tree
    from Synto.utils.config import read_planning_config
    from CGRtools import smiles

    # set target
    target = 'C1=CC=NC(CNC(=O)C2=C3C(=NC(N)=N2)C(OC)=CC=C3)=N1'
    target = smiles(target)
    target.canonicalize()
    target.clean2d()

    # read config file
    config_path = 'planning_config.yaml'
    config = read_planning_config(config_path)

    # run search
    tree = Tree(target=target, config=config)
    _ = list(tree)
