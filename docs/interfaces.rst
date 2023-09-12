Interfaces
================

Graphical user interface
-------------------

You can use the pretrained GSLRetro via Graphical User Interface (GUI) available by this link.


Command line interface
-------------------

GSLRetro provides with CLI commands for training and planning modes.

    For planning mode:

    * gslretro_download_data
    * gslretro_default_planning_config
    * gslretro_tree_search

    For training mode:

    * gslretro_default_training_config
    * gslretro_extract_reaction_rules
    * gslretro_policy_training
    * gslretro_self_learning

    For full end-to-end training mode (combine all commands for training mode):

    * gslretro_default_training_config
    * gslretro_training


Python interface
-------------------

Python interface for planning

.. code-block:: python

    from GSLRetro.mcts import Tree
    from GSLRetro.utils.config import read_planning_config

    # set target
    target = 'C1=CC=NC(CNC(=O)C2=C3C(=NC(N)=N2)C(OC)=CC=C3)=N1'
    target = smiles(target)
    target.canonicalize()
    target.clean2d()

    # read config file
    config_path = 'planning_config.yaml'
    config = read_planning_config(config_path)

    # Run search
    tree = Tree(target=target, config=config)
    _ = list(tree)
