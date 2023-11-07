Configuration
===========================

Configuration

Planning configuration
------------
Planning configuration yaml file

.. code-block:: yaml

    General:
      reaction_rules_path: Synto_training/reaction_rules/reaction_rules.pickle
      building_blocks_path: Synto_training/building_blocks/building_blocks.pickle
    Tree:
      ucb_type: UCT  # UCB type
      c_usb: 0.1
      max_depth: 6  # maximum depth of tree
      max_iterations: 10  # maximum number of iterations
      max_time: 120  # search time limit
      max_tree_size: 1000000  # maximum number of nodes
      verbose: false  # tree search progress bar
      evaluation_mode: gcn  # node evaluation type: rollout or gcn
      evaluation_agg: max
      backprop_type: muzero  # value backpropagation type
      init_new_node_value: null  # default initial value for new nodes
    PolicyNetwork:
      weights_path: Synto_training/policy_network/weights/policy_network.ckpt
      priority_rules_fraction: 0.5
      top_rules: 50
      rule_prob_threshold: 0.0
    ValueNetwork:
      weights_path: Synto_training/value_network/weights/value_network.ckpt


Training configuration
------------
Training configuration yaml file

.. code-block:: yaml

    General:
      results_root: Synto_tmp
      building_blocks_path: Synto_training/building_blocks/building_blocks.pickle
      num_cpus: 10
      num_gpus: 1
    Tree:
      ucb_type: UCT
      c_usb: 0.1
      max_depth: 6
      max_iterations: 50
      max_time: 120
      max_tree_size: 1000000
      verbose: false
      evaluation_mode: gcn
      evaluation_agg: max
      backprop_type: muzero
      init_new_node_value: null
      epsilon: 0.0
    ReactionRules:
      results_root: Synto_training/reaction_rules
      reaction_data_path: Synto_training/reaction_data/reaction_data.rdf
      reaction_rules_path: Synto_training/reaction_rules/reaction_rules.pickle
    SelfTuning:
      results_root: Synto_training/value_network
      dataset_path: Synto_training/value_molecules/value_molecules.sdf
      num_simulations: 1
      batch_size: 5
    PolicyNetwork:
      results_root: Synto_training/policy_network
      dataset_path: Synto_training/policy_molecules/policy_molecules.sdf
      datamodule_path: Synto_training/policy_network/policy_dataset.pt
      weights_path: Synto_training/policy_network/policy_network.ckpt
      top_rules: 50
      priority_rules_fraction: 0.5
      rule_prob_threshold: 0.0
      num_conv_layers: 5
      vector_dim: 512
      dropout: 0.4
      learning_rate: 0.0005
      num_epoch: 100
      batch_size: 500
    ValueNetwork:
      results_root: Synto_training/value_network
      weights_path: Synto_training/value_network/value_network.ckpt
      num_conv_layers: 5
      vector_dim: 512
      dropout: 0.4
      learning_rate: 0.0005
      num_epoch: 30
      batch_size: 500


