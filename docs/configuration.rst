Configuration
===========================

Synto can be configured with advanced versions of configuration files that provide more flexibility in retrosynthesis
planning and training of retrosynthetic models

Planning configuration
---------------------------
Planning configuration yaml file

.. code-block:: yaml

    General:
      reaction_rules_path: synto_planning_data/reaction_rules.pickle
      building_blocks_path: synto_planning_data/building_blocks.txt
    Tree:
      ucb_type: uct
      c_usb: 0.1
      max_depth: 6
      max_iterations: 50
      max_time: 120
      max_tree_size: 100000
      verbose: true
      evaluation_mode: gcn
      evaluation_agg: max
      backprop_type: muzero
      init_new_node_value: null
    PolicyNetwork:
      weights_path: synto_planning_data/policy_network.ckpt
      priority_rules_fraction: 0.5
      top_rules: 50
      rule_prob_threshold: 0.0
    ValueNetwork:
      weights_path: synto_planning_data/value_network.ckpt


Training configuration
---------------------------
Training configuration yaml file

.. code-block:: yaml

    General:
      results_root: synto_training_data
      reaction_rules_path: synto_training_data/reaction_rules/reaction_rules.pickle
      building_blocks_path: synto_training_data/building_blocks/building_blocks.pickle
      num_cpus: 5
      num_gpus: 1
    Tree:
      ucb_type: uct
      c_usb: 0.1
      max_depth: 6
      max_iterations: 15
      max_time: 600
      max_tree_size: 1000000
      verbose: false
      evaluation_mode: gcn
      evaluation_agg: max
      backprop_type: muzero
      init_new_node_value: null
      epsilon: 0.0
    ReactionRules:
      results_root: synto_training_data/reaction_rules
      reaction_data_path: synto_training_data/reaction_data/reaction_data.rdf
      min_popularity: 5
    SelfTuning:
      results_root: synto_training_data/value_network
      dataset_path: synto_training_data/value_molecules/value_molecules.smi
      num_simulations: 1
      batch_size: 5
    PolicyNetwork:
      results_root: synto_training_data/policy_network
      dataset_path: synto_training_data/policy_molecules/policy_molecules.smi
      datamodule_path: synto_training_data/policy_network/policy_dataset.pt
      weights_path: synto_training_data/policy_network/policy_network.ckpt
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
      results_root: synto_training_data/value_network
      weights_path: synto_training_data/value_network/value_network.ckpt
      num_conv_layers: 5
      vector_dim: 512
      dropout: 0.4
      learning_rate: 0.0005
      num_epoch: 100
      batch_size: 500


