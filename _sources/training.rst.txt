Training
===========================

**Data**

The following data files are required for retrosynthetic models training with Synto:

`reaction_data.rdf` - dataset of reactions for extraction of reaction rules

`policy_molecules.sdf` - dataset of molecules for training policy neural network

`value_molecules.sdf` - dataset of targets for self-tuning of value neural network

`building_blocks.txt` - building blocks in SMILES format (.smi or .txt)

**Configuration**

Retrosynthetic models training is configured with the `config_training.yaml` configuration file:

.. code-block:: yaml

    General:
      results_root: synto_training_data
      reaction_rules_path: synto_training_data/reaction_rules/reaction_rules.pickle
      building_blocks_path: synto_training_data/building_blocks/building_blocks.pickle
      num_cpus: 10
      num_gpus: 1
    Tree:
      max_depth: 6
      max_iterations: 5
      evaluation_mode: gcn
      verbose: false
    ReactionRules:
      results_root: synto_training_data/reaction_rules
      reaction_data_path: synto_training_data/reaction_data/reaction_data.rdf
    SelfTuning:
      results_root: synto_training_data/value_network
      dataset_path: synto_training_data/value_molecules/value_molecules.sdf
    PolicyNetwork:
      results_root: synto_training_data/policy_network
      dataset_path: synto_training_data/policy_molecules/policy_molecules.sdf
      datamodule_path: synto_training_data/policy_network/policy_dataset.pt
      weights_path: synto_training_data/policy_network/policy_network.ckpt
    ValueNetwork:
      results_root: synto_training_data/value_network
      weights_path: synto_training_data/value_network/value_network.ckpt


**Full-pipeline execution**

For running retrosynthetic models training one needs three commands: (i) download training data
(ii) prepare building blocks (canonicalize, standardize) (iii) run training pipeline:

.. code-block:: bash

    synto_training_data
    synto_building_blocks --input="synto_training_data/building_blocks.txt" --output="synto_training_data/building_blocks.txt" # skip for loaded data
    synto_training --config="training_config.yaml"


**Step-by-step execution**

* synto_training_data
* synto_building_blocks
* synto_extract_rules
* synto_policy_training
* synto_self_tuning

**Run planning on trained networks**

Extracted reaction rules and trained policy and value networks can be used in retrosynthesis planning:

.. code-block:: yaml

    General:
      reaction_rules_path: synto_training_data/reaction_rules/reaction_rules.pickle
      building_blocks_path: synto_training_data/building_blocks/building_blocks.txt
    PolicyNetwork:
      weights_path: synto_training_data/policy_network/policy_network.ckpt
    ValueNetwork:
      weights_path: synto_training_data/value_network/value_network.ckpt
    Tree:
      max_depth: 9  # maximum depth of tree
      max_iterations: 100  # maximum number of iterations
      max_time: 600  # search time limit
      evaluation_mode: gcn
      verbose: True  # tree search progress bar


