Synto - SYNthesis planning TOol
========
Synto is a tool for chemical synthesis planning based on Monte-Carlo Tree Search (MCTS)
with various implementations of policy and value functions.


Installation
------------

Important: all versions require **python from 3.8 or up to 3.10**!

GPU version (Linux)
^^^^^^^^^^^
It requires only poetry 1.3.2. To install poetry, follow the instructions on
https://python-poetry.org/docs/#installation

For example, on Ubuntu we can install miniconda in which we will install poetry with the following commands:

.. code-block:: bash

    # install miniconda
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh

    # install poetry
    conda create -n synto -c conda-forge "poetry=1.3.2" "python=3.10" -y
    conda activate synto

    # install Synto
    git clone https://github.com/Laboratoire-de-Chemoinformatique/Synto.git

    # navigate to the Synto folder and run the following command:
    cd Synto/
    poetry install --with cpu

If Poetry fails with error, a possible solution is to update the bashrc file with the following command:

.. code-block:: bash

    echo 'export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring' >> ~/.bashrc
    exec "bash"

Optional
^^^^^^^^^^^
One can install the environment in your jupyter kernel

.. code-block:: bash

    python -m ipykernel install --user --name synto --display-name "synto"

Usage
------------
Mostly the usage is optimized for in command line interface.
Here are some implemented commands:

* synto_planning
* synto_training
* synto_extract_rules
* synto_policy_training
* synto_self_tuning

Each command has a description that can be called with ``command --help``

Run retrosynthetic planning
^^^^^^^^^^^
.. code-block:: bash

    synto_planning_data
    synto_building_blocks --input="synto_planning_data/building_blocks.txt" --output="synto_planning_data/building_blocks.txt" # skip for loaded data
    synto_planning --targets="targets.txt" --config="planning_config.yaml" --results_root="synto_results"

Run training from scratch
^^^^^^^^^^^
.. code-block:: bash

    synto_training_data
    synto_building_blocks --input="synto_training_data/building_blocks.txt" --output="synto_training_data/building_blocks.txt" # skip for loaded data
    synto_training --config="training_config.yaml"


Documentation
-----------

The the detailed documentation can be found `here <https://laboratoire-de-chemoinformatique.github.io/Synto/>`_