Synto (合成道) - the way of chemical synthesis
========
Synto is inspired by the Japanese term *Gouseido* (合成道), which roughly translates to "the way of synthesis".
This repository is a toolbox for chemical synthesis planning based on Monte-Carlo Tree Search (MCTS)
with various implementations of policy and value functions.

Installation
------------

Important: all versions require **python from 3.8 or up to 3.10**!

GPU version (Linux)
^^^^^^^^^^^
It requires only poetry 1.3.2. To install poetry, follow the instructions on
https://python-poetry.org/docs/#installation

For example, on Ubuntu 20.04 we can install miniconda in which we will install poetry with the following commands:

.. code-block:: bash

    # Install miniconda
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh

    # Install poetry
    conda create -n Synto_env -c conda-forge "poetry=1.3.2" "python=3.10" -y
    conda activate Synto_env

    # Install Synto
    git clone https://git.unistra.fr/isida_gtmtoolkit/Synto.git
    # or the github mirror https://github.com/tagirshin/Synto.git

    # Navigate to the Synto folder and run the following command:
    cd Synto/
    poetry install --with cpu

If Poetry fails with error, possible solution is to update bashrc file with the following command:

.. code-block:: bash

    echo 'export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring' >> ~/.bashrc
    exec "bash"

Optional
^^^^^^^^^^^
You can install environment in your jupyter kernel

.. code-block:: bash

    python -m ipykernel install --user --name Synto_env --display-name "Synto"

Usage
------------
Mostly the usage is optimized for in command line interface.
Here are some implemented commands:

* synto_planning
* synto_training
* synto_extract_rules
* synto_policy_training
* synto_self_learning

Each command has a description that can be called with ``command --help``

Run retrosynthetic planning
^^^^^^^^^^^
.. code-block:: bash

    synto_planning --targets="targets.txt" --config="planning_config.yaml" --results_root="synto_results"