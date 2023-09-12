Synto (合成道) - the way of chemical synthesis
========
Synto is inspired by the Japanese term *Gouseido* (合成道), which roughly translates to "the way of synthesis".
This repository is a toolbox for chemical synthesis planning based on Monte-Carlo Tree Search (MCTS)
with various implementations of policy and value functions.

Installation
------------

Important: all versions require **python from 3.8 or up to 3.10**!

CPU version
^^^^^^^^^^^
The CPU version requires only poetry 1.3.2. To install poetry, follow the instructions on
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
    cd Synto/
    poetry install --with cpu

If Poetry fails with error, possible solition is to update bashrc file with the following command:

.. code-block:: bash

    echo 'export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring' >> ~/.bashrc
    exec "bash"

Then navigate to the Synto folder and run the following command:

.. code-block:: bash

    conda activate Synto_env
    poetry install --with cpu

GPU version
^^^^^^^^^^^


Optional
^^^^^^^^^^^
you can install enviroment in your jupyter kernel

.. code-block:: bash

    python -m ipykernel install --user --name Synto_env --display-name "Synto"

Usage
------------
Mostly the usage is optimized to for in command line interface.
Here are some implemented commands:

* Synto_default_config
* Synto_search
* Synto_simulate
* Synto_tune_vn
* Synto_self_learning
* Synto_micro_self_learning
* Synto_policy_train

Each command has a description that can be called with ``command --help``

Example commands:
::
    Synto_search --targets test.sdf --config config.yaml --stats_name "test.csv" --retropaths_files_name test
    Synto_self_learning --experiment_root test --targets_set training.sdf --config config.yaml --num_simulations 5 --batch_size 500 --logging_file test.log
    Synto_policy_train --config /path/to/config