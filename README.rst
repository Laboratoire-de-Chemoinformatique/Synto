GSLRetro
========
Graph-based Self-Learning Retrosynthesis

Installation
------------

Important: all version require **python from 3.8 or up to 3.10**!

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
    conda create -n gslretro_env -c conda-forge "poetry=1.3.2" "python=3.10" -y
    conda activate gslretro_env

    # Install GSLRetro
    git clone https://git.unistra.fr/isida_gtmtoolkit/gslretro.git
    # or the github mirror https://github.com/tagirshin/GSLRetro.git
    cd gslretro/
    poetry install --with cpu

If Poetry fails with error, possible solition is to update bashrc file with the following command:

.. code-block:: bash

    echo 'export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring' >> ~/.bashrc
    exec "bash"

Then navigate to the GSLRetro folder and run the following command:

.. code-block:: bash

    conda activate gslretro_env
    poetry install --with cpu

GPU version
^^^^^^^^^^^


Optional
^^^^^^^^^^^
you can install enviroment in your jupyter kernel

.. code-block:: bash

    python -m ipykernel install --user --name gslretro_env --display-name "gslretro"

Usage
------------
Mostly the usage is optimized to for in command line interface.
Here are some implemented commands:

* gslretro_default_config
* gslretro_search
* gslretro_simulate
* gslretro_tune_vn
* gslretro_self_learning
* gslretro_micro_self_learning
* gslretro_policy_train

Each command has a description that can be called with ``command --help``

Example commands:
::
    gslretro_search --targets test.sdf --config config.yaml --stats_name "test.csv" --retropaths_files_name test
    gslretro_self_learning --experiment_root test --targets_set training.sdf --config config.yaml --num_simulations 5 --batch_size 500 --logging_file test.log
    gslretro_policy_train --config /path/to/config