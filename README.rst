Syntool - Synthesis planning tool
========
Syntool is a tool for chemical synthesis planning based on Monte-Carlo Tree Search (MCTS)
with various implementations of policy and value functions.


Installation
------------

Important: all versions require **python from 3.8 and up to 3.10**!

Linux distributions
^^^^^^^^^^^

Dev: Installation with conda-lock
""""""

`conda-lock` is a tool used for creating deterministic environment specifications for conda environments. This is useful for ensuring consistent environments across different machines or at different times. To install `conda-lock`, follow these steps:

**1. Install conda-lock**

You need to have `conda` or `mamba` installed on your system to install `conda-lock`. If you have not installed `conda` yet, you can download it from `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ or `Anaconda <https://www.anaconda.com/products/individual>`_.

wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh


Once `conda` is installed, you can install `conda-lock` by running:

.. code-block:: bash

   conda install -c conda-forge -n base conda-lock

or, if you are using `mamba`:

.. code-block:: bash

   mamba install -c conda-forge -n base conda-lock

**2. Install the environment using the conda-lock file**


Once you have a `.conda-lock` file, you can create a conda environment that exactly matches the specifications in the lock file. To do this, use:

.. code-block:: bash

   conda-lock install -n synto --file conda-linux64-GPU-lock.yml

This command will read the `.conda-lock` file and create an environment with the exact package versions specified in the file.

.. note::
   Make sure that the `.conda-lock` file is in your current working directory or provide the path to the file when using the `conda-lock install` command.

Dev: Installation with poetry
""""""

It requires only poetry 1.3.2. To install poetry, follow the example below, or the instructions on
https://python-poetry.org/docs/#installation

For example, on Ubuntu we can install miniconda and set an environment in which we will install poetry with the following commands:

.. code-block:: bash

    # install miniconda
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh

    # install poetry
    conda create -n syntool -c conda-forge "poetry=1.3.2" "python=3.10" -y
    conda activate syntool

    # install Syntool
    git clone https://github.com/Laboratoire-de-Chemoinformatique/Syntool.git

    # navigate to the Syntool folder and run the following command:
    cd Syntool/
    poetry install --with cpu

If Poetry fails with error, a possible solution is to update the bashrc file with the following command:

.. code-block:: bash

    echo 'export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring' >> ~/.bashrc
    exec "bash"

Optional
^^^^^^^^^^^
After installation, one can add the Synto environment in their Jupyter platform:

.. code-block:: bash

    python -m ipykernel install --user --name synto --display-name "syntool"

Usage
------------
The usage is mostly optimized for the command line interface.
Here are some implemented commands:

* syntool_planning
* syntool_training

Each command has a description that can be called with ``command --help``

Run retrosynthetic planning
^^^^^^^^^^^
.. code-block:: bash

    syntool_planning_data
    syntool_planning --config="planning_config.yaml"

Run training from scratch
^^^^^^^^^^^
.. code-block:: bash

    syntool_training_data
    syntool_training --config="training_config.yaml"


Documentation
-----------

The detailed documentation can be found `here <https://laboratoire-de-chemoinformatique.github.io/Syntool/>`_

Tests
-----------

.. code-block:: bash

    syntool_training --config="configs/training_config.yaml"
    syntool_planning --config="configs/planning_config.yaml"
