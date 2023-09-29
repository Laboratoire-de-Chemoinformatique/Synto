Installation
===========================

The Synto tool can be used in two scenarios:

    * retrosynthesis planning for the set of target molecules using pre-trained policy and value networks.

    * retrosynthesis planning with preliminary training of the neural networks on the reaction and building blocks database.


Package installation
---------------------------

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