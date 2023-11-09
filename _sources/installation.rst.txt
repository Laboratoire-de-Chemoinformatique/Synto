Installation
===========================

Important: all versions require **python from 3.8 or up to 3.10**!

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

    # Navigate to the Synto folder and run the following command:
    cd Synto/
    poetry install --with cpu

If Poetry fails with error, a possible solution is to update the bashrc file with the following command:

.. code-block:: bash

    echo 'export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring' >> ~/.bashrc
    exec "bash"

One can install the environment in your jupyter kernel

.. code-block:: bash

    python -m ipykernel install --user --name synto --display-name "synto"