Installation
============

TxCellNet runs with Python 3.11 (the version used in the paper) and PyTorch. Training and whole-section inference expect a
CUDA GPU (the paper used a single NVIDIA A800); the package, the smoke test and the quick-start tutorial also run on the
CPU.

Clone the repository and create an environment with ``conda``:

.. code-block:: bash

   git clone https://github.com/wrab12/TxCellNet
   cd TxCellNet
   conda create -n txcellnet python=3.11
   conda activate txcellnet

Install PyTorch and torchvision for your CUDA version (see the
`PyTorch installation guide <https://pytorch.org/get-started/locally/>`_; ``nvidia-smi`` shows the CUDA version). The
paper used PyTorch 2.5 with CUDA 12.1. The official NicheTrans model, one of the compared methods, imports
torchvision:

.. code-block:: bash

   pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

Install the dependencies (``requirements.txt`` also holds ``einops``, which NicheTrans imports) and the package:

.. code-block:: bash

   pip install -r requirements.txt
   pip install -e .

To run the tutorial notebooks, also install Jupyter (``pip install jupyterlab``).

Check the installation
----------------------

The smoke test runs from the repository root on small synthetic sections. It checks the molecule splatting, a few
training steps of stage 1, whole-section inference with genes outside the vocabulary and with a partial panel, stage-2
sampling, and that every cell mean of the refined image equals the stage-1 cell mean. It takes well under a minute on a
laptop CPU:

.. code-block:: bash

   python smoke_test.py

The last line is ``[smoke] ALL PASS``; the script exits with a non-zero status if any check fails.

Environments of the compared methods
------------------------------------

Only the comparisons of the paper need these. The compared methods run with their official code in two further
environments:

.. code-block:: bash

   # PyG environment (DGAT, SpaMosaic, spot-level evaluation): the main environment plus torch_geometric 2.6.1
   conda create -n txcellnet-pyg --clone txcellnet && conda activate txcellnet-pyg
   pip install -r envs/requirements-pyg.txt -f https://data.pyg.org/whl/torch-2.5.0+cu121.html
   # scvi environment (totalVI, trained on the CPU): scvi-tools 1.5.1
   conda create -n txcellnet-scvi python=3.12 && conda activate txcellnet-scvi
   pip install torch==2.8.0 && pip install -r envs/requirements-scvi.txt && pip install -e .

The official repositories go into ``external/`` (or set the environment variables to another location):

.. list-table::
   :header-rows: 1

   * - Method
     - Code
     - Variable (default)
   * - NicheTrans
     - https://github.com/YSTLab/NicheTrans (commit 85fab47)
     - ``NICHETRANS_DIR`` (``external/NicheTrans``)
   * - DGAT
     - https://github.com/osmanbeyoglulab/DGAT (commit b0128f3)
     - ``DGAT_DIR`` (``external/DGAT``)
   * - SpaMosaic
     - https://github.com/JinmiaoChenLab/SpaMosaic (commit cc13367)
     - ``SPAMOSAIC_DIR`` (``external/SpaMosaic``)
   * - totalVI
     - scvi-tools 1.5.1 (PyPI)
     -

.. code-block:: bash

   git clone https://github.com/YSTLab/NicheTrans external/NicheTrans && git -C external/NicheTrans checkout 85fab47
   git clone https://github.com/osmanbeyoglulab/DGAT external/DGAT && git -C external/DGAT checkout b0128f3
   git clone https://github.com/JinmiaoChenLab/SpaMosaic external/SpaMosaic && git -C external/SpaMosaic checkout cc13367
