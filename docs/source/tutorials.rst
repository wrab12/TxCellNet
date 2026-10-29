Tutorials
=========

Each tutorial is a Jupyter notebook in ``docs/source/tutorials`` of the repository, stored with the outputs of a real
run. Install TxCellNet first (see :doc:`installation`).

* **Quick start on synthetic data** (CPU, about a minute): build small synthetic sections, train stage 1 briefly,
  predict a whole section, refine it with stage 2, check that every cell mean is preserved, compute cell-level metrics
  and compare the truth with the stage-1 and stage-2 images.
* **Kidney training and prediction** (GPU, hours): prepare the two public 10x Genomics kidney Xenium Protein sections,
  train one clean stage-1 network on the training stripes of both, predict the held-out stripes, score every protein,
  train stage 2 and compare measured, stage-1 and stage-2 images of a held-out window.

.. toctree::
   :maxdepth: 1

   tutorials/Tutorial_1_quickstart
   tutorials/Tutorial_2_kidney
