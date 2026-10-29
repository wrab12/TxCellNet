API
===

The ``txcellnet`` package holds the networks, whole-section inference and the cell-level metrics; the scripts in
``scripts/`` build on it. Every name below is importable from ``txcellnet``.

.. currentmodule:: txcellnet

Stage 1
-------

``txcellnet.model``. Every molecule is replaced by a learnable embedding of its gene and the embeddings are summed at the
molecule's pixel (gene field), together with log(1 + molecule count). DAPI and the gene field enter a U-Net (widths 64,
128, 256, 512). A cell-neighborhood transformer pools the U-Net features and the gene field into one token per cell,
exchanges information between the cells of the window with distance-biased attention (2 layers, 4 heads, length scales
initialized to 6, 12, 24 and 48 px) and adds each cell's update back to its pixels; a 1x1 convolution gives the P output
channels.

The configuration dict ``cfg`` is saved next to the weights as ``cfg.json``. Its optional keys are ``panel_genes`` (panel
matching: the model reads only these genes) and the ablations ``nodapi`` (DAPI input zeroed) and ``noniche``
(cell-neighborhood transformer removed). ``prepare(genes)`` returns the genes the model reads (their order defines the
gene ids); ``forward(x_dapi, (flat_pixel_index, gene_id, n, h, w), cell_labels)`` returns an ``(n, P, h, w)`` image.

.. autoclass:: GenNet
   :members: prepare, encode

.. autoclass:: CellNiche

.. autoclass:: UNet

   U-Net backbone of stage 1: ``cin`` input channels, ``w`` channel widths per level (two convolutions with GroupNorm and
   SiLU per block, max pooling down, transposed convolutions up).

.. autofunction:: gauss_kernel

   Normalized 1-D Gaussian kernel of radius ``int(3 * sigma + 0.5)``.

Whole-section inference
-----------------------

``txcellnet.predict``. Windows of up to 1024 px with 64-px context margins (only the 896-px core of each window is
kept), four flips averaged, per-cell sums accumulated window by window.

.. autofunction:: predict_case

Stage 2
-------

``txcellnet.flow``. A time-conditioned U-Net (FiLM blocks) learns the velocity that carries standard normal noise to the
normalized residual r = (Y - mu) / sigma_r, conditioned on inputs available for any section: the stage-1 mean mu, DAPI
and three cell geometry channels (inside-cell mask, cell boundary, distance to the boundary). Sampling takes Euler steps
from scaled noise; the sampled residual is set to zero outside tissue and its per-cell mean is removed, so each cell
keeps its stage-1 mean.

.. autofunction:: refine

.. autofunction:: load_refiner

.. autoclass:: FlowUNet

   Time-conditioned U-Net of stage 2: ``cin`` input channels (noisy residual and condition), ``cout`` output channels
   (velocity), ``w`` channel widths per level, ``td`` size of the sinusoidal time embedding.

.. autoclass:: TBlock

   Residual convolution block of :class:`FlowUNet` (``i`` input and ``o`` output channels); the time embedding (size
   ``td``) scales and shifts its features (FiLM).

.. autofunction:: geometry

.. autofunction:: sample

.. autofunction:: remove_cell_mean

Metrics
-------

``txcellnet.metrics``. Cell-level metrics of the benchmark: Spearman (primary), Pearson and the AUROC of Otsu-positive
cells.

.. autofunction:: metrics

.. autofunction:: auroc

.. autofunction:: otsu

   Otsu threshold of the values ``v`` (histogram with ``bins`` bins).

Prepared sections
-----------------

``txcellnet.io``. Reading of the prepared section format (file list in :doc:`data`).

.. autofunction:: load_case_image

.. autofunction:: label_counts

Utilities
---------

.. autofunction:: default_device

.. autofunction:: amp

.. autofunction:: log

   Print the arguments after the current time (HH:MM:SS), flushed.
