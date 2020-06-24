.. Astraea documentation master file, created by
   sphinx-quickstart on Thu Dec 12 16:33:47 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Astraea
===================================
*Astraea* is a package to train Random Forest (RF) models on datasets. It provides tools to train RF classifiers and regressors as well as perform simple cross-validation tests and create performance plots for the test set.

It was first developed to calculate rotation period of stars from various stellar properties provided and is intended to predict long rotation periods (e.g. those of M-dwarfs) from short TESS  lightcurves (27-day lightcurves). 

We provide access to models trained on stars from the catalog by `McQuillan et al. (2014) <https://arxiv.org/abs/1402.5694>`_. Users can predict whether rotation period can be recovered and predict recoverable rotation periods for the stars in the Kepler field by using their temperatures, colors, kinematics, and other stellar parameters. 

.. Contents:

User Guide
----------
.. toctree::
   :maxdepth: 2
   
   user/install
   user/tests
   user/api

Tutorials
---------

.. toctree::
   :maxdepth: 2

   tutorial/Tutorial

License & attribution
---------------------
Copyright 2020, Yuxi Lu.

The source code is made available under the terms of the MIT license.

If you make use of this code, please cite this package and its dependencies. You can find more information about how and what to cite in the citation documentation (not yet complete).
