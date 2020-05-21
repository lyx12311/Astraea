Astraea
====================================
.. image:: https://zenodo.org/badge/227692575.svg
   :target: https://zenodo.org/badge/latestdoi/227692575

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3628778.svg
   :target: https://doi.org/10.5281/zenodo.3628778

.. image:: https://travis-ci.com/lyx12311/Astraea.svg?branch=master
   :target: https://travis-ci.com/lyx12311/Astraea
   
.. image:: https://readthedocs.org/projects/stardate/badge/?version=latest
    :target: https://Astraea.readthedocs.io/en/latest/?badge=latest

*Astraea* is a package to train Random Forest (RF) models on datasets. It provides tools to train RF classifiers and regressors as well as perform simple cross-validation tests and performance plots on the test set.

It was first developed to calculate rotation period of stars from various stellar properties provided and is intended to predict long rotation periods (e.g. that of M-dwarfs) from short TESS  lightcurves (27-day lightcurves). 

We provide access to trained models on stars from the catalog by `McQuillian et all. (2014) <https://arxiv.org/abs/1402.5694>`_. User can predict whether the rotation period can be recovered and measure recoverable rotation periods for the stars in the Kepler field by using their temperatures, colors, kinematics, etc. 

For user guides and tutorials, see `Astraea readthedocs <https://astraea.readthedocs.io/en/latest/?badge=latest>`_
