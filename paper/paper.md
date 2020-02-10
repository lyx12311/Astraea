---
title: "Astraea: A Python package for predicting rotation periods from Kepler/TESS light curves"
tags:
  - Python
  - astronomy
  - stellar astrophysics
  - rotation periods
authors:
  - name: Yuxi (Lucy) Lu
    orcid: 0000-0003-4769-3273
    affiliation: "1" 
  - name: Ruth Angus
    orcid: 0000-0003-4540-5661
    affiliation: "2, 3"
affiliations:
 - name: Department of Astronomy, Columbia University, New Yrok, NY, 10027, USA
   index: 1
 - name: Department of Astrophysics, American Museum of Natural History, New York, NY, 10024, USA
   index: 2
 - name: Center for Computational Astrophysics, Flatiron Institute, New York, NY, 10010, USA
   index: 3
aas-doi: 
aas-jornal: 
date: 10 February 2020
bibliography: paper.bib
---

# Summary
Most of the fundamental properties of a star, for example, mass, temperature, age etc., are closely related through complicated physical mechanisms that can be fairly well described via stellar evolution models.
Although well understood between most of the fundamental properties, the relation between stellar rotation and other stellar properties are yet to be clear.
Study rotation periods is not only useful for understanding intrinsic physical connections between stellar properties but also important for modeling the radial velocity variations that are caused by star spots.
Also, using gyrochronology, one can learn the age of a main-sequence star from its rotation period [@Barnes2003; @Barnes2007] and therefore probe the evolution state of exoplanets within the system.
The most common tools used to measure rotation periods are periodograms (e.g. Lomb-Scargle) [@Reinhold2015], Auto-correlation Functions (ACF) [@McQuillan2014] and Gaussian processes [@Angus2018].
These methods typically require the observed light curve to contain data for more than one rotation period of the star in order to get a more accurate estimate.
Although precisely measured for stars that shows periodic signals in the Kepler Input Catalog (KIC) [@Borucki2010], rotation periods for stars in the Transiting Exoplanet Survey Satellite (TESS) Input Catalog (TIC), especially those with only 27 days of observations in every one-year period (most stars being observed are located in these zones) [@TESS], are extremely hard to measure.

``Astraea`` is a tool to predict rotation periods without needing long time-series observations using Random Forest, a machine learning algorithm that combines multiple decision trees to prevent over-fitting and a suitable algorithm to learn complex non-linear relations between different stellar properties.
User can either train their own model or use our model that trained on stars from the catalog by McQuillian et all. (2014).
User can predict whether the rotation period can be recovered and measure recoverable rotation periods for the stars in the Kepler or TESS field by using their temperatures, colors, kinematics, etc.

``Astraea`` is built on the Random Forest models in Python scikit-learn package [@scikit-learn]. 
Development of ``Astraea`` happens on GitHub and any issues can be raisedthere.

# References
