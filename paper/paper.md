---
title: "Astraea: A Python package for predicting long rotation periods from Kepler/TESS light curves"
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
The rotation periods of planet hosting stars are useful for a number of reasons.
For example, they can be used for modeling and mitigating the impact of magnetic activity in radial velocity measurements and can help constrain the high-energy flux environment and ‘space weather’ of planetary systems.
Also, using gyrochronology, one can learn the age of a main-sequence star from its rotation period [@Barnes2003; @Barnes2007] and therefore probe the evolution state of exoplanets within the system.
The most common tools used to measure rotation periods are periodograms (e.g. Lomb-Scargle) [@Reinhold2015], Auto-correlation Functions (ACF) [@McQuillan2014] and Gaussian processes [@Angus2018, @ForemanMackey2017].
These methods typically require the observed light curve to contain continuous data for more than one rotation period of the star in order to get an accurate estimate.
Although they can be measured precisely for stars that show periodic signals, that were observed by Kepler [@Borucki2010],, long rotation periods for stars observed by TESS, especially those with only 27 days of observations per year (most stars being observed are located in these zones, [@TESS]), are extremely hard to measure directly.
Even worse, low-mass stars (e.g. M-dwarf stars) usually have long rotation periods ($>$ 25-30 days) [@McQuillan2014].

``Astraea`` is a tool to predict rotation periods without needing long time-series observations using Random Forest, a machine learning algorithm that combines multiple decision trees to prevent over-fitting and a suitable algorithm to learn complex non-linear relations between different stellar properties.
User can either train their own model or use our model that trained on stars from the catalog by McQuillian et all. (2014).
User can predict whether the rotation period can be predicted and predict rotation periods for the stars in the Kepler or TESS field by using their temperatures, colors, kinematics, etc.

``Astraea`` is built on the Random Forest models in Python scikit-learn package [@scikit-learn]. 
Development of ``Astraea`` happens on GitHub and any issues can be raisedthere.

# References
