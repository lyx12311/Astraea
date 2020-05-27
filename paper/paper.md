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
    affiliation: "1, 2, 3"
affiliations:
 - name: Department of Astronomy, Columbia University, New Yrok, NY, 10027, USA
   index: 1
 - name: Department of Astrophysics, American Museum of Natural History, New York, NY, 10024, USA
   index: 2
 - name: Center for Computational Astrophysics, Flatiron Institute, New York, NY, 10010, USA
   index: 3
aas-doi: 
aas-jornal: 
date: 21 May 2020
bibliography: paper.bib
---

# Summary

``Astraea`` is a tool to predict long stellar rotation periods without requiring long time-series light curves. It uses Random Forest (RF), a classical machine learning algorithm suitable to learn complex non-linear relations between different stellar properties. RF combines multiple decision trees, also a classical machine learning algorithm that uses ``features`` (e.g. stellar properties) to split the data set into subsets and fit piecewise smooth functions to predict the ``label`` (e.g. rotation periods). By averaging the prediction results from multiple Decision Trees that fit different subsets of the data, Random Forest can be used to predict ``labels`` without over-fitting the data.

Unlike traditional methods (e.g. Lomb-Scargle [@Reinhold2015], Auto-correlation Functions (ACF) [@McQuillan2014] and Gaussian processes [@Angus2018; @ForemanMackey2017]), whose accuracy rely on how many revolutions the star has gone through in the time-span of the light curve, ``Astraea`` models the complex relations between stellar properties (which can be obtained from sources other than the light curves) and rotation periods. 

Using ``Astraea``, user can either train their own model or use our built-in model trained on stars from the rotation period catalog by McQuillian et all. (2014) and Gaia [@Prusti2016; @Brown2018] stellar parameters. The built-in model first determines whether the rotation period is ``measurable`` using a RF classifier. If the rotation period is ``measurable``, the model will then predict the rotation period using the trained RF regressor. Currently, the built-in model works best for stars in the Kepler field [@Borucki2010] by using their temperatures, colors, kinematics, etc. However, one can easily re-train the model on other surveys. We have tested the built-in model on TESS stars [@TESS] in Lu et all. 2020. and it exhibits promising results.

``Astraea`` is built on the Random Forest models in Python scikit-learn package [@scikit-learn]. Development of ``Astraea`` happens on GitHub and any issues can be raised there.

# References
