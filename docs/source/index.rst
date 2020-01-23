.. Astraea documentation master file, created by
   sphinx-quickstart on Thu Dec 12 16:33:47 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Astraea
===================================
*Astraea* is a package to train Random Forest (RF) models on datasets. It provides tools to train RF classifiers and regressors as well as perform simple cross-validation tests and performance plots on the test set.

It was first developed to calculate rotation period of stars from various stellar properties provided and is intended to predict long rotation periods (e.g. that of M-dwarfs) from short TESS  lightcurves (27-day lightcurves). 

We provide access to trained models on stars from the catalog by `McQuillian et all. <https://arxiv.org/abs/1402.5694>`. User can predict whether the rotation period can be recovered and measure recoverable rotation periods for the stars in the Kepler field by using their temperatures, colors, kinematics, etc. 

Example usage
----------

::

    import Astraea
    import pandas as pd
    import numpy as np
    
    # create random feature matrix with 20 features and 5000 total data points
    X=np.random.rand(5000,20)

    # create labels from features
    y=sum([X[:,i]*np.random.rand(1) for i in range(np.shape(X)[1])])

    # put features and labels into one pandas dataFrame
    X_y=pd.DataFrame(np.hstack((X,np.reshape(y,(5000,1)))),columns=np.append(['X'+str(i) for i in range(np.shape(X)[1])],['y']))

    # assign random errors
    X_y['y_err']=np.random.rand(5000)

    # train the model with default settings
    regr,regr_outs=Astraea.RFregressor(X_y,['X'+str(i) for i in range(np.shape(X)[1])],target_var='y',target_var_err='y_err',n_estimators=3)

    >> Simpliest example:
    >>  regr,importance,actrualF,ID_train,ID_test,predictp,ave_chi,MRE_val,X_test,y_test,X_train,y_train = RFregressor(df,testF)

    >> Fraction of data used to train: 0.8
    >> # of Features attempt to train: 20
    >> Features attempt to train: ['X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11', 'X12', 'X13', 'X14', 'X15', 'X16', 'X17', 'X18', 'X19']
    >> 5000 stars in dataframe!
    >> 5000 total stars used for RF!
    >> 4000 training stars!
    >> Finished training! Making predictions!
    >> Finished predicting! Calculating statistics!
    >> Median Relative Error is: 0.06573519959935004
    >> Average chi^2 is: 1.5145545959431705
    >> Finished!


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

   tutorials/Tutorial

License & attribution
---------------------

