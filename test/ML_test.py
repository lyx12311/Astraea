import Astraea
import sys
import pandas as pd
from sklearn.datasets import make_classification
import sklearn.metrics as metrics
import numpy as np

def test_classifier():
    """ 
    testing the classifier trainer
    """
    X, y = make_classification(n_samples=1000,n_features=4, n_informative=2, n_redundant=0,random_state=0, shuffle=False)
    X_y=pd.DataFrame(np.hstack((X,np.reshape(y,(1000,1)))),columns=np.append(['X'+str(i) for i in range(np.shape(X)[1])],['y']))
    regr,regr_outs=Astraea.RFclassifier(X_y,['X'+str(i) for i in range(np.shape(X)[1])],target_var='y',n_jobs=1)
    probs = regr.predict_proba(regr_outs.X_test)
    preds = probs[:,1]
    fpr, tpr, threshold = metrics.roc_curve(regr_outs.y_test, preds)
    roc_auc = metrics.auc(fpr, tpr)
    assert roc_auc > 0.98, "classification accuracy is low!"

def test_regressor():
    """
    testing the regressor
    """
    # create random feature matrix with 20 features and 5000 total data points
    X=np.random.rand(5000,20)

    # create labels from features
    y=sum([X[:,i]*(20-i) for i in range(np.shape(X)[1])])

    # put features and labels into one pandas dataFrame
    X_y=pd.DataFrame(np.hstack((X,np.reshape(y,(5000,1)))),columns=np.append(['X'+str(i) for i in range(np.shape(X)[1])],['y']))

    # assign random errors
    X_y['y_err']=np.random.rand(5000)
    
    regr,regr_outs=Astraea.RFregressor(X_y,['X'+str(i) for i in range(np.shape(X)[1])],target_var='y',target_var_err='y_err',n_estimators=3)
    assert regr_outs['ave_chi2']<1500, "regression accuracy is low! run test again. If still fails then something is wrong!"

    
    

