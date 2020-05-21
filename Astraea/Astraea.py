import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os 

from astropy import units as u
from astropy.coordinates import SkyCoord
import astropy.utils as au
import astropy.coordinates as coord

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
#from sklearn.externals import joblib
#import joblib
    
"""--------------------------------------------- start of function to download/load RF ---------------------------------------------"""
def download_RF_class(url='https://zenodo.org/record/3838683/files/RF_Class_model.sav?download=1'):
    os.system('wget '+url)
    os.system('mv RF_Class_model.sav?download=1 ./data/RF_Class_model.sav')
   
def download_RF_regr_1est(url='https://zenodo.org/record/3838683/files/RF_Regre_model_1est_flicker.sav?download=1'):
    os.system('wget '+url)
    os.system('mv RF_Regre_model_1est_flicker.sav?download=1 ./data/RF_Regre_model_1est_flicker.sav')

def download_RF_regr_100est(url='https://zenodo.org/record/3838683/files/RF_Regre_model_100est_flicker.sav?download=1'):
    os.system('wget '+url)
    os.system('mv RF_Regre_model_100est_flicker.sav?download=1 ./data/RF_Regre_model_100est_flicker.sav')


# load testing data
def load_KeplerTest():
    if not os.path.exists('./data/'):
        os.system('mkdir data')
    if not os.path.exists('./data/TestingData.pkl'):
        os.system('mkdir data')
        os.system('wget https://zenodo.org/record/3628778/files/TestingData.pkl?download=1')
        os.system('mv TestingData.pkl?download=1 ./data/TestingData.pkl')
    return pd.read_pickle('./data/TestingData.pkl')
    

def load_RF():
    """ Load random forest classifier and regressors from zendo.org. 
    
    Two regressors will be laoded, one with 1 estimator and one with 100 estimators. 1 estimator minimizes
    bias (systematic offset) with the cost of high varience (scattering) and 100 estimators minimizes varience and
    maximizes bias. If user wants to predict rotation period from Kepler light curves then it is best to use model 
    with 100 estimiators and user should use model with 1 estimator otherwise. Model trained on TESS light curves are
    still being developed. 
   
    
    """
    if not os.path.exists('./data/'):
        os.system('mkdir data')
    if not os.path.exists('./data/RF_Class_model.sav'):
        print('downloading classifier!')
        download_RF_class()
    if not os.path.exists('./data/RF_Regre_model_1est_flicker.sav'):
        print('downloading regressor with 1 estimator!')
        download_RF_regr_1est()
    if not os.path.exists('./data/RF_Regre_model_100est_flicker.sav'):
        print('downloading regressor with 100 estimators!')
        download_RF_regr_100est()
    return pickle.load(open('./data/RF_Class_model.sav'), 'rb')),pickle.load(open('./data/RF_Regre_model_100est_flicker.sav'), 'rb')),pickle.load(open('./data/RF_Regre_model_1est_flicker.sav'), 'rb'))

"""--------------------------------------------- end of function to download/load RF ---------------------------------------------"""



"""--------------------------------------------- start of function to use trained RF to get Prot ---------------------------------------------"""
def FLICKERinstall():
    """ Installs the FLICKER software to calcualte Flicker values from light curves. Documentation: https://flicker.readthedocs.io.
    
    """
    os.system('pip install git+https://github.com/lyx12311/FLICKER.git')
    #os.system('git clone https://github.com/lyx12311/FLICKER.git')
    #os.system('cd FLICKER')
    #os.system('python setup.py install')
    #os.system('cd ..')

def getTrainF():
    """Print out needed featuers for the model in order.
    
    """
    Features_class=['LG_peaks', 'Rvar', 'parallax', 'radius_percentile_lower', 'radius_percentile_upper', 'phot_g_mean_flux_over_error', 'bp_g']
    Features_regr=['teff','bp_g','lum_val','v_tan','phot_g_mean_flux_over_error','v_b','radius_val','b','Rvar','flicker']
    print(">>> classification features are: ['LG_peaks [Lomb-Scargle peak height]', 'Rvar [ppm]', 'parallax [gaia]', 'radius_percentile_lower [gaia]', 'radius_percentile_upper [gaia]', 'phot_g_mean_flux_over_error [gaia]', 'bp_g [gaia]']\n\n")
    print(">>> regression features are: ['teff [gaia]','bp_g [gaia]','lum_val [gaia]','v_tan [getVs()]','phot_g_mean_flux_over_error [gaia]','v_b [getVs()]','radius_val [gaia]','b [gaia]','Rvar [ppm]','flicker [FLICKER]']\n\n")
    return Features_class,Features_regr

def getKeplerProt(X_pred):
    """Predict rotation period from trained models. 
    
    This function predicts rotation periods for stars in the Kepler field. The models are trained on rotation periods from 
    `McQuillian et all. (2014) <https://arxiv.org/abs/1402.5694>`_, `Santos et all. (2019) <https://arxiv.org/abs/1908.05222>`_ 
    and `Garcia et all. (2014) <https://arxiv.org/abs/1403.7155>`_. If the models are not already downloaded, this tool will download 
    the model which might take a couple of minues. It first passes the stars through a classifier, which identifies what stars 
    have measureable rotation periods. Then it uses two regressor models (one with 1 estimator and another one with 100 estimators) 
    to predict rotation periods. If column "Prot" exist, it will also output the true periods associated with the predicted periods.
    The light curve feature "flicker" can be calculated using software `FLICKER <https://flicker.readthedocs.io/en/latest/>`_.
    
    Args:
      X_pred ([Pandas DataFrame]): DataFrame contains all variables needed, run Astraea.getTrainF() to print out requirements
      
    Returns: 
      <pandas.DataFrame> or <pandas.Series>:
      :Containing:
   
        :TrueProt: True rotation period (if avaliable)
	:Prot_prediction_1est: Period predictions with 1 estimator
	:Prot_prediction_100est: Period prediction with 100 estimators
      
      
    """
    # see what features are needed
    Features_class,Features_regr=getTrainF()
    
    # check if satisfy classification features, if not, exit function
    try:
        X_pred[Features_class]
	
    except KeyError:
        print("Make sure ['LG_peaks', 'Rvar', 'parallax', 'radius_percentile_lower', 'radius_percentile_upper', 'phot_g_mean_flux_over_error', 'bp_g'] are in DataFrame for classifier!")
        print("Exiting...")
        return None
    
    # check if satisfy regressor features
    try:
        X_pred[Features_regr]
    except KeyError:
        print("Make sure ['teff','bp_g','lum_val','v_tan','phot_g_mean_flux_over_error','v_b','radius_val','b','Rvar','flicker'] are in DataFrame for regressor!")
        print("Exiting...")
        return None
    
    # load random forest models
    RF_class,RF_regr_1,RF_regr_100=load_RF()
    
    # get subset dropping NAs
    X_pred_class=X_pred[np.append(Features_regr,Features_class)].dropna()
    
    # drop duplicate columns
    X_pred_class=X_pred_class.loc[:,~X_pred_class.columns.duplicated()]
    
    # classify first...
    print("Total "+str(len(X_pred))+" stars!")
    print('Classifing '+str(len(X_pred_class))+' stars!')
    
    probs = RF_class.predict_proba(X_pred_class[Features_class])
    preds = probs[:,1]
    for i in range(len(preds)):
        if preds[i]<=0.4:
            preds[i]=0
        else:
            preds[i]=1
    print(str(sum(preds))+' stars have predictable rotation periods ('+ str(float(sum(preds))/float(len(X_pred))*100)+'%)')
    
    X_pred_class['Prot_class']=preds
    
    # pick out stars that have rotation periods
    X_pred_regr=X_pred_class[Features_regr]
    
    print("Predicting rotation periods!")
    
    
    # predict rotation periods with 1 estimator
    Prot_predictions_1est=RF_regr_1.predict(X_pred_regr.values)
    
    # predict rotation periods with 100 estimator
    Prot_predictions_100est=RF_regr_100.predict(X_pred_regr.values)
    
    print("Finished!")
    
    try:
        TrueProt=X_pred.iloc[X_pred_class.index].Prot.values
        try:
            TrueProt_err=X_pred.iloc[X_pred_class.index].Prot_err.values
            return pd.DataFrame(np.array((TrueProt,TrueProt_err, Prot_predictions_1est, Prot_predictions_100est)).T,columns=['True Prot','True Prot_err','Prot prediction w/ 1 est','Prot prediction w/ 100 est'])	
        except:
            return pd.DataFrame(np.array((TrueProt, Prot_predictions_1est, Prot_predictions_100est)).T,columns=['True Prot','Prot prediction w/ 1 est','Prot prediction w/ 100 est'])	
    except:
        print("Can't identify true rotation period! Does not exist or rename column to 'Prot'!")
        return pd.DataFrame(np.array((Prot_predictions_1est, Prot_predictions_100est)).T,columns=['Prot prediction w/ 1 est','Prot prediction w/ 100 est'])
    
    

"""--------------------------------------------- end of function to use trained RF to get Prot ---------------------------------------------"""



"""--------------------------------------------- start of calculating light curve statistics ---------------------------------------------"""
def getRvar(Flux):
    """Calculates light curve Rvar
    
    Args:
      Flux: The light curve flux in ppm
    
    Returns:
      Rvar ([float]): The variability of the light curve
    """
    return abs(np.percentile(Flux,95)-np.percentile(Flux,5))

from astropy.timeseries import LombScargle
def getLGpeak(t,sig,sig_err):
    """Calculates the location of the highest peak and the maximum power of the hightest peak
    
    Args:
      t: Time [days]
      sig: Flux
      sig_err: Flux error
    
    Returns:
      :LG_Prot ([float]): The period calculated from Lomb-Scargle
      :LG_peaks ([float]): The maximum peak height
    """
    
    # lomb-scargle
    # get frequencies
    freq = np.linspace(1./abs(t[-1]-t[0]), 1./abs(t[1]-t[0])*0.5, 100000)
    # periods 
    ps = 1./freq
    power = LombScargle(t,sig,sig_err).power(freq)
    peaks = np.array([i for i in range(1, len(ps)-1) if power[i-1]<power[i] and power[i+1] < power[i]])

    return ps[power == max(power[peaks])][0],max(power[peaks])

# calcualte v_t, v_b by passing in a dataframe with parallax, pmra, pmdec, ra, dec
def getVs(df):
    """ Calculates tangential velocity (v_tan) and vertical velocity proximation (v_b).
    
    Args:
      df ([Pandas DataFrame]): DataFrame contains columns 'parallax', 'pmra', 'pmdec', 'ra', 'dec', which are parallax, ra proper motion, dec propermotion, right ascension and declination, respectively  
    
    Returns:
      :v_t ([array-like]): Tangential velocity
      :v_b ([array-like]): Proxy for vertical velocity
    """
    d = coord.Distance(parallax=np.array(df.parallax) * u.mas,allow_negative=True)
    vra = (np.array(df.pmra)*u.mas/u.yr * d).to(u.km/u.s, u.dimensionless_angles())
    vdec = (np.array(df.pmdec)*u.mas/u.yr * d).to(u.km/u.s, u.dimensionless_angles())
    v_t=np.sqrt(np.power(vra,2.)+np.power(vdec,2.)) # vtan
    # v_b as a proxy for v_z:
    c = coord.SkyCoord(ra=np.array(df.ra)*u.deg, dec=np.array(df.dec)*u.deg, distance=d,
	              pm_ra_cosdec=np.array(df.pmra)*u.mas/u.yr,
	              pm_dec=np.array(df.pmdec)*u.mas/u.yr)
    gal = c.galactic
    v_b = (gal.pm_b * gal.distance).to(u.km/u.s, u.dimensionless_angles()) # vb
    return v_t,v_b
    
    
def getFlicker(t,sig):
    """Calculates the Flicker value
    
    Args:
      t: Time [days]
      sig: Flux
    
    Returns:
      :Flicker ([float]): Flicker value
    """
    try:
        import FLICKER
    except:
        FLICKERinstall()
        import FLICKER
    return FLICKER.Flicker(t,sig)



"""--------------------------------------------- end of calculating light curve statistics ---------------------------------------------"""
    

"""--------------------------------------------- star of functions not related to RF --------------------------------------------- """

# calculates chisq
def calcChi(TrueVal,PreVal,TrueVal_err):
    """Calculate average chisq value and ignore stars without error measurements
    
    Args:
      TrueVal ([array-like]): True values to compare to
      PreVal ([array-like]): Predicted values
      TrueVal_err ([array-like]): Errors for true values
    
    Returns:
      ave_chi ([float]): Average chisq value
    """
    validv=0
    for i in range(len(TrueVal)):
        if TrueVal_err[i]==0 or TrueVal_err[i]==np.nan:
            TrueVal[i]=0
            PreVal[i]=0
            TrueVal_err[i]=1
            validv=validv+1
    ave_chi=sum([(TrueVal[i]-PreVal[i])**2./TrueVal_err[i] for i in range(len(TrueVal_err))])/(len(PreVal)-validv)
    return ave_chi
    
# calculates median relative error
def MRE(TrueVal,PreVal,TrueVal_err=[]):
    """Calculate median relative error 
    
    Args:
      TrueVal ([array-like]): True values to compare to
      PreVal ([array-like]): Predicted values
      TrueVal_err (Optional [array-like]): Errors for true values
    
    Returns:
      meree ([float]): Median relative error
    """
    validv=0
    meree=np.median([abs(TrueVal[i]-PreVal[i])/TrueVal[i] for i in range(len(TrueVal))])
    return meree


# plot different features vs Prot
def plot_corr(df,y_vars,x_var='Prot',logplotarg=[],logarg=[],MS=1):
    """Plot correlations on one variable vs other variables specified by user
    
    Args:
         df ([Pandas DataFrame]): DataFrame contains all variables needed
      y_vars ([string list]): List of variables on y axis
      x_var (optional [string]): Value for all x axis 
      logplotarg (Optional [string list]): Variables to plot in loglog scale
      logarg (Optional [string] or [string list]): 'loglog' or 'logx' or 'logy' (default is linear). If it is a list, each argument in *logplotarg* correspond to each scale in *logarg* in order 
      MS (Optional [int]): Markersize for plotting 
    
    Returns:
      <matplotlib.plot>: plots for feature correlations
    """
    # add in Prot
    Prot=df[x_var]
    df=df[y_vars].dropna()
    Prot=Prot[df.index]
    topn=len(y_vars)
    # get subplot config
    com_mul=[] 
    # get all multiplier
    for i in range(1,topn):
        if float(topn)/float(i)-int(float(topn)/float(i))==0:
            com_mul.append(i)
        
    # total rows and columns
    col=int(np.median(com_mul))
    row=int(topn/col)
    if col*row<topn:
        if col<row:
            row=row+1
        else:
            col=col+1
        
    # plot feature vs Prot
    plt.figure(figsize=(int(topn*2.5),int(topn*2.5)))
    for i in range(topn):
        plt.subplot(row,col,i+1)
        featurep=df[y_vars[i]]
        if len(logarg)==1:
            if y_vars[i] in logplotarg:
                if logarg=='loglog':
                    plt.loglog(Prot,featurep,'k.',markersize=MS)
                elif logarg=='logx':
                    plt.semilogx(Prot,featurep,'k.',markersize=MS)
                elif logarg=='logy':
                    plt.semilogy(Prot,featurep,'k.',markersize=MS)
            else:
                plt.plot(Prot,featurep,'k.',markersize=MS)
        else:
            if y_vars[i] in logplotarg:
                logsca=logarg[logplotarg.index(y_vars[i])]
                if logsca=='loglog':
                    plt.loglog(Prot,featurep,'k.',markersize=MS)
                elif logsca=='logx':
                    plt.semilogx(Prot,featurep,'k.',markersize=MS)
                elif logsca=='logy':
                    plt.semilogy(Prot,featurep,'k.',markersize=MS)
            else:
                plt.plot(Prot,featurep,'k.',markersize=MS)
		    
        plt.title(y_vars[i],fontsize=25)
        stddata=np.std(featurep)
        plt.ylim([np.median(featurep)-3*stddata,np.median(featurep)+3*stddata])
        plt.xlabel(x_var)
        plt.ylabel(y_vars[i])
	
"""--------------------------------------------- end of functions not related to RF --------------------------------------------- """




"""--------------------------------------------- RF training and results --------------------------------------------- """
 
# RF classifier 
def RFclassifier(df,testF,modelout=False,traind=0.8,ID_on='KID',X_train_ind=[],X_test_ind=[],target_var='Prot_flag',n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight=None):
    """Train RF classifier model and predict values for cross-validation dataset. 
    
    It uses scikit-learn Random Forest classifier model. All default hyper-parameters are taken from the scikit-learn model that user can change by adding in optional inputs. More details on hyper-parameters, see https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html. To use the module to train a RF model to predict rotation period, input a pandas dataFrame with column names as well as a list of attribute names. 
    
    Args:
      df ([Pandas DataFrame]): DataFrame contains all variables needed
      testF ([string list]): List of feature names used to train
      modelout (Optional [bool]): Whether to only output the trained model 
      traind (Optinal [float]): Fraction of data use to train, the rest will be used to perform cross-validation test (default 0.8)
      ID_on (Optional [string]): What is the star identifier column name (default 'KID'). If specified ID column does not exist, it will just take the index as ID
      X_train_ind (Optional [list]): List of *ID_on* for training set, if not specified, take random *traind* fraction of indexes from *ID_on* column
      X_test_ind (Optional [list]): List of *ID_on* for testing set, if not specified, take the remaining (1-*traind*) fraction of indexes from *ID_on* column that is not in the training set (*X_train_ind*)
      target_var (Optional [string]): Label column name (default 'Prot_flag')
      
    Returns: 
      <RF model>, <pandas.Series>:
      
      :regr: Sklearn RF classifier model (attributes see https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
      
      :<pandas.Series> containing:
         
        :actrualF ([string list]): Actrual features used
        :importance ([float list]): Impurity-based feature importance ordering as *actrualF*
        :ID_train ([list]): List of *ID_on* used for training set 
        :ID_test ([list]): List of *ID_on* used for testing set
        :predictp ([float list]): List of prediction on testing set
        :X_test ([matrix]): Matrix used to predict label values for testing set
        :y_test ([array-like]): Array of true label values of testing set
        :X_train ([matrix]): Matrix used to predict label values for training set
        :y_train ([array-like]): Array of true label values of training set
      
      
    """
   
    print('Simpliest example:\n regr,regr_outs = RFregressor(df,testF)\n')

    if len(X_train_ind)==0:
        print('Fraction of data used to train:',traind)
    else:
        print('Training KID specified!\n')
        print('Estimated fraction of data used to train:',float(len(X_train_ind))/float(len(df[target_var])))
    print('# of Features attempt to train:',len(testF))
    print('Features attempt to train:',testF)

    # check if there is an ID
    if ID_on not in df.columns:
        df[ID_on]=range(len(df))
        print('ID column not found, using index as ID!')	
    
    fl=len(df.columns) # how many features
    keys=range(fl)
    flib=dict(zip(keys, df.columns))
    
    featl_o=len(df[target_var]) # old feature length before dropping
    
    actrualF=[] # actrual feature used
    # fill in feature array
    lenX=0
    missingf=[]
    for i in df.columns:
        feature=df[i].values
        if (type(feature[0]) is not str) and (i in testF):
            if sum(np.isnan(feature))<0.1*featl_o:
                lenX=lenX+1
                actrualF.append(i)
            else:
                missingf.append(i)
            
    X=df[actrualF]
    X=X.replace([np.inf, -np.inf], np.nan)
    X=X.dropna()

    featl=np.shape(X)[0]
    #print(featl)
    print(str(featl_o)+' stars in dataframe!')
    if len(missingf)!=0:
        print('Missing features:',missingf)
    if (featl_o-featl)!=0:
        print('Missing '+ str(featl_o-featl)+' stars from null values in data!\n')

    print(str(featl)+' total stars used for RF!')
    

    #print(X_train_ind)

    if len(X_train_ind)==0:
        # output
        y=df[target_var][X.index].values
        ID_ar=df[ID_on][X.index].values
        X=X.values
	
        Ntrain = int(traind*featl)
        # Choose stars at random and split.
        shuffle_inds = np.arange(len(y))
        np.random.shuffle(shuffle_inds)
        train_inds = shuffle_inds[:Ntrain]
        test_inds = shuffle_inds[Ntrain:]
	
        y_train, ID_train, X_train = y[train_inds],ID_ar[train_inds],X[train_inds, :]
        y_test, ID_test, X_test = y[test_inds], ID_ar[test_inds],X[test_inds, :]
	
        test_inds,y_test, ID_test, X_test=zip(*sorted(zip(test_inds,y_test, ID_test, X_test)))
        test_inds=np.array(test_inds)
        y_test=np.array(y_test)
        ID_test=np.array(ID_test)
        X_test=np.asarray(X_test)
	
    else:
        datafT=df.loc[X.index].loc[df[ID_on].isin(X_train_ind)]
        datafTes=df.loc[X.index].loc[df[ID_on].isin(X_test_ind)]
        y_train, X_train = datafT[target_var].values, X.loc[df[ID_on].isin(X_train_ind)].values
        y_test, X_test = datafTes[target_var].values, X.loc[df[ID_on].isin(X_test_ind)].values
    print(str(len(y_train))+' training stars!')



    # run random forest
    regr = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features, max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease, min_impurity_split=min_impurity_split, bootstrap=bootstrap, oob_score=oob_score, n_jobs=n_jobs, random_state=random_state, verbose=verbose, warm_start=warm_start, class_weight=class_weight)
    regr.fit(X_train, y_train)      

    # get the importance of each feature
    importance=regr.feature_importances_
    
    print('Finished training! Making predictions!')
    # make prediction
    predictp=regr.predict(X_test)
    print('Finished predicting!')
     
    if len(X_train_ind)!=0:
        ID_train=datafT[ID_on].values
        ID_test=datafTes[ID_on].values
        ID_train=[int(i) for i in ID_train]
        ID_test=[int(i) for i in ID_test]
    print('Finished!')
    return regr,pd.Series([importance,actrualF,ID_train,ID_test,predictp,X_test,y_test,X_train,y_train],index=['importance','actrualF','ID_train','ID_test','prediction','X_test','y_test','X_train','y_train'])
	
# RF regressor	 
def RFregressor(df,testF,modelout=False,traind=0.8,ID_on='KID',X_train_ind=[],X_test_ind=[],target_var='Prot',target_var_err='Prot_err',chisq_out=False,MREout=False,n_estimators=100,
criterion='mse', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None,
bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False):
    """Train RF regression model and perform cross-validation test. 
    
    It uses scikit-learn Random Forest regressor model. All default hyper-parameters are taken from the scikit-learn model that user can change by adding in optional inputs. More details on hyper-parameters, see https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html. To use the module to train a RF model to predict rotation period, input a pandas dataFrame with column names as well as a list of attribute names. 
    
    Args:
      df ([Pandas DataFrame]): DataFrame contains all variables needed
      testF ([string list]): List of feature names used to train
      modelout (Optional [bool]): Whether to only output the trained model 
      traind (Optinal [float]): Fraction of data use to train, the rest will be used to perform cross-validation test (default 0.8)
      ID_on (Optional [string]): What is the star identifier column name (default 'KID'). If specified ID column does not exist, it will just take the index as ID
      X_train_ind (Optional [list]): List of *ID_on* for training set, if not specified, take random *traind* fraction of indexes from *ID_on* column
      X_test_ind (Optional [list]): List of *ID_on* for testing set, if not specified, take the remaining (1-*traind*) fraction of indexes from *ID_on* column that is not in the training set (*X_train_ind*)
      target_var (Optional [string]): Label column name (default 'Prot')
      target_var_err (Optional [string]): Label error column name (default 'Prot_err')
      chisq_out (optional [bool]): If true, only output average chisq value
      MREout (optional [bool]): If true, only output median relative error. If both *chisq_out* and *MREout* are true, then output only these two values
      
    Returns: 
      <RF model>, <pandas.Series>:
      
      :regr: Sklearn RF regressor model (attributes see https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
      
      :<pandas.Series> containing:
       
        :actrualF ([string list]): Actrual features used
	:importance ([float list]): Impurity-based feature importance ordering as *actrualF*
	:ID_train ([list]): List of *ID_on* used for training set 
	:ID_test ([list]): List of *ID_on* used for testing set
	:predictp ([float list]): List of prediction on testing set
	:ave_chi ([float]): Average chisq on cross-validation (testing) set
	:MRE_val ([float]): Median relative error on cross-validation (testing) set
	:X_test ([matrix]): Matrix used to predict label values for testing set
	:y_test ([array-like]): Array of true label values of testing set
	:X_train ([matrix]): Matrix used to predict label values for training set
	:y_train ([array-like]): Array of true label values of training set
      
      
    """
   
    print('Simpliest example:\n regr,regr_outs = RFregressor(df,testF)\n')

    if len(X_train_ind)==0:
        print('Fraction of data used to train:',traind)
    else:
        print('Training KID specified!\n')
        print('Estimated fraction of data used to train:',float(len(X_train_ind))/float(len(df[target_var])))
    print('# of Features attempt to train:',len(testF))
    print('Features attempt to train:',testF)

    # check if there is an ID
    if ID_on not in df.columns:
        df[ID_on]=range(len(df))
        print('ID column not found, using index as ID!')

    fl=len(df.columns) # how many features
    keys=range(fl)
    flib=dict(zip(keys, df.columns))
    
    featl_o=len(df[target_var]) # old feature length before dropping
    
    actrualF=[] # actrual feature used
    # fill in feature array
    lenX=0
    missingf=[]
    for i in df.columns:
        feature=df[i].values
        if (type(feature[0]) is not str) and (i in testF):
            if sum(np.isnan(feature))<0.1*featl_o:
                lenX=lenX+1
                actrualF.append(i)
            else:
                missingf.append(i)
            
    X=df[actrualF]
    X=X.replace([np.inf, -np.inf], np.nan)
    X=X.dropna()

    featl=np.shape(X)[0]
    #print(featl)
    print(str(featl_o)+' stars in dataframe!')
    if len(missingf)!=0:
        print('Missing features:',missingf)
    if (featl_o-featl)!=0:
        print('Missing '+ str(featl_o-featl)+' stars from null values in data!\n')

    print(str(featl)+' total stars used for RF!')
    

    #print(X_train_ind)

    if len(X_train_ind)==0:
        # output
        y=df[target_var][X.index].values
        y_err=df[target_var_err][X.index].values
        ID_ar=df[ID_on][X.index].values
        X=X.values
	
        Ntrain = int(traind*featl)
        # Choose stars at random and split.
        shuffle_inds = np.arange(len(y))
        np.random.shuffle(shuffle_inds)
        train_inds = shuffle_inds[:Ntrain]
        test_inds = shuffle_inds[Ntrain:]
	
        y_train, y_train_err, ID_train, X_train = y[train_inds], y_err[train_inds],ID_ar[train_inds],X[train_inds, :]
        y_test, y_test_err, ID_test, X_test = y[test_inds], y_err[test_inds],ID_ar[test_inds],X[test_inds, :]
	
        test_inds,y_test, y_test_err, ID_test, X_test=zip(*sorted(zip(test_inds,y_test, y_test_err, ID_test, X_test)))
        test_inds=np.array(test_inds)
        y_test=np.array(y_test)
        y_test_err=np.array(y_test_err)
        ID_test=np.array(ID_test)
        X_test=np.asarray(X_test)
	
    else:
        datafT=df.loc[X.index].loc[df[ID_on].isin(X_train_ind)]
        datafTes=df.loc[X.index].loc[df[ID_on].isin(X_test_ind)]
        y_train, y_train_err,X_train = datafT[target_var].values, datafT[target_var_err].values,X.loc[df[ID_on].isin(X_train_ind)].values
        y_test, y_test_err,X_test = datafTes[target_var].values, datafTes[target_var].values,X.loc[df[ID_on].isin(X_test_ind)].values
    print(str(len(y_train))+' training stars!')



    # run random forest
    regr = RandomForestRegressor(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features, max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease, min_impurity_split=min_impurity_split, bootstrap=bootstrap, oob_score=oob_score, n_jobs=n_jobs, random_state=random_state, verbose=verbose, warm_start=warm_start)
    regr.fit(X_train, y_train)  
    
    

    # get the importance of each feature
    importance=regr.feature_importances_
    
    print('Finished training! Making predictions!')
    # make prediction
    predictp=regr.predict(X_test)
    print('Finished predicting! Calculating statistics!')
     
    # calculate chisq and MRE
    MRE_val=MRE(y_test,predictp,y_test_err)
    ave_chi=calcChi(y_test,predictp,y_test_err)

    print('Median Relative Error is:',MRE_val)
    print('Average chi^2 is:',ave_chi)
    
    if chisq_out:
        if MREout:
            print('Finished!')
            return ave_chi,MRE_val
        else:
            print('Finished!')
            return ave_chi
    elif MREout:
        print('Finished!')
        return MRE_val
    else:
        if len(X_train_ind)!=0:
            ID_train=datafT[ID_on].values
            ID_test=datafTes[ID_on].values
            ID_train=[int(i) for i in ID_train]
            ID_test=[int(i) for i in ID_test]
        print('Finished!')
        return regr,pd.Series([importance,actrualF,ID_train,ID_test,predictp,ave_chi,MRE_val,X_test,y_test,X_train,y_train],index=['importance','actrualF','ID_train','ID_test','prediction','ave_chi2','MRE','X_test','y_test','X_train','y_train'])


# for plotting results for importance and predict vs true
def plot_result(actrualF,importance,prediction,y_test,y_test_err=[],topn=20,MS=3,labelName='Period'):
    """Plot impurity-based feature importance as well as predicted values vs true values for a random forest model
    
    Args:
      actrualF ([array-like]): Feature used (from function output of RFregressor())
      importance ([array-like]): importance of the model (from function output of RFregressor())
      prediction ([array-like]): Predicted values (from function output of RFregressor())
      y_test ([array-like]): true values (from function output of RFregressor())
      y_test_err (Optional [array-like]): Errors for true values (from function output of RFregressor())
      topn (Optional [int]): How many most important features to plot
      MS (Optional [int]): Markersize for plotting true vs predicted values
      labelName (Optional [string]): Label name
    
    Returns:
      <matplotlib.plot>: importance plot as well as true vs prediction plot
    """
    
    plt.rcParams.keys()
    plt.rc('font', family='serif')
    params = {
   'axes.labelsize': 30,
   'axes.linewidth': 1.5,
   'legend.fontsize': 25,
   'legend.frameon': False,
   'lines.linewidth': 2,
   'xtick.direction': 'in',
   'xtick.labelsize': 25,
   'xtick.major.bottom': True,
   'xtick.major.pad': 10,
   'xtick.major.size': 10,
   'xtick.major.width': 1,
   'xtick.minor.bottom': True,
   'xtick.minor.pad': 3.5,
   'xtick.minor.size': 5,
   'xtick.minor.top': True,
   'xtick.minor.visible': True,
   'xtick.minor.width': 1,
   'xtick.top': True,
   'ytick.direction': 'in',
   'ytick.labelsize': 25,
   'ytick.major.pad': 10,
   'ytick.major.size': 10,
   'ytick.major.width': 1,
   'ytick.minor.pad': 3.5,
   'ytick.minor.size': 5,
   'ytick.minor.visible': True,
   'ytick.minor.width': 1,
   'ytick.right': True,
    }
    plt.rcParams.update(params)

    topn=min([topn,len(actrualF)])
    # zip the importance with its feature name
    list1 = list(zip(actrualF,importance))
    # sort the zipped list
    decend=sorted(list1, key=lambda x:x[1],reverse=True)
    #print(decend)

    # how many features to plot 
    x=range(topn)
    
    ####################  get most important features ############################################################
    y_val=[decend[i][1] for i in range(topn)]
    my_xticks=[decend[i][0] for i in range(topn)]

    plt.figure(figsize=(20,5))
    plt.title('Most important features',fontsize=25)
    plt.xticks(x, my_xticks)
    plt.plot(x, y_val,'k-')
    plt.xlim([min(x),max(x)])
    plt.xticks(rotation=90)
    plt.ylabel('importance')
    ####################  get most important features ############################################################

    # prediction vs true
    if len(y_test_err)==0:
        plt.figure(figsize=(10,8))
        plt.plot(sorted(prediction),sorted(prediction),'k-',label='y=x')
        plt.plot(sorted(prediction),sorted(1.1*prediction),'b--',label='10% Error')
        plt.plot(sorted(prediction),sorted(0.9*prediction),'b--')
        plt.plot(y_test,prediction,'r.',Markersize=MS,alpha=0.2)
        plt.ylabel('Predicted '+labelName)
        plt.xlabel('True '+labelName)
        plt.ylim([min(prediction),max(prediction)])
        plt.xlim([min(prediction),max(prediction)])
        plt.legend()
    else:
        plt.figure(figsize=(20,8))
        plt.subplot(1,2,1)
        plt.plot(sorted(prediction),sorted(prediction),'k-',label='y=x')
        plt.plot(sorted(prediction),sorted(1.1*prediction),'b--',label='10% Error')
        plt.plot(sorted(prediction),sorted(0.9*prediction),'b--')
        plt.plot(y_test,prediction,'r.',Markersize=MS,alpha=0.2)
        plt.ylabel('Predicted '+labelName)
        plt.xlabel('True Period')
        plt.ylim([min(prediction),max(prediction)])
        plt.xlim([min(prediction),max(prediction)])
        plt.legend()
        plt.subplot(1,2,2)
        plt.plot(sorted(prediction),sorted(prediction),'k-',label='y=x')
        plt.plot(sorted(prediction),sorted(1.1*prediction),'b--',label='10% Error')
        plt.plot(sorted(prediction),sorted(0.9*prediction),'b--')
        plt.errorbar(y_test,prediction,xerr=y_test_err,fmt='r.',Markersize=MS,alpha=0.2)
        plt.ylabel('Predicted '+labelName)
        plt.xlabel('True '+labelName)
        plt.ylim([min(prediction),max(prediction)])
        plt.xlim([min(prediction),max(prediction)])
        plt.legend()
    
        avstedv=MRE(y_test,prediction,y_test_err)
        print('Median relative error is: ',avstedv)
    
