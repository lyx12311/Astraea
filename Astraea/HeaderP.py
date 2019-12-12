import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import SkyCoord
import astropy.utils as au
from astropy.io import fits
import astropy.coordinates as coord

from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.preprocessing import normalize

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
   'figure.figsize': [10,10], # instead of 4.5, 4.5
   'savefig.format': 'eps',
   'text.usetex': False,
   }
plt.rcParams.update(params)

# use to print progress bar
import time, sys
from IPython.display import clear_output
def update_progress(progress):
    bar_length = 20
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
    if progress < 0:
        progress = 0
    if progress >= 1:
        progress = 1

    block = int(round(bar_length * progress))

    clear_output(wait = True)
    text = "Progress: [{0}] {1:.1f}%".format( "#" * block + "-" * (bar_length - block), progress * 100)
    print(text)
    
# calcualte v_t, v_b by passing in a dataframe with parallax, pmra, pmdec, ra, dec
def CalcV(df):
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
	# print(vb)


# calculates chisq
def calcChi(Prot,Prot_pre,Prot_err):
    # Prot: rotation periods
    # Prot_pre: predicted rotation periods
    # Prot_err: rotation period errors
    validv=0
    for i in range(len(Prot)):
        if Prot_err[i]==0 or Prot_err[i]==np.nan:
            Prot[i]=0
            Prot_pre[i]=0
            Prot_err[i]=1
            validv=validv+1
    avstedv=sum([(Prot[i]-Prot_pre[i])**2./Prot_err[i] for i in range(len(Prot_err))])/(len(Prot_pre)-validv)
    return avstedv
    
# calculates median relative error
def MRE(Prot,Prot_pre,Prot_err):
    # Prot: rotation periods
    # Prot_pre: predicted rotation periods
    # Prot_err: rotation period errors
    validv=0
    #print(Prot-Prot_pre)
    #print(Prot)
    meree=np.median([abs(Prot[i]-Prot_pre[i])/Prot[i] for i in range(len(Prot_err))])
    return meree

# for plotting results for importance and predict vs true
def plot_result(actrualF,importance,prediction,y_test,y_test_err,topn=20):
    # inputs:
    # actrualF: feature used in training (output from my_randF_mask)
    # importance/prediction: output from my_randF function
    # topn: how many features to plot (default=20)
    # X: features, if X is inputed then plot feature vs Prot
    # y_test: tested values
    # y_test_err: tested values errors
    
    # output: 
    # my_xticks: importance of features in decending order
    
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
    plt.figure(figsize=(20,8))
    plt.subplot(1,2,1)
    plt.plot(sorted(prediction),sorted(prediction),'k-',label='y=x')
    plt.plot(sorted(prediction),sorted(1.1*prediction),'b--',label='10% Error')
    plt.plot(sorted(prediction),sorted(0.9*prediction),'b--')
    plt.plot(y_test,prediction,'r.',Markersize=3,alpha=0.2)
    plt.ylabel('Predicted Period')
    plt.xlabel('True Period')
    plt.ylim([0,max(prediction)])
    plt.xlim([0,max(prediction)])
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(sorted(prediction),sorted(prediction),'k-',label='y=x')
    plt.plot(sorted(prediction),sorted(1.1*prediction),'b--',label='10% Error')
    plt.plot(sorted(prediction),sorted(0.9*prediction),'b--')
    plt.errorbar(y_test,prediction,xerr=y_test_err,fmt='r.',Markersize=3,alpha=0.2)
    plt.ylabel('Predicted Period')
    plt.xlabel('True Period')
    plt.ylim([0,max(prediction)])
    plt.xlim([0,max(prediction)])
    plt.legend()
    #plt.savefig('RF.png')
    
    avstedv=MRE(y_test,prediction,y_test_err)
    print('Median relative error is: ',avstedv)
    return(my_xticks)



# plot different features vs Prot
def plot_corr(df,my_xticks,logplotarg=[],logarg=[]):
    # df: dataframe
    # my_xticks: features to plot against Prot
    # logplotarg: arguments to plot in loglog space
    # logarg: which log to plot
    
    # add in Prot
    Prot=df.Prot
    df=df[my_xticks].dropna()
    Prot=Prot[df.index]
    topn=len(my_xticks)
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
        featurep=df[my_xticks[i]]
        if my_xticks[i] in logplotarg:
            if logarg=='loglog':
                plt.loglog(Prot,featurep,'k.',markersize=1)
            elif logarg=='logx':
                plt.semilogx(Prot,featurep,'k.',markersize=1)
            elif logarg=='logy':
                plt.semilogy(Prot,featurep,'k.',markersize=1)
            else:
                raise SyntaxError("Log scale input not recognized!")
        else:
            plt.plot(Prot,featurep,'k.',markersize=1)
        plt.title(my_xticks[i],fontsize=25)
        stddata=np.std(featurep)
        #print([np.median(featurep)-3*stddata,np.median(featurep)+3*stddata])
        plt.ylim([np.median(featurep)-3*stddata,np.median(featurep)+3*stddata])
        plt.xlabel('Prot')
        plt.ylabel(my_xticks[i])
        #plt.tight_layout()


############################# RF training #########################################
# use only a couple of features 
def my_randF_SL(df,traind,testF,X_train_ind=[],X_test_ind=[],chisq_out=False,MREout=False,n_estimators=100, criterion='mse', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False):
    # df: dataframe to train with all the features including Prot and Prot_err
    # traind: fraction of data use to train
    # testF: training feature names
    # X_train_ind: KID for training stars
    # X_test_ind: KID for testing stars
    # chisq_out: output only median relative error?
   
    print('regr,importance,actrualF,KID_train,KID_test,predictp,avstedv,avMRE = my_randF_SL(df,traind,testF,chisq_out=0,MREout=False,hyperp=[])\n')

    if len(X_train_ind)==0:
        print('Fraction of data used to train:',traind)
    else:
        print('Training KID specified!\n')
        print('Estimated fraction of data used to train:',len(X_train_ind)/len(df['Prot']))
    print('# Features used to train:',len(testF))
    print('Features used to train:',testF)

    fl=len(df.columns) # how many features
    keys=range(fl)
    flib=dict(zip(keys, df.columns))
    
    featl_o=len(df.Prot) # old feature length before dropping
    
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
        y=df.Prot[X.index].values
        y_err=df.Prot_err[X.index].values
        KID_ar=df.KID[X.index].values
        X=X.values
	
        Ntrain = int(traind*featl)
        # Choose stars at random and split.
        shuffle_inds = np.arange(len(y))
        np.random.shuffle(shuffle_inds)
        train_inds = shuffle_inds[:Ntrain]
        test_inds = shuffle_inds[Ntrain:]
	
        y_train, y_train_err, KID_train, X_train = y[train_inds], y_err[train_inds],KID_ar[train_inds],X[train_inds, :]
        y_test, y_test_err, KID_test, X_test = y[test_inds], y_err[test_inds],KID_ar[test_inds],X[test_inds, :]
	
        test_inds,y_test, y_test_err, KID_test, X_test=zip(*sorted(zip(test_inds,y_test, y_test_err, KID_test, X_test)))
        test_inds=np.array(test_inds)
        y_test=np.array(y_test)
        y_test_err=np.array(y_test_err)
        KID_test=np.array(KID_test)
        X_test=np.asarray(X_test)
	
    else:
        datafT=df.loc[X.index].loc[df['KID'].isin(X_train_ind)]
        datafTes=df.loc[X.index].loc[df['KID'].isin(X_test_ind)]
        y_train, y_train_err,X_train = datafT.Prot.values, datafT.Prot_err.values,X.loc[df['KID'].isin(X_train_ind)].values
        y_test, y_test_err,X_test = datafTes.Prot.values, datafTes.Prot_err.values,X.loc[df['KID'].isin(X_test_ind)].values
    print(str(len(y_train))+' training stars!')



    # run random forest
    regr = RandomForestRegressor(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features, max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease, min_impurity_split=min_impurity_split, bootstrap=bootstrap, oob_score=oob_score, n_jobs=n_jobs, random_state=random_state, verbose=verbose, warm_start=warm_start)
    regr.fit(X_train, y_train)  
    
    

    # get the importance of each feature
    importance=regr.feature_importances_
    
    print('Finished training! Making predictions!')
    # make prediction
    predictp=regr.predict(X_test)
    print('Finished predicting! Calculating chi^2!')
     
    # calculate chisq and MRE
    avMRE=MRE(y_test,predictp,y_test_err)
    avstedv=calcChi(y_test,predictp,y_test_err)

    print('Median Relative Error is:',avMRE)
    print('Average Chi^2 is:',avstedv)
    
    if chisq_out:
        if MREout:
            print('Finished!')
            return avstedv,avMRE
        else:
            print('Finished!')
            return avstedv
    elif MREout:
        print('Finished!')
        return avMRE
    else:
        if len(X_train_ind)!=0:
            KID_train=datafT.KID.values
            KID_test=datafTes.KID.values
            KID_train=[int(i) for i in KID_train]
            KID_test=[int(i) for i in KID_test]
        print('Finished!')
        return regr,importance,actrualF,KID_train,KID_test,predictp,avstedv,avMRE,X_test,y_test,X_train,y_train
