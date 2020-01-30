import Astraea
import sys
import pandas as pd

def test_download():
    """
    make sure all the downloads still work
    """
    KeplerTest=Astraea.load_KeplerTest()
    assert len(KeplerTest)!=0, "Can not download Kepler testing data!"
    RF_class,RF_regr_1,RF_regr_100=Astraea.load_RF()
    assert len(RF_class), "Classifier download failed!"
    assert len(RF_regr_1), "Regressor with 1 estimator download failed!"
    assert len(RF_regr_100), "Regressor with 100 estimator download failed!"
    Astraea.FLICKERinstall()
    import FLICKER
    assert 'FLICKER' in sys.modules, "Could not install FLICKER!"

def test_predictor():
    """
    make sure you can get all periods for the testing stars
    """
    KeplerTest=Astraea.load_KeplerTest()
    starStat={'LG_peaks':KeplerTest.LG_peaks.values,'Rvar':KeplerTest.Rvar.values,'parallax':KeplerTest.parallax.values,
          'radius_percentile_lower':KeplerTest.radius_percentile_lower.values,'radius_val':KeplerTest.radius_val.values,
          'radius_percentile_upper':KeplerTest.radius_percentile_upper.values,'phot_g_mean_flux_over_error':KeplerTest.phot_g_mean_flux_over_error.values,
          'bp_g':KeplerTest.bp_g.values,'teff':KeplerTest.teff.values,'lum_val':KeplerTest.lum_val.values,'v_tan':KeplerTest.v_tan.values,
          'v_b':KeplerTest.v_b.values,'b':KeplerTest.b.values,'flicker':KeplerTest.flicker.values,'Prot':KeplerTest.Prot.values,'Prot_err':KeplerTest.Prot_err.values}
    star_data=pd.DataFrame(starStat)
    predics=Astraea.getKeplerProt(star_data)
    assert len(predics)==205, "Cannot predict rotation periods for all stars! Classification failed!"
    # chisq for 1 estimator
    chisq1=sum((predics['True Prot']-predics['Prot prediction w/ 1 est'])**2/predics['True Prot_err'])/len(predics)
    assert chisq1<1000, "Regressor for 1 estimator does not work!"
    # chisq for 100 estimator
    chisq100=sum((predics['True Prot']-predics['Prot prediction w/ 100 est'])**2/predics['True Prot_err'])/len(predics)
    assert chisq100<1000, "Regressor for 100 estimator does not work!"

