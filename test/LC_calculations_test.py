import Astraea
import numpy as np

# load in data
t=np.load('./test/t.npy')
sig=np.load('./test/sig.npy')
sig_err=np.load('./test/sig_err.npy')

def test_Rvar():
    """ 
    testing Rvar for KIC 2157356
    """
    assert abs(Astraea.getRvar(sig)-42303.)/42303.<0.05, "Rvar calculation error!"

def test_LGpeaks():
    """
    testing LG_peak for KIC 2157356
    """
    Prot,peak=Astraea.getLGpeak(t,sig,sig_err)
    assert abs(Prot-13.3)/13.3 < 0.05, "Period from LG wrong!"
    assert abs(peak-0.207)/0.207 < 0.05, "LG peak from LG wrong!"

def test_flicker():
    """
    testing flicker calculation
    """
    assert abs(Astraea.getFlicker(t,sig)-12609.)/12609. < 0.05, "flicker calculation wrong!"

#def test_velocity():
    """ 
    calculating velocity calculations (v_tan and v_b)
    """
    
