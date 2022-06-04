import numpy as np
from kts.nonlin_alter import *
import pdb

def cpd_auto2(K, ncp, vmax, desc_rate=1, **kwargs):
    """Main interface
    
    Detect change points automatically selecting their number
        K       - kernel between each pair of frames in video
        ncp     - maximum ncp
        vmax    - special parameter
    Optional arguments:
        lmin     - minimum segment length
        lmax     - maximum segment length
        desc_rate - rate of descriptor sampling (vmax always corresponds to 1x)

    Note:
        - cps are always calculated in subsampled coordinates irrespective to
            desc_rate
        - lmin and m should be in agreement
    ---
    Returns: (cps, costs)
        cps   - best selected change-points
        costs - costs for 0,1,2,...,m change-points
    """
    m = ncp
    scatters=scatter_v3(K)
    # pdb.set_trace()
    (_, scores) = cpd_fast(K, m, J=scatters,backtrack=False, verbose=False,**kwargs)
    
    N = K.shape[0]
    N2 = N*desc_rate  # length of the video before subsampling

    penalties = np.zeros(m+1)
    # Prevent division by zero (in case of 0 changes)
    ncp = np.arange(1, m+1)

    # penalties[1:] = (vmax * ncp / (2.0 * N2)) * (np.log(float(N2) / ncp) + 1)
    penalties[1:] =(vmax * ncp / 2.0) * (np.log(float(N2) / ncp) + 1)

    # pdb.set_trace()

    costs = scores/float(N) + penalties
    m_best = np.argmin(costs)

    (cps, scores2) = cpd_fast(K, m_best,J=scatters,verbose=False,**kwargs)

    return (cps, costs)

