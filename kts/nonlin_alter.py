import numpy as np


def kernel(X1,X2,n1):
    n=X1.shape[0]
    K=np.zeros((n,n1))
    for i in range(n):
        K[i,:min(n1,n-i)]=np.dot(X1[i,:],X2[:,i:min(i+n1,n)])
    return K

def scatter_v3(K):
    n, n1 = K.shape
    K1 = np.cumsum([0] + list(K[:, 0]))
    K2 = np.zeros((n1,n))
    K2[0]=K[:, 0]

    # Triangle computing
    for i in range(1, n1):
        l = n - i  # The number of triangles
        for j in range(l):
            if i == 1:
                K2[i][j] = K2[i - 1][j] + K2[i - 1][j + 1] + 2 * K[j][i]
            else:
                K2[i][j] = K2[i - 1][j] + K2[i - 1][j + 1] + 2 * K[j][i] - K2[i - 2][j + 1]

    scatters = np.zeros((n, n1))
    for i in range(n):
        for j in range(i, min(i + n1, n)):
            depth = j - i
            scatters[i, depth] = K1[j + 1] - K1[i] - (K2[depth][i]) / (depth+1)

    return scatters

def cpd_fast(K, ncp, J,lmin=1, backtrack=True, verbose=True):
    """ Change point detection with dynamic programming
    K - tailored kernel matrix
    ncp - number of change points to detect (ncp >= 0)
    J - precomputed scatter matrix
    lmin - minimal length of a segment
    backtrack - when False - only evaluate objective scores (to save memory)

    Returns: (cps, obj)
        cps - detected array of change points: mean is thought to be constant on [ cps[i], cps[i+1] )
        obj_vals - values of the objective function for 0..m changepoints

    """
    m = int(ncp)  # prevent numpy.int64

    (n, n1) = K.shape  # n - number of frames; n1 - maximal frames in a segment

    assert (n >= (m + 1) * lmin)
    assert (n <= (m + 1) * n1)
    assert (n1 >= lmin >= 1)

    if verbose:
        print("Inferring best change points...")

    # dp[k, l] - value of the objective for k change-points and l first frames
    dp = 1e101 * np.ones((m + 1, n + 1))
    dp[0, lmin:n1+1] = J[0, lmin-1:n1]

    if backtrack:
        # p[k, l] --- "previous change" --- best t_k when t_k+1 == l
        p = np.zeros((m + 1, n + 1), dtype=int)
    else:
        p = np.zeros((1, 1), dtype=int)

    for k in range(1, m + 1):
        for l in range((k + 1) * lmin, min((k+1)*n1,n)+1):
            for t in range(max(k * lmin, l - n1), min(k*n1,l - lmin)+ 1):
                c = dp[k - 1, t] + J[t, l-t-1]
                if c < dp[k, l]:
                    dp[k, l] = c
                    if backtrack:
                        p[k, l] = t

    # Collect change points
    cps = np.zeros(m, dtype=int)

    if backtrack:
        cur = n
        for k in range(m, 0, -1):
            cps[k - 1] = p[k, cur]
            cur = cps[k - 1]

    scores = dp[:, n].copy()
    scores[scores > 1e99] = np.inf

    # dpt=ed-st
    return cps, scores

