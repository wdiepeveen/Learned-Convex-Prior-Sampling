import numpy as np


def find_T(ar):
    """
    :param: ar = 1D numpy array 
    """
    for i in range(len(ar)):
        if ar[i] < 0.1:
            return i
    return len(ar)


def sample_autocorrelation_time(ests, C_0):
    """
    :param: ests = 1D numpy array of scalar estimates of phi
    :param: C_0 = covariance of the chain's starting point
    """
    n = len(ests)
    phi_bar = np.mean(ests)
    ests_centered = ests - phi_bar

    rho_hats = np.array([ np.dot(ests_centered[:-t], ests_centered[t:]) for t in range(1,len(ests)) ]) / C_0
    T = find_T(rho_hats)
    return 1 + 2 * np.sum( rho_hats[:T] )


def gelman_rubin_ratio(ests):
    """
    :param: ests = n * m numpy array of scalar estimates of phi, from m parallel chains
    """
    n, m = ests.shape

    phi_bar_js = np.mean(ests, axis = 1)
    phi_bar = np.mean(phi_bar_js)

    s_j_sqrds = np.sum( (ests - phi_bar_js)**2, axis = 1) / (n - 1)

    B = n * np.sum( (phi_bar_js - phi_bar )**2 ) / (m - 1)
    W = np.sum( s_j_sqrds ) / m

    var_hat = ( (n - 1) * W / n ) + ( B / n )

    return np.sqrt( var_hat / W )