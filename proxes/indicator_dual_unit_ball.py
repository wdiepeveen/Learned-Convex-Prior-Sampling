import numpy as np

def prox_indicator_dual_unit_ball(lambd, y, alpha=1):
    den = np.maximum(np.abs(y)/alpha, np.ones(y.shape))

    return y / (den)