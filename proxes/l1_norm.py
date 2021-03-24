import numpy as np

def prox_l1_norm(lambd, x):
    """
    Computes prox_{\lambda f}(x) for  f(x) = \|x\|_1, which is given by u_i = sign(x_i) max(0, |x_i| - \lambda)
    :param x: current iterate
    :return: [ sign(x_i) * max(0, abs(x_i) - lambd) for x_i in x ]
    """
    return np.sign(x) * np.fmax(np.zeros(x.shape), np.fabs(x) - lambd)