def prox_l1_norm(lambd, x, var=None):
    """
    Computes prox_{\lambda f}(x) for  f(x) = \frac{1}{2}\|x\|_1, which is given by u_i = sign(x_i) max(0, |x_i| - \lambda)
    :param x: current iterate
    :return: [ sign(x_i) * max(0, abs(x_i) - lambd) for x_i in x ]
    """
    # TODO variance must be matrix
    # TODO if var is None assume that we have unit variance
    import numpy as np   
    return np.sign(x) * np.fmax(np.zeros(len(x)), np.fabs(x) - lambd)